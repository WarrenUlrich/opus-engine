#pragma once

#include "../math/mat4.hpp"
#include "../math/vec3.hpp"
#include "../math/vec4.hpp"
#include "lighting.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace scene {

// ─── Forward+ Tiling Constants ───────────────────────────────────────────────
//
// TILE_SIZE:           Screen-space tile edge in pixels. 16 is the sweet spot —
//                      small enough for tight culling, large enough to keep
//                      the tile count manageable.
//
// FP_MAX_LIGHTS:       Hard cap on scene lights the tiling system can handle.
//
// FP_MAX_PER_TILE:     Maximum lights assigned to one tile.  If a tile
//                      gets more, extras are silently dropped.
//
// TILE_DATA_STRIDE:    Number of RGBA32F texels per tile in the packed
//                      tile-light texture.  Value layout inside one tile row:
//                          value[0]   = light count
//                          value[1…N] = light indices
//                      Packed 4 values per texel → ceil((1 + MAX) / 4).

inline constexpr int FP_TILE_SIZE      = 16;
inline constexpr int FP_MAX_LIGHTS     = 1024;
inline constexpr int FP_MAX_PER_TILE   = 64;
inline constexpr int TILE_DATA_STRIDE  = (1 + FP_MAX_PER_TILE + 3) / 4;  // 17

// ─── Light Grid ──────────────────────────────────────────────────────────────
//
// The CPU-side counterpart to the Forward+ fragment shader.
// Every frame:
//   1. cull() projects each light's bounding sphere to screen space
//   2. Determines which tiles the sphere overlaps
//   3. Packs per-tile light lists + light properties into GPU textures
//
// The renderer then binds these textures so the fragment shader can
// look up "which lights affect this pixel's tile" via texelFetch().
//
// Memory footprint (1920×1080, 16 px tiles = 120×68 = 8 160 tiles):
//   Light data texture :  1024 × 4 × 16 B  ≈  64 KB  (always allocated)
//   Tile data texture  :  2040 × 68 × 16 B ≈ 2.2 MB  (resized on window change)

class light_grid {
public:
	light_grid() = default;

	// ── Lifecycle ────────────────────────────────────────────────────

	/// Create GPU resources.  Call once after sg_setup().
	void init() {
		// Nearest-neighbor sampler (required for RGBA32F / unfilterable-float)
		sg_sampler_desc smp = {};
		smp.min_filter = SG_FILTER_NEAREST;
		smp.mag_filter = SG_FILTER_NEAREST;
		smp.wrap_u = SG_WRAP_CLAMP_TO_EDGE;
		smp.wrap_v = SG_WRAP_CLAMP_TO_EDGE;
		sampler_ = sg_make_sampler(&smp);

		// Light data texture:  RGBA32F, (FP_MAX_LIGHTS × 4)
		//   row 0 : position + type
		//   row 1 : direction + range
		//   row 2 : color + intensity
		//   row 3 : inner_cone_cos, outer_cone_cos, 0, 0
		{
			sg_image_desc d = {};
			d.width = FP_MAX_LIGHTS;
			d.height = 4;
			d.pixel_format = SG_PIXELFORMAT_RGBA32F;
			d.usage.immutable = false;
			d.usage.stream_update = true;
			d.label = "fp_light_data";
			light_data_img_ = sg_make_image(&d);

			sg_view_desc vd = {};
			vd.texture.image = light_data_img_;
			light_data_view_ = sg_make_view(&vd);
		}

		// CPU-side pixel buffer for the light data texture
		light_data_pixels_.resize(FP_MAX_LIGHTS * 4 * 4, 0.0f); // width × height × 4 channels
	}

	/// Release all GPU resources.
	void destroy() {
		if (light_data_view_.id) sg_destroy_view(light_data_view_);
		if (tile_data_view_.id) sg_destroy_view(tile_data_view_);
		if (light_data_img_.id) sg_destroy_image(light_data_img_);
		if (tile_data_img_.id) sg_destroy_image(tile_data_img_);
		if (sampler_.id) sg_destroy_sampler(sampler_);
	}

	// ── Per-Frame Culling ────────────────────────────────────────────

	/// CPU light culling.  Call once per frame before rendering.
	///
	/// Projects each light's bounding sphere onto the screen, determines
	/// which tiles it overlaps, then uploads:
	///   • All light properties  →  light_data_tex  (RGBA32F)
	///   • Per-tile light lists  →  tile_data_tex   (RGBA32F, packed)
	///
	/// @param view      Camera view matrix (world → view).
	/// @param proj      Camera projection matrix (view → clip).
	/// @param lights    Array of scene::light structs to cull.
	/// @param n         Number of lights in the array.
	/// @param screen_w  Framebuffer width  in pixels.
	/// @param screen_h  Framebuffer height in pixels.
	void cull(const math::mat4 &view, const math::mat4 &proj,
	          const light *lights, int n, int screen_w, int screen_h) {
		ensure_tile_texture(screen_w, screen_h);

		const int total_tiles = num_tiles_x_ * num_tiles_y_;
		num_lights_ = std::min(n, FP_MAX_LIGHTS);

		// ── 1. Clear tile counts + pixel buffer ──
		std::memset(tile_counts_.data(), 0, total_tiles * sizeof(int));
		std::memset(tile_data_pixels_.data(), 0, tile_data_pixels_.size() * sizeof(float));

		// ── 2. Pack light properties into CPU buffer ──
		upload_light_data(lights, num_lights_);

		// ── 3. Assign lights to tiles ──
		const float focal_x = proj.m[0]; // P[0][0]
		const float focal_y = proj.m[5]; // P[1][1]

		for (int li = 0; li < num_lights_; ++li) {
			const auto &l = lights[li];

			if (l.type == light_type::directional) {
				// Directionals have infinite range → every tile
				for (int ty = 0; ty < num_tiles_y_; ++ty)
					for (int tx = 0; tx < num_tiles_x_; ++tx)
						add_to_tile(tx, ty, li);
				continue;
			}

			// Transform light position to view space
			math::vec4 vp = view * math::vec4(l.position, 1.0f);
			float dist = -vp.z; // positive = in front of camera
			float range = l.range;

			// Frustum reject: entirely behind camera
			if (dist + range < 0.001f) continue;

			// Light encompasses the camera → assign to all tiles (conservative)
			if (dist < range) {
				for (int ty = 0; ty < num_tiles_y_; ++ty)
					for (int tx = 0; tx < num_tiles_x_; ++tx)
						add_to_tile(tx, ty, li);
				continue;
			}

			// Project sphere center to NDC
			float ndc_x = vp.x * focal_x / dist;
			float ndc_y = vp.y * focal_y / dist;

			// Conservative projected radius (use nearest sphere edge for largest projection)
			float min_dist = std::max(dist - range, 0.001f);
			float r_ndc_x = range * std::abs(focal_x) / min_dist;
			float r_ndc_y = range * std::abs(focal_y) / min_dist;

			// NDC → screen pixels
			float cx = (ndc_x + 1.0f) * 0.5f * (float)screen_w;
			float cy = (ndc_y + 1.0f) * 0.5f * (float)screen_h;
			float rx = r_ndc_x * 0.5f * (float)screen_w;
			float ry = r_ndc_y * 0.5f * (float)screen_h;

			// Determine overlapping tile range
			int min_tx = std::max(0, (int)std::floor((cx - rx) / (float)FP_TILE_SIZE));
			int max_tx = std::min(num_tiles_x_ - 1, (int)std::floor((cx + rx) / (float)FP_TILE_SIZE));
			int min_ty = std::max(0, (int)std::floor((cy - ry) / (float)FP_TILE_SIZE));
			int max_ty = std::min(num_tiles_y_ - 1, (int)std::floor((cy + ry) / (float)FP_TILE_SIZE));

			for (int ty = min_ty; ty <= max_ty; ++ty)
				for (int tx = min_tx; tx <= max_tx; ++tx)
					add_to_tile(tx, ty, li);
		}

		// ── 4. Upload tile data to GPU ──
		{
			sg_image_data d = {};
			d.mip_levels[0].ptr = tile_data_pixels_.data();
			d.mip_levels[0].size = tile_data_pixels_.size() * sizeof(float);
			sg_update_image(tile_data_img_, &d);
		}
	}

	// ── Accessors (used by the renderer for binding) ─────────────────

	[[nodiscard]] sg_view light_data_view() const { return light_data_view_; }
	[[nodiscard]] sg_view tile_data_view() const { return tile_data_view_; }
	[[nodiscard]] sg_sampler sampler() const { return sampler_; }
	[[nodiscard]] int num_tiles_x() const { return num_tiles_x_; }
	[[nodiscard]] int num_tiles_y() const { return num_tiles_y_; }
	[[nodiscard]] int num_lights() const { return num_lights_; }

private:
	// ── GPU handles ──
	sg_image light_data_img_{};
	sg_image tile_data_img_{};
	sg_view light_data_view_{};
	sg_view tile_data_view_{};
	sg_sampler sampler_{};

	// ── Tile metrics ──
	int num_tiles_x_{0};
	int num_tiles_y_{0};
	int num_lights_{0};
	int tile_tex_width_{0};

	// ── CPU-side pixel buffers ──
	std::vector<float> light_data_pixels_;                    // RGBA32F for light data
	std::vector<float> tile_data_pixels_;                     // RGBA32F for tile light lists
	std::vector<int> tile_counts_;                            // Per-tile light count

	// ── Helpers ──────────────────────────────────────────────────────

	/// Upload light property data to the light_data texture.
	void upload_light_data(const light *lights, int n) {
		std::memset(light_data_pixels_.data(), 0, light_data_pixels_.size() * sizeof(float));

		for (int i = 0; i < n; ++i) {
			const auto &l = lights[i];

			// Pixel (i, row) → offset = (row * width + i) * 4
			float *row0 = &light_data_pixels_[(0 * FP_MAX_LIGHTS + i) * 4];
			float *row1 = &light_data_pixels_[(1 * FP_MAX_LIGHTS + i) * 4];
			float *row2 = &light_data_pixels_[(2 * FP_MAX_LIGHTS + i) * 4];
			float *row3 = &light_data_pixels_[(3 * FP_MAX_LIGHTS + i) * 4];

			row0[0] = l.position.x;
			row0[1] = l.position.y;
			row0[2] = l.position.z;
			row0[3] = static_cast<float>(static_cast<int>(l.type));

			row1[0] = l.direction.x;
			row1[1] = l.direction.y;
			row1[2] = l.direction.z;
			row1[3] = l.range;

			row2[0] = l.color.x;
			row2[1] = l.color.y;
			row2[2] = l.color.z;
			row2[3] = l.intensity;

			row3[0] = l.inner_cone_cos;
			row3[1] = l.outer_cone_cos;
		}

		sg_image_data d = {};
		d.mip_levels[0].ptr = light_data_pixels_.data();
		d.mip_levels[0].size = light_data_pixels_.size() * sizeof(float);
		sg_update_image(light_data_img_, &d);
	}

	/// Ensure the tile texture matches the current screen dimensions.
	/// Recreates only when the tile count changes (e.g., window resize).
	void ensure_tile_texture(int screen_w, int screen_h) {
		int ntx = (screen_w + FP_TILE_SIZE - 1) / FP_TILE_SIZE;
		int nty = (screen_h + FP_TILE_SIZE - 1) / FP_TILE_SIZE;

		if (ntx == num_tiles_x_ && nty == num_tiles_y_) return;

		num_tiles_x_ = ntx;
		num_tiles_y_ = nty;
		tile_tex_width_ = ntx * TILE_DATA_STRIDE;

		// Tear down old texture + view
		if (tile_data_view_.id) {
			sg_destroy_view(tile_data_view_);
			tile_data_view_ = {};
		}
		if (tile_data_img_.id) {
			sg_destroy_image(tile_data_img_);
			tile_data_img_ = {};
		}

		// Create tile data texture:  RGBA32F, (tile_tex_width × num_tiles_y)
		sg_image_desc desc = {};
		desc.width = tile_tex_width_;
		desc.height = num_tiles_y_;
		desc.pixel_format = SG_PIXELFORMAT_RGBA32F;
		desc.usage.immutable = false;
		desc.usage.stream_update = true;
		desc.label = "fp_tile_data";
		tile_data_img_ = sg_make_image(&desc);

		sg_view_desc vd = {};
		vd.texture.image = tile_data_img_;
		tile_data_view_ = sg_make_view(&vd);

		// Resize CPU buffers (4 channels per texel for RGBA32F)
		tile_data_pixels_.resize(tile_tex_width_ * num_tiles_y_ * 4, 0.0f);
		tile_counts_.resize(ntx * nty, 0);
	}

	/// Add a light index to a tile's list (packed RGBA32F).
	///
	/// Per-tile value layout (linearized):
	///   value[0]     = light count
	///   value[1..N]  = light indices
	/// Packed 4 values per RGBA texel.
	void add_to_tile(int tx, int ty, int light_idx) {
		int tile_linear = ty * num_tiles_x_ + tx;
		int count = tile_counts_[tile_linear];
		if (count >= FP_MAX_PER_TILE) return;

		// Position in the per-tile value sequence
		int p = 1 + count;

		// Map (tx, ty, p) → texel (texel_x, ty) channel ch
		int texel_x = tx * TILE_DATA_STRIDE + p / 4;
		int ch = p % 4;
		int offset = (ty * tile_tex_width_ + texel_x) * 4 + ch;
		tile_data_pixels_[offset] = static_cast<float>(light_idx);

		// Update count (value position 0)
		count++;
		tile_counts_[tile_linear] = count;

		int count_texel_x = tx * TILE_DATA_STRIDE; // p=0 → texel = base, ch = 0
		int count_offset = (ty * tile_tex_width_ + count_texel_x) * 4 + 0;
		tile_data_pixels_[count_offset] = static_cast<float>(count);
	}
};

} // namespace scene
