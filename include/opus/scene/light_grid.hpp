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

// Tiling constants for Forward+ light culling.
// TILE_DATA_STRIDE: texels per tile in the packed texture.
//   value[0] = light count, value[1..N] = light indices, 4 values per texel.

inline constexpr int FP_TILE_SIZE = 16;
inline constexpr int FP_MAX_LIGHTS = 1024;
inline constexpr int FP_MAX_PER_TILE = 128;
inline constexpr int TILE_DATA_STRIDE = (1 + FP_MAX_PER_TILE + 3) / 4; // 33

// CPU-side Forward+ light culler. Projects light bounding spheres to screen space,
// bins them into tiles, and uploads per-tile light lists as GPU textures for the
// fragment shader to sample via texelFetch().

class light_grid {
public:
	light_grid() = default;

	void init() {
		sg_sampler_desc smp = {};
		smp.min_filter = SG_FILTER_NEAREST;
		smp.mag_filter = SG_FILTER_NEAREST;
		smp.wrap_u = SG_WRAP_CLAMP_TO_EDGE;
		smp.wrap_v = SG_WRAP_CLAMP_TO_EDGE;
		sampler_ = sg_make_sampler(&smp);

		// Light data texture: RGBA32F, (FP_MAX_LIGHTS x 4)
		//   row 0: position + type
		//   row 1: direction + range
		//   row 2: color + intensity
		//   row 3: inner_cone_cos, outer_cone_cos, 0, 0
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

		light_data_pixels_.resize(FP_MAX_LIGHTS * 4 * 4, 0.0f);
	}

	void destroy() {
		if (light_data_view_.id)
			sg_destroy_view(light_data_view_);
		if (tile_data_view_.id)
			sg_destroy_view(tile_data_view_);
		if (light_data_img_.id)
			sg_destroy_image(light_data_img_);
		if (tile_data_img_.id)
			sg_destroy_image(tile_data_img_);
		if (sampler_.id)
			sg_destroy_sampler(sampler_);
	}

	void cull(const math::mat4 &view, const math::mat4 &proj, const light *lights, int n,
	          int screen_w, int screen_h) {
		ensure_tile_texture(screen_w, screen_h);

		const int total_tiles = num_tiles_x_ * num_tiles_y_;
		num_lights_ = std::min(n, FP_MAX_LIGHTS);

		std::memset(tile_counts_.data(), 0, total_tiles * sizeof(int));
		std::memset(tile_data_pixels_.data(), 0, tile_data_pixels_.size() * sizeof(float));

		upload_light_data(lights, num_lights_);

		// Project each light bounding sphere to screen-space tiles
		const float focal_x = proj.m[0]; // P[0][0]
		const float focal_y = proj.m[5]; // P[1][1]

		for (int li = 0; li < num_lights_; ++li) {
			const auto &l = lights[li];

			if (l.type == light_type::directional) {
				for (int ty = 0; ty < num_tiles_y_; ++ty)
					for (int tx = 0; tx < num_tiles_x_; ++tx)
						add_to_tile(tx, ty, li);
				continue;
			}

			math::vec4 vp = view * math::vec4(l.position, 1.0f);
			float dist = -vp.z;
			float range = l.range;

			if (dist + range < 0.001f)
				continue;

			// Light encompasses camera -> assign to all tiles (conservative)
			if (dist < range) {
				for (int ty = 0; ty < num_tiles_y_; ++ty)
					for (int tx = 0; tx < num_tiles_x_; ++tx)
						add_to_tile(tx, ty, li);
				continue;
			}

			// Project the light's view-space AABB to screen tiles.
			// We project the 4 x-z and 4 y-z AABB corners at both the near
			// and far depth of the sphere to get a correct conservative bound.
			// (A simple center +/- radius in NDC is wrong for off-axis spheres
			//  because the center is projected at dist while the radius uses
			//  min_dist, producing an inconsistent screen-space rectangle.)
			float z_near = std::max(dist - range, 0.001f);
			float z_far = dist + range;

			float nx0 = (vp.x - range) * focal_x / z_far;
			float nx1 = (vp.x - range) * focal_x / z_near;
			float nx2 = (vp.x + range) * focal_x / z_far;
			float nx3 = (vp.x + range) * focal_x / z_near;
			float ndc_x_min = std::min({nx0, nx1, nx2, nx3});
			float ndc_x_max = std::max({nx0, nx1, nx2, nx3});

			float ny0 = (vp.y - range) * focal_y / z_far;
			float ny1 = (vp.y - range) * focal_y / z_near;
			float ny2 = (vp.y + range) * focal_y / z_far;
			float ny3 = (vp.y + range) * focal_y / z_near;
			float ndc_y_min = std::min({ny0, ny1, ny2, ny3});
			float ndc_y_max = std::max({ny0, ny1, ny2, ny3});

			// NDC -> screen pixels -> overlapping tile range
			float sx_min = (ndc_x_min + 1.0f) * 0.5f * (float)screen_w;
			float sx_max = (ndc_x_max + 1.0f) * 0.5f * (float)screen_w;
			float sy_min = (ndc_y_min + 1.0f) * 0.5f * (float)screen_h;
			float sy_max = (ndc_y_max + 1.0f) * 0.5f * (float)screen_h;

			int min_tx = std::max(0, (int)std::floor(sx_min / (float)FP_TILE_SIZE));
			int max_tx = std::min(num_tiles_x_ - 1, (int)std::floor(sx_max / (float)FP_TILE_SIZE));
			int min_ty = std::max(0, (int)std::floor(sy_min / (float)FP_TILE_SIZE));
			int max_ty = std::min(num_tiles_y_ - 1, (int)std::floor(sy_max / (float)FP_TILE_SIZE));

			for (int ty = min_ty; ty <= max_ty; ++ty)
				for (int tx = min_tx; tx <= max_tx; ++tx)
					add_to_tile(tx, ty, li);
		}

		{
			sg_image_data d = {};
			d.mip_levels[0].ptr = tile_data_pixels_.data();
			d.mip_levels[0].size = tile_data_pixels_.size() * sizeof(float);
			sg_update_image(tile_data_img_, &d);
		}
	}

	[[nodiscard]] sg_view light_data_view() const { return light_data_view_; }
	[[nodiscard]] sg_view tile_data_view() const { return tile_data_view_; }
	[[nodiscard]] sg_sampler sampler() const { return sampler_; }
	[[nodiscard]] int num_tiles_x() const { return num_tiles_x_; }
	[[nodiscard]] int num_tiles_y() const { return num_tiles_y_; }
	[[nodiscard]] int num_lights() const { return num_lights_; }

private:
	sg_image light_data_img_{};
	sg_image tile_data_img_{};
	sg_view light_data_view_{};
	sg_view tile_data_view_{};
	sg_sampler sampler_{};

	int num_tiles_x_{0};
	int num_tiles_y_{0};
	int num_lights_{0};
	int tile_tex_width_{0};

	std::vector<float> light_data_pixels_;
	std::vector<float> tile_data_pixels_;
	std::vector<int> tile_counts_;

	void upload_light_data(const light *lights, int n) {
		std::memset(light_data_pixels_.data(), 0, light_data_pixels_.size() * sizeof(float));

		for (int i = 0; i < n; ++i) {
			const auto &l = lights[i];

			// Pixel (i, row) -> offset = (row * width + i) * 4
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

	// Recreates tile texture when screen dimensions change.
	void ensure_tile_texture(int screen_w, int screen_h) {
		int ntx = (screen_w + FP_TILE_SIZE - 1) / FP_TILE_SIZE;
		int nty = (screen_h + FP_TILE_SIZE - 1) / FP_TILE_SIZE;

		if (ntx == num_tiles_x_ && nty == num_tiles_y_)
			return;

		num_tiles_x_ = ntx;
		num_tiles_y_ = nty;
		tile_tex_width_ = ntx * TILE_DATA_STRIDE;

		if (tile_data_view_.id) {
			sg_destroy_view(tile_data_view_);
			tile_data_view_ = {};
		}
		if (tile_data_img_.id) {
			sg_destroy_image(tile_data_img_);
			tile_data_img_ = {};
		}

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

		tile_data_pixels_.resize(tile_tex_width_ * num_tiles_y_ * 4, 0.0f);
		tile_counts_.resize(ntx * nty, 0);
	}

	// Per-tile layout: value[0] = count, value[1..N] = light indices.
	// Packed 4 values per RGBA texel.
	void add_to_tile(int tx, int ty, int light_idx) {
		int tile_linear = ty * num_tiles_x_ + tx;
		int count = tile_counts_[tile_linear];
		if (count >= FP_MAX_PER_TILE)
			return;

		int p = 1 + count;

		// Map (tx, ty, p) -> texel (texel_x, ty) channel ch
		int texel_x = tx * TILE_DATA_STRIDE + p / 4;
		int ch = p % 4;
		int offset = (ty * tile_tex_width_ + texel_x) * 4 + ch;
		tile_data_pixels_[offset] = static_cast<float>(light_idx);

		count++;
		tile_counts_[tile_linear] = count;

		int count_texel_x = tx * TILE_DATA_STRIDE;
		int count_offset = (ty * tile_tex_width_ + count_texel_x) * 4 + 0;
		tile_data_pixels_[count_offset] = static_cast<float>(count);
	}
};

} // namespace scene
