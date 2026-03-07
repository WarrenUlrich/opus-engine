#pragma once

namespace scene {

class tonemap {
public:
	tonemap() = default;

	void init() {
		float verts[] = {-1.0f, -1.0f, 3.0f, -1.0f, -1.0f, 3.0f};
		sg_buffer_desc bd = {};
		bd.data = SG_RANGE(verts);
		bd.usage.vertex_buffer = true;
		fsq_vbuf_ = sg_make_buffer(&bd);

		sg_sampler_desc sd = {};
		sd.min_filter = SG_FILTER_NEAREST;
		sd.mag_filter = SG_FILTER_NEAREST;
		sd.wrap_u = SG_WRAP_CLAMP_TO_EDGE;
		sd.wrap_v = SG_WRAP_CLAMP_TO_EDGE;
		sampler_ = sg_make_sampler(&sd);

		static const unsigned char vs[] = {
#embed "fullscreen.vert"
		    , 0};
		static const unsigned char fs[] = {
#embed "tonemap.frag"
		    , 0};

		sg_shader_desc shd = {};
		shd.vertex_func.source = reinterpret_cast<const char *>(vs);
		shd.fragment_func.source = reinterpret_cast<const char *>(fs);

		shd.views[0].texture.stage = SG_SHADERSTAGE_FRAGMENT;
		shd.views[0].texture.image_type = SG_IMAGETYPE_2D;
		shd.views[0].texture.sample_type = SG_IMAGESAMPLETYPE_UNFILTERABLE_FLOAT;

		shd.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
		shd.samplers[0].sampler_type = SG_SAMPLERTYPE_NONFILTERING;

		shd.texture_sampler_pairs[0].stage = SG_SHADERSTAGE_FRAGMENT;
		shd.texture_sampler_pairs[0].view_slot = 0;
		shd.texture_sampler_pairs[0].sampler_slot = 0;
		shd.texture_sampler_pairs[0].glsl_name = "scene_tex";

		sg_pipeline_desc pip = {};
		pip.shader = sg_make_shader(&shd);
		pip.layout.attrs[0].format = SG_VERTEXFORMAT_FLOAT2;
		pip.depth.write_enabled = false;
		pip.depth.compare = SG_COMPAREFUNC_ALWAYS;
		pipeline_ = sg_make_pipeline(&pip);

		scene_pass_action_.colors[0].load_action = SG_LOADACTION_CLEAR;
		scene_pass_action_.colors[0].clear_value = {0.01f, 0.01f, 0.02f, 1.0f};
		scene_pass_action_.depth.load_action = SG_LOADACTION_CLEAR;
		scene_pass_action_.depth.clear_value = 1.0f;

		swapchain_pass_action_.colors[0].load_action = SG_LOADACTION_DONTCARE;
		swapchain_pass_action_.depth.load_action = SG_LOADACTION_CLEAR;
		swapchain_pass_action_.depth.clear_value = 1.0f;
	}

	void ensure_targets(int w, int h) {
		if (w == width_ && h == height_)
			return;
		destroy_targets();
		width_ = w;
		height_ = h;
		create_targets(w, h);
	}

	[[nodiscard]] sg_attachments scene_attachments() const { return scene_att_; }
	[[nodiscard]] sg_pass_action scene_pass_action() const { return scene_pass_action_; }

	void composite_to_swapchain() {
		sg_pass pass = {};
		pass.action = swapchain_pass_action_;
		pass.swapchain = sglue_swapchain();
		sg_begin_pass(pass);

		sg_apply_pipeline(pipeline_);
		sg_bindings bind = {};
		bind.vertex_buffers[0] = fsq_vbuf_;
		bind.views[0] = color_tex_view_;
		bind.samplers[0] = sampler_;
		sg_apply_bindings(&bind);
		sg_draw(0, 3, 1);

		sg_end_pass();
	}

	void destroy() {
		destroy_targets();
		if (sampler_.id)
			sg_destroy_sampler(sampler_);
		if (pipeline_.id)
			sg_destroy_pipeline(pipeline_);
		if (fsq_vbuf_.id)
			sg_destroy_buffer(fsq_vbuf_);
	}

private:
	sg_image color_img_{};
	sg_image depth_img_{};
	sg_view color_att_view_{};
	sg_view depth_att_view_{};
	sg_view color_tex_view_{};
	sg_attachments scene_att_{};
	sg_pass_action scene_pass_action_{};
	sg_pass_action swapchain_pass_action_{};

	sg_pipeline pipeline_{};
	sg_sampler sampler_{};
	sg_buffer fsq_vbuf_{};

	int width_{0};
	int height_{0};

	void create_targets(int w, int h) {
		sg_image_desc cd = {};
		cd.usage.color_attachment = true;
		cd.width = w;
		cd.height = h;
		cd.pixel_format = SG_PIXELFORMAT_RGBA16F;
		cd.sample_count = 1;
		color_img_ = sg_make_image(&cd);

		sg_image_desc dd = {};
		dd.usage.depth_stencil_attachment = true;
		dd.width = w;
		dd.height = h;
		dd.pixel_format = SG_PIXELFORMAT_DEPTH;
		dd.sample_count = 1;
		depth_img_ = sg_make_image(&dd);

		sg_view_desc va = {};
		va.color_attachment.image = color_img_;
		color_att_view_ = sg_make_view(&va);

		sg_view_desc vda = {};
		vda.depth_stencil_attachment.image = depth_img_;
		depth_att_view_ = sg_make_view(&vda);

		sg_view_desc vt = {};
		vt.texture.image = color_img_;
		color_tex_view_ = sg_make_view(&vt);

		scene_att_ = {};
		scene_att_.colors[0] = color_att_view_;
		scene_att_.depth_stencil = depth_att_view_;
	}

	void destroy_targets() {
		if (color_att_view_.id)
			sg_destroy_view(color_att_view_);
		if (depth_att_view_.id)
			sg_destroy_view(depth_att_view_);
		if (color_tex_view_.id)
			sg_destroy_view(color_tex_view_);
		if (color_img_.id)
			sg_destroy_image(color_img_);
		if (depth_img_.id)
			sg_destroy_image(depth_img_);
		color_att_view_ = {};
		depth_att_view_ = {};
		color_tex_view_ = {};
		color_img_ = {};
		depth_img_ = {};
		width_ = 0;
		height_ = 0;
	}
};

} // namespace scene
