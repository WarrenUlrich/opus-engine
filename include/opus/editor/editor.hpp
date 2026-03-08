#pragma once

#include "sokol_app.h"
#include "sokol_gfx.h"
#include "sokol_imgui.h"

#include "imgui.h"
#include "imgui_internal.h"

#include "../scene/world.hpp"
#include "../scripting/script.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace editor {

constexpr float PI_F = 3.14159265f;
constexpr float DEG2RAD = PI_F / 180.0f;
constexpr float RAD2DEG = 180.0f / PI_F;

class editor_layer {
public:
	// Current viewport size (updated each frame from the ImGui viewport panel)
	int viewport_width{1};
	int viewport_height{1};

	// Whether the viewport panel is hovered (useful for gating camera input)
	bool viewport_hovered{false};
	bool viewport_focused{false};

	// Play-mode edge triggers — main.cpp reads and clears these
	bool play_requested{false};
	bool stop_requested{false};
	bool is_playing{false};

	// Simple log buffer (main.cpp can push messages too)
	std::vector<std::string> console_log;

	void log(const std::string &msg) {
		console_log.push_back(msg);
		if (console_log.size() > 1000) console_log.erase(console_log.begin());
	}

	void init() {
		ImGuiIO &io = ImGui::GetIO();
		io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

		ImGui::StyleColorsDark();
		ImGuiStyle &style = ImGui::GetStyle();
		style.WindowRounding = 0.0f;
		style.FrameRounding = 2.0f;
		style.GrabRounding = 2.0f;
		style.FrameBorderSize = 1.0f;
		style.WindowBorderSize = 1.0f;

		auto &c = style.Colors;
		c[ImGuiCol_WindowBg]       = ImVec4(0.10f, 0.10f, 0.10f, 1.00f);
		c[ImGuiCol_TitleBg]        = ImVec4(0.08f, 0.08f, 0.08f, 1.00f);
		c[ImGuiCol_TitleBgActive]  = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
		c[ImGuiCol_Tab]            = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
		c[ImGuiCol_TabSelected]    = ImVec4(0.24f, 0.24f, 0.24f, 1.00f);
		c[ImGuiCol_TabHovered]     = ImVec4(0.32f, 0.32f, 0.32f, 1.00f);
		c[ImGuiCol_FrameBg]        = ImVec4(0.16f, 0.16f, 0.16f, 1.00f);
		c[ImGuiCol_FrameBgHovered] = ImVec4(0.22f, 0.22f, 0.22f, 1.00f);
		c[ImGuiCol_FrameBgActive]  = ImVec4(0.28f, 0.28f, 0.28f, 1.00f);
		c[ImGuiCol_Header]         = ImVec4(0.18f, 0.18f, 0.18f, 1.00f);
		c[ImGuiCol_HeaderHovered]  = ImVec4(0.26f, 0.26f, 0.26f, 1.00f);
		c[ImGuiCol_HeaderActive]   = ImVec4(0.30f, 0.30f, 0.30f, 1.00f);
	}

	// Call once per frame *after* simgui_new_frame() and *before* simgui_render().
	void draw(sg_view viewport_tex_view, sg_sampler viewport_sampler,
	          scene::world &world, const std::vector<std::string> &available_scripts) {

		// Invalidate selection if the entity was killed
		if (has_selection_ && !world.is_alive(selected_entity_)) {
			has_selection_ = false;
		}

		// ---- Full-window dockspace ----
		const ImGuiViewport *vp = ImGui::GetMainViewport();
		ImGui::SetNextWindowPos(vp->WorkPos);
		ImGui::SetNextWindowSize(vp->WorkSize);
		ImGui::SetNextWindowViewport(vp->ID);

		ImGuiWindowFlags host_flags =
		    ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
		    ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
		    ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus |
		    ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_MenuBar;

		ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
		ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

		ImGui::Begin("DockSpaceHost", nullptr, host_flags);
		ImGui::PopStyleVar(3);

		draw_menu_bar();

		ImGuiID dockspace_id = ImGui::GetID("EditorDockSpace");
		ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None);

		if (!layout_initialized_) {
			layout_initialized_ = true;
			build_default_layout(dockspace_id);
		}

		ImGui::End(); // DockSpaceHost

		// ---- Panels ----
		if (show_viewport_) draw_viewport(viewport_tex_view, viewport_sampler);
		if (show_hierarchy_) draw_hierarchy(world);
		if (show_properties_) draw_properties(world, available_scripts);
		if (show_console_) draw_console_panel();
		if (show_demo_) ImGui::ShowDemoWindow(&show_demo_);
	}

private:
	bool layout_initialized_{false};
	bool show_viewport_{true};
	bool show_hierarchy_{true};
	bool show_properties_{true};
	bool show_console_{true};
	bool show_demo_{false};

	ecs::entity_id selected_entity_{0, 0};
	bool has_selection_{false};

	float fps_smooth_{60.0f};
	float frame_time_smooth_{1.0f / 60.0f};

	// ==================== Menu bar =========================================

	void draw_menu_bar() {
		if (!ImGui::BeginMenuBar()) return;

		if (ImGui::BeginMenu("File")) {
			if (ImGui::MenuItem("Exit")) sapp_request_quit();
			ImGui::EndMenu();
		}
		if (ImGui::BeginMenu("View")) {
			ImGui::MenuItem("Viewport", nullptr, &show_viewport_);
			ImGui::MenuItem("Scene Hierarchy", nullptr, &show_hierarchy_);
			ImGui::MenuItem("Properties", nullptr, &show_properties_);
			ImGui::MenuItem("Console", nullptr, &show_console_);
			ImGui::MenuItem("Demo Window", nullptr, &show_demo_);
			ImGui::EndMenu();
		}

		// Right-aligned play / stop button
		float avail = ImGui::GetContentRegionAvail().x;
		float btn_w = 80.0f;
		ImGui::SameLine(ImGui::GetCursorPosX() + avail - btn_w);

		if (is_playing) {
			ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.75f, 0.22f, 0.22f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.90f, 0.30f, 0.30f, 1.0f));
			if (ImGui::Button("  Stop  ")) stop_requested = true;
			ImGui::PopStyleColor(2);
		} else {
			ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.65f, 0.30f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.30f, 0.80f, 0.40f, 1.0f));
			if (ImGui::Button("  Play  ")) play_requested = true;
			ImGui::PopStyleColor(2);
		}

		ImGui::EndMenuBar();
	}

	// ==================== Scene Hierarchy ==================================

	void draw_hierarchy(scene::world &world) {
		ImGui::Begin("Scene Hierarchy", &show_hierarchy_);

		if (ImGui::Button("+ New Entity")) {
			auto e = world.new_entity();
			world.add_component(e, scene::entity_name{"New Entity"});
			world.add_component(e, scene::transform{});
			selected_entity_ = e;
			has_selection_ = true;
		}

		ImGui::Separator();

		world.for_each_alive([&](ecs::entity_id id) {
			char label[128];
			if (auto *n = world.try_get_component<scene::entity_name>(id))
				snprintf(label, sizeof(label), "%s##%u", n->name, id.index);
			else
				snprintf(label, sizeof(label), "Entity %u##%u", id.index, id.index);

			bool selected = has_selection_ && selected_entity_ == id;
			if (ImGui::Selectable(label, selected)) {
				selected_entity_ = id;
				has_selection_ = true;
			}
		});

		ImGui::Separator();

		if (has_selection_ && world.is_alive(selected_entity_)) {
			ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.15f, 0.15f, 1.0f));
			if (ImGui::Button("Delete Entity")) {
				// Clean up GPU / runtime resources before erasing
				if (auto *mesh = world.try_get_component<scene::mesh_instance>(selected_entity_))
					mesh->destroy();
				if (auto *sr = world.try_get_component<scene::script_ref>(selected_entity_)) {
					delete sr->instance;
					sr->instance = nullptr;
				}
				world.remove_entity(selected_entity_);
				has_selection_ = false;
			}
			ImGui::PopStyleColor();
		}

		ImGui::End();
	}

	// ==================== Properties =======================================

	void draw_properties(scene::world &world,
	                     const std::vector<std::string> &available_scripts) {
		ImGui::Begin("Properties", &show_properties_);

		if (!has_selection_ || !world.is_alive(selected_entity_)) {
			ImGui::TextDisabled("(select an entity)");
			ImGui::End();
			return;
		}

		auto id = selected_entity_;

		// ---- Entity name ----
		if (auto *n = world.try_get_component<scene::entity_name>(id)) {
			ImGui::SetNextItemWidth(-1);
			ImGui::InputText("##Name", n->name, sizeof(n->name));
		}
		ImGui::Separator();

		// ---- Transform ----
		draw_transform_section(world, id);
		// ---- Camera ----
		draw_camera_section(world, id);
		// ---- Light ----
		draw_light_section(world, id);
		// ---- Material ----
		draw_material_section(world, id);
		// ---- Mesh ----
		draw_mesh_section(world, id);
		// ---- Script ----
		draw_script_section(world, id, available_scripts);

		ImGui::Spacing();
		ImGui::Separator();
		ImGui::Spacing();

		// ---- Add Component popup ----
		if (ImGui::Button("+ Add Component", ImVec2(-1, 0)))
			ImGui::OpenPopup("AddComponent");

		if (ImGui::BeginPopup("AddComponent")) {
			if (!world.has_component<scene::entity_name>(id))
				if (ImGui::MenuItem("Entity Name"))
					world.add_component(id, scene::entity_name{});

			if (!world.has_component<scene::transform>(id))
				if (ImGui::MenuItem("Transform"))
					world.add_component(id, scene::transform{});

			if (!world.has_component<scene::camera>(id))
				if (ImGui::MenuItem("Camera"))
					world.add_component(id, scene::camera{});

			if (!world.has_component<scene::light>(id)) {
				if (ImGui::BeginMenu("Light")) {
					if (ImGui::MenuItem("Directional"))
						world.add_component(id, scene::light::directional({0, -1, 0}));
					if (ImGui::MenuItem("Point"))
						world.add_component(id, scene::light::point({0, 0, 0}));
					if (ImGui::MenuItem("Spot"))
						world.add_component(id, scene::light::spot({0, 2, 0}, {0, -1, 0}));
					ImGui::EndMenu();
				}
			}

			if (!world.has_component<scene::material>(id)) {
				if (ImGui::BeginMenu("Material")) {
					if (ImGui::MenuItem("Default"))  world.add_component(id, scene::material{});
					if (ImGui::MenuItem("Gold"))     world.add_component(id, scene::material::gold());
					if (ImGui::MenuItem("Copper"))   world.add_component(id, scene::material::copper());
					if (ImGui::MenuItem("Silver"))   world.add_component(id, scene::material::silver());
					if (ImGui::MenuItem("Plastic"))  world.add_component(id, scene::material::plastic());
					if (ImGui::MenuItem("Ceramic"))  world.add_component(id, scene::material::ceramic());
					ImGui::EndMenu();
				}
			}

			if (!world.has_component<scene::mesh_instance>(id)) {
				if (ImGui::BeginMenu("Mesh")) {
					if (ImGui::MenuItem("Plane"))
						world.add_component(id, scene::mesh_instance::plane(10.0f, 10.0f));
					if (ImGui::MenuItem("Cube"))
						world.add_component(id, scene::mesh_instance::cube(1.0f, 1.0f, 1.0f));
					if (ImGui::MenuItem("Sphere"))
						world.add_component(id, scene::mesh_instance::sphere(1.0f, 32, 16));
					if (ImGui::MenuItem("Torus"))
						world.add_component(id, scene::mesh_instance::torus(1.0f, 0.3f, 32, 16));
					ImGui::EndMenu();
				}
			}

			if (!world.has_component<scene::script_ref>(id))
				if (ImGui::MenuItem("Script"))
					world.add_component(id, scene::script_ref{});

			ImGui::EndPopup();
		}

		ImGui::End();
	}

	// ---- Component sections (each draws header + fields + remove button) --

	void draw_transform_section(scene::world &world, ecs::entity_id id) {
		if (!world.has_component<scene::transform>(id)) return;
		auto *t = world.try_get_component<scene::transform>(id);

		bool removed = false;
		bool open = component_header("Transform", "rm_xform", removed);
		if (removed) {
			world.remove_component<scene::transform>(id);
			return;
		}
		if (open) {
			ImGui::DragFloat3("Position", &t->position.x, 0.1f);

			math::vec3 euler = t->rotation.to_euler();
			float deg[3] = {euler.x * RAD2DEG, euler.y * RAD2DEG, euler.z * RAD2DEG};
			if (ImGui::DragFloat3("Rotation", deg, 1.0f)) {
				t->rotation = math::quat::from_euler(
				    {deg[0] * DEG2RAD, deg[1] * DEG2RAD, deg[2] * DEG2RAD});
			}

			ImGui::DragFloat3("Scale", &t->scale.x, 0.05f, 0.01f, 1000.0f);
		}
	}

	void draw_camera_section(scene::world &world, ecs::entity_id id) {
		if (!world.has_component<scene::camera>(id)) return;
		auto *cam = world.try_get_component<scene::camera>(id);

		bool removed = false;
		bool open = component_header("Camera", "rm_cam", removed);
		if (removed) {
			world.remove_component<scene::camera>(id);
			return;
		}
		if (open) {
			float fov = cam->fov_y_rad * RAD2DEG;
			if (ImGui::DragFloat("FOV (deg)", &fov, 1.0f, 1.0f, 179.0f))
				cam->fov_y_rad = fov * DEG2RAD;
			ImGui::DragFloat("Near Z", &cam->near_z, 0.01f, 0.001f, 100.0f);
			ImGui::DragFloat("Far Z", &cam->far_z, 1.0f, 1.0f, 10000.0f);
		}
	}

	void draw_light_section(scene::world &world, ecs::entity_id id) {
		if (!world.has_component<scene::light>(id)) return;
		auto *l = world.try_get_component<scene::light>(id);

		bool removed = false;
		bool open = component_header("Light", "rm_light", removed);
		if (removed) {
			world.remove_component<scene::light>(id);
			return;
		}
		if (open) {
			const char *types[] = {"Directional", "Point", "Spot"};
			int cur = static_cast<int>(l->type);
			if (ImGui::Combo("Type", &cur, types, 3))
				l->type = static_cast<scene::light_type>(cur);

			ImGui::ColorEdit3("Color", &l->color.x);
			ImGui::DragFloat("Intensity", &l->intensity, 0.1f, 0.0f, 100.0f);

			if (l->type != scene::light_type::directional)
				ImGui::DragFloat("Range", &l->range, 0.1f, 0.0f, 1000.0f);

			if (l->type == scene::light_type::directional || l->type == scene::light_type::spot)
				ImGui::DragFloat3("Direction", &l->direction.x, 0.01f);

			if (l->type == scene::light_type::spot) {
				float inner = std::acos(l->inner_cone_cos) * RAD2DEG;
				float outer = std::acos(l->outer_cone_cos) * RAD2DEG;
				if (ImGui::DragFloat("Inner Cone", &inner, 0.5f, 0.0f, 89.0f))
					l->inner_cone_cos = std::cos(inner * DEG2RAD);
				if (ImGui::DragFloat("Outer Cone", &outer, 0.5f, 0.0f, 89.0f))
					l->outer_cone_cos = std::cos(outer * DEG2RAD);
			}
		}
	}

	void draw_material_section(scene::world &world, ecs::entity_id id) {
		if (!world.has_component<scene::material>(id)) return;
		auto *mat = world.try_get_component<scene::material>(id);

		bool removed = false;
		bool open = component_header("Material", "rm_mat", removed);
		if (removed) {
			world.remove_component<scene::material>(id);
			return;
		}
		if (open) {
			ImGui::ColorEdit3("Albedo", &mat->albedo.x);
			ImGui::SliderFloat("Metallic", &mat->metallic, 0.0f, 1.0f);
			ImGui::SliderFloat("Roughness", &mat->roughness, 0.0f, 1.0f);
			ImGui::SliderFloat("AO", &mat->ao, 0.0f, 1.0f);
		}
	}

	void draw_mesh_section(scene::world &world, ecs::entity_id id) {
		if (!world.has_component<scene::mesh_instance>(id)) return;
		auto *mesh = world.try_get_component<scene::mesh_instance>(id);

		bool removed = false;
		bool open = component_header("Mesh", "rm_mesh", removed);
		if (removed) {
			mesh->destroy();
			world.remove_component<scene::mesh_instance>(id);
			return;
		}
		if (open) {
			ImGui::Text("Indices: %d", mesh->index_count);
		}
	}

	void draw_script_section(scene::world &world, ecs::entity_id id,
	                         const std::vector<std::string> &scripts) {
		if (!world.has_component<scene::script_ref>(id)) return;
		auto *sr = world.try_get_component<scene::script_ref>(id);

		bool removed = false;
		bool open = component_header("Script", "rm_script", removed);
		if (removed) {
			if (sr->instance) { delete sr->instance; sr->instance = nullptr; }
			world.remove_component<scene::script_ref>(id);
			return;
		}
		if (open) {
			const char *preview = sr->script_name[0] ? sr->script_name : "(none)";
			if (ImGui::BeginCombo("##ScriptCombo", preview)) {
				if (ImGui::Selectable("(none)", sr->script_name[0] == '\0')) {
					sr->script_name[0] = '\0';
					if (sr->instance) {
						delete sr->instance;
						sr->instance = nullptr;
						sr->started = false;
					}
				}
				for (auto &name : scripts) {
					bool sel = std::strcmp(sr->script_name, name.c_str()) == 0;
					if (ImGui::Selectable(name.c_str(), sel)) {
						std::strncpy(sr->script_name, name.c_str(),
						             sizeof(sr->script_name) - 1);
						sr->script_name[sizeof(sr->script_name) - 1] = '\0';
						if (sr->instance) {
							delete sr->instance;
							sr->instance = nullptr;
							sr->started = false;
						}
					}
				}
				ImGui::EndCombo();
			}

			if (sr->started)
				ImGui::TextColored(ImVec4(0.3f, 0.9f, 0.3f, 1.0f), "Running");
		}
	}

	// Draws a collapsing header with an inline "X" remove button.
	// `removed` is set true if the X was clicked; returns true when the section is open.
	bool component_header(const char *label, const char *remove_id, bool &removed) {
		bool open = ImGui::CollapsingHeader(
		    label, ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_AllowOverlap);

		ImGui::SameLine(ImGui::GetWindowWidth() - 30.0f);
		ImGui::PushID(remove_id);
		removed = ImGui::SmallButton("X");
		ImGui::PopID();
		return open;
	}

	// ==================== Console ==========================================

	void draw_console_panel() {
		ImGui::Begin("Console", &show_console_);
		for (auto &msg : console_log) {
			ImGui::TextUnformatted(msg.c_str());
		}
		if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
			ImGui::SetScrollHereY(1.0f);
		ImGui::End();
	}

	// ==================== Viewport =========================================

	void draw_viewport(sg_view tex_view, sg_sampler smp) {
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
		ImGui::Begin("Viewport", &show_viewport_);
		ImGui::PopStyleVar();

		viewport_focused = ImGui::IsWindowFocused();
		viewport_hovered = ImGui::IsWindowHovered();

		ImVec2 avail = ImGui::GetContentRegionAvail();
		viewport_width  = std::max(1, static_cast<int>(avail.x));
		viewport_height = std::max(1, static_cast<int>(avail.y));

		ImTextureID imtex = (ImTextureID)simgui_imtextureid_with_sampler(tex_view, smp);
		ImVec2 cursor = ImGui::GetCursorScreenPos();
		ImGui::Image(imtex, avail, ImVec2(0, 1), ImVec2(1, 0));

		// FPS overlay
		update_fps_counter();
		{
			char fps_buf[64];
			snprintf(fps_buf, sizeof(fps_buf), "%.1f FPS (%.2f ms)",
			         fps_smooth_, frame_time_smooth_ * 1000.0f);

			ImVec2 text_size = ImGui::CalcTextSize(fps_buf);
			ImVec2 padding{6.0f, 4.0f};
			ImVec2 box_min{cursor.x + 8.0f, cursor.y + 8.0f};
			ImVec2 box_max{box_min.x + text_size.x + padding.x * 2,
			               box_min.y + text_size.y + padding.y * 2};

			ImDrawList *dl = ImGui::GetWindowDrawList();
			dl->AddRectFilled(box_min, box_max, IM_COL32(0, 0, 0, 160), 4.0f);
			dl->AddText(ImVec2(box_min.x + padding.x, box_min.y + padding.y),
			            IM_COL32(180, 230, 120, 255), fps_buf);
		}

		ImGui::End();
	}

	// ==================== Layout ===========================================

	void build_default_layout(ImGuiID dockspace_id) {
		ImGui::DockBuilderRemoveNode(dockspace_id);
		ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
		ImGui::DockBuilderSetNodeSize(dockspace_id, ImGui::GetMainViewport()->WorkSize);

		ImGuiID left, centre;
		ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Left, 0.20f, &left, &centre);

		ImGuiID top, bottom;
		ImGui::DockBuilderSplitNode(centre, ImGuiDir_Down, 0.25f, &bottom, &top);

		ImGuiID viewport_dock, right;
		ImGui::DockBuilderSplitNode(top, ImGuiDir_Right, 0.25f, &right, &viewport_dock);

		ImGui::DockBuilderDockWindow("Scene Hierarchy", left);
		ImGui::DockBuilderDockWindow("Viewport", viewport_dock);
		ImGui::DockBuilderDockWindow("Properties", right);
		ImGui::DockBuilderDockWindow("Console", bottom);

		ImGui::DockBuilderFinish(dockspace_id);
	}

	void update_fps_counter() {
		float dt = ImGui::GetIO().DeltaTime;
		if (dt <= 0.0f) dt = 1.0f / 60.0f;
		constexpr float alpha = 0.05f;
		frame_time_smooth_ += alpha * (dt - frame_time_smooth_);
		fps_smooth_ += alpha * ((1.0f / dt) - fps_smooth_);
	}
};

} // namespace editor
