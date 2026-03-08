#pragma once

#include "../math/vec3.hpp"

#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace scene {

// ---------------------------------------------------------------------------
// obj_data — runtime Wavefront OBJ mesh parser.
//
//   auto mesh = obj_data::parse(src);          // from memory / #embed
//   auto mesh = obj_data::load("model.obj");   // from disk
//
// Supports: v, vn, f (v, v//vn, v/vt/vn), negative indices,
//           n-gon fan triangulation.  Comments & unknown lines skipped.
// ---------------------------------------------------------------------------

struct obj_data {
	struct vertex {
		math::vec3 position{};
		math::vec3 normal{};
	};

	std::vector<vertex>   vertices;
	std::vector<uint16_t> indices;

	// -- parse from memory (works with #embed data via string_view) ----------

	[[nodiscard]] static obj_data parse(std::string_view src) {
		obj_data   out;
		const char *cur = src.data(), *end = cur + src.size();

		std::vector<math::vec3> positions, normals;

		// -- scanning helpers ------------------------------------------------

		auto skip = [&] { while (cur < end && (*cur == ' ' || *cur == '\t' || *cur == '\r')) ++cur; };

		auto skip_line = [&] {
			while (cur < end && *cur != '\n') ++cur;
			if (cur < end) ++cur;
		};

		auto at_eol = [&] { skip(); return cur >= end || *cur == '\n' || *cur == '#'; };

		auto read_float = [&] {
			skip();
			float v{};
			auto [p, ec] = std::from_chars(cur, end, v);
			if (ec != std::errc{}) throw std::runtime_error("obj: bad float");
			cur = p;
			return v;
		};

		auto read_int = [&] {
			skip();
			int v{};
			auto [p, ec] = std::from_chars(cur, end, v);
			if (ec != std::errc{}) throw std::runtime_error("obj: bad int");
			cur = p;
			return v;
		};

		auto read_vec3 = [&] { return math::vec3{read_float(), read_float(), read_float()}; };

		// -- face index: v, v//vn, v/vt/vn -----------------------------------

		struct fi { int v{}, vt{}, vn{}; };

		auto read_fi = [&] {
			fi f{.v = read_int()};
			if (cur < end && *cur == '/') {
				++cur;
				if (cur < end && *cur == '/') { ++cur; f.vn = read_int(); }
				else { f.vt = read_int(); if (cur < end && *cur == '/') { ++cur; f.vn = read_int(); } }
			}
			return f;
		};

		auto resolve = [&](const fi &f) -> uint16_t {
			auto idx = [](int i, size_t n) -> size_t {
				auto r = i > 0 ? size_t(i - 1) : size_t(int(n) + i);
				if (r >= n) throw std::out_of_range("obj: index out of range");
				return r;
			};
			math::vec3 pos  = positions[idx(f.v, positions.size())];
			math::vec3 norm = f.vn ? normals[idx(f.vn, normals.size())]
			                       : math::vec3{0.f, 1.f, 0.f};
			auto vi = static_cast<uint16_t>(out.vertices.size());
			out.vertices.push_back({pos, norm});
			return vi;
		};

		// -- main loop -------------------------------------------------------

		while (cur < end) {
			skip();
			if (cur >= end) break;
			char c = *cur;

			if (c == '#' || c == '\n') { skip_line(); continue; }

			if (c == 'v') {
				++cur;
				if      (cur < end && (*cur == ' ' || *cur == '\t')) positions.push_back(read_vec3());
				else if (cur < end && *cur == 'n') { ++cur; normals.push_back(read_vec3()); }
				skip_line();
				continue;
			}

			if (c == 'f') {
				++cur;
				std::vector<uint16_t> face;
				while (!at_eol()) face.push_back(resolve(read_fi()));
				for (size_t i = 1; i + 1 < face.size(); ++i) {
					out.indices.push_back(face[0]);
					out.indices.push_back(face[i + 1]);
					out.indices.push_back(face[i]);
				}
				skip_line();
				continue;
			}

			skip_line(); // o, g, s, mtllib, usemtl, …
		}
		return out;
	}

	// -- load from disk ------------------------------------------------------

	[[nodiscard]] static obj_data load(const std::filesystem::path &path) {
		std::ifstream f(path, std::ios::in | std::ios::binary | std::ios::ate);
		if (!f) throw std::runtime_error("obj: cannot open " + path.string());
		auto sz = f.tellg();
		std::string buf(static_cast<size_t>(sz), '\0');
		f.seekg(0);
		f.read(buf.data(), sz);
		return parse(buf);
	}
};

} // namespace scene
