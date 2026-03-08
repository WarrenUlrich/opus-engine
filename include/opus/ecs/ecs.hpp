#pragma once

#include <bitset>
#include <cassert>
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace ecs {

// Generational entity ID: version increments on reuse to invalidate stale handles.
struct entity_id {
	uint32_t index;
	uint32_t version;
	bool operator==(const entity_id &) const = default;
};

namespace detail {
constexpr uint32_t NULL_INDEX = ~0u;

// Sparse set: components packed contiguously in memory.
template <typename T> struct sparse_set {
	std::vector<T> data;
	std::vector<uint32_t> packed_entities;
	std::vector<uint32_t> sparse;

	void insert(uint32_t entity_idx, T &&component) {
		if (entity_idx >= sparse.size()) {
			sparse.resize(entity_idx + 1, NULL_INDEX);
		}
		if (sparse[entity_idx] == NULL_INDEX) {
			sparse[entity_idx] = packed_entities.size();
			packed_entities.push_back(entity_idx);
			data.push_back(std::forward<T>(component));
		} else {
			data[sparse[entity_idx]] = std::forward<T>(component);
		}
	}

	void erase(uint32_t entity_idx) {
		if (entity_idx < sparse.size() && sparse[entity_idx] != NULL_INDEX) {
			uint32_t packed_idx = sparse[entity_idx];
			uint32_t last_entity = packed_entities.back();

			// Swap-and-pop to keep data contiguous
			data[packed_idx] = std::move(data.back());
			packed_entities[packed_idx] = last_entity;

			sparse[last_entity] = packed_idx;
			sparse[entity_idx] = NULL_INDEX;

			data.pop_back();
			packed_entities.pop_back();
		}
	}

	T &get(uint32_t entity_idx) { return data[sparse[entity_idx]]; }
};
} // namespace detail

template <typename... Components> class context {
public:
	entity_id new_entity() noexcept {
		uint32_t idx;
		if (!_inactive_indices.empty()) {
			idx = _inactive_indices.back();
			_inactive_indices.pop_back();
			_alive[idx] = true;
		} else {
			idx = static_cast<uint32_t>(_entity_versions.size());
			_entity_versions.push_back(0);
			_component_masks.push_back(_component_bitset());
			_alive.push_back(true);
		}
		return {idx, _entity_versions[idx]};
	}

	bool is_alive(entity_id entity) const noexcept {
		return entity.index < _entity_versions.size() &&
		       _entity_versions[entity.index] == entity.version;
	}

	void remove_entity(entity_id entity) noexcept {
		if (!is_alive(entity))
			return;

		auto &mask = _component_masks[entity.index];

		_erase_components_impl(entity.index, mask, std::index_sequence_for<Components...>{});

		mask.reset();
		_entity_versions[entity.index]++;
		_inactive_indices.push_back(entity.index);
		_alive[entity.index] = false;
	}

	template <typename Component>
	void add_component(entity_id entity, Component &&component) noexcept {
		assert(is_alive(entity));
		using CleanT = std::decay_t<Component>;

		auto &set = std::get<detail::sparse_set<CleanT>>(_components);
		set.insert(entity.index, std::forward<Component>(component));

		_component_masks[entity.index].set(_get_bitset_index<CleanT>());
	}

	template <typename Component> bool remove_component(entity_id entity) noexcept {
		assert(is_alive(entity));
		constexpr auto bitset_idx = _get_bitset_index<Component>();

		if (!_component_masks[entity.index].test(bitset_idx))
			return false;

		auto &set = std::get<detail::sparse_set<Component>>(_components);
		set.erase(entity.index);
		_component_masks[entity.index].reset(bitset_idx);

		return true;
	}

	template <typename Component> bool has_component(entity_id entity) const noexcept {
		if (!is_alive(entity))
			return false;
		return _component_masks[entity.index].test(_get_bitset_index<Component>());
	}

	template <typename... QueryComponents> bool has_components(entity_id entity) const noexcept {
		if (!is_alive(entity))
			return false;
		constexpr auto query = _get_query_bitset<QueryComponents...>();
		return (_component_masks[entity.index] & query) == query;
	}

	template <typename... QueryComponents>
	std::optional<std::tuple<QueryComponents *...>> try_get_components(entity_id id) noexcept {
		if (!has_components<QueryComponents...>(id))
			return std::nullopt;
		return std::tuple<QueryComponents *...>(
		    &_get_component_unchecked<QueryComponents>(id.index)...);
	}

	template <typename Component> Component *try_get_component(entity_id id) noexcept {
		if (!has_component<Component>(id))
			return nullptr;
		return &_get_component_unchecked<Component>(id.index);
	}

	// Iterate every living entity (regardless of components).
	void for_each_alive(auto &&fn) {
		for (uint32_t i = 0; i < _alive.size(); ++i) {
			if (_alive[i]) {
				fn(entity_id{i, _entity_versions[i]});
			}
		}
	}

	[[nodiscard]] uint32_t alive_count() const noexcept {
		uint32_t n = 0;
		for (uint32_t i = 0; i < _alive.size(); ++i)
			if (_alive[i]) ++n;
		return n;
	}

	template <typename... QueryComponents> void for_each_entity(auto &&fn) {
		static_assert(sizeof...(QueryComponents) > 0, "Query must have at least one component.");
		using fn_t = std::decay_t<decltype(fn)>;
		static_assert(std::is_invocable_v<fn_t, entity_id, QueryComponents &...> ||
		                  std::is_invocable_v<fn_t, QueryComponents &...>,
		              "Invalid lambda signature for for_each_entity");

		constexpr auto query_bitset = _get_query_bitset<QueryComponents...>();

		using FirstComponent = std::tuple_element_t<0, std::tuple<QueryComponents...>>;
		auto &driving_set = std::get<detail::sparse_set<FirstComponent>>(_components);

		for (uint32_t entity_idx : driving_set.packed_entities) {
			if ((_component_masks[entity_idx] & query_bitset) == query_bitset) {

				if constexpr (std::is_invocable_v<fn_t, entity_id, QueryComponents &...>) {
					entity_id id{entity_idx, _entity_versions[entity_idx]};
					fn(id, _get_component_unchecked<QueryComponents>(entity_idx)...);
				} else {
					fn(_get_component_unchecked<QueryComponents>(entity_idx)...);
				}
			}
		}
	}

private:
	using _component_bitset = std::bitset<sizeof...(Components)>;

	std::vector<_component_bitset> _component_masks;
	std::vector<uint32_t> _entity_versions;
	std::vector<uint32_t> _inactive_indices;
	std::vector<bool> _alive;

	std::tuple<detail::sparse_set<Components>...> _components;

	template <typename Component, std::size_t I = 0>
	static constexpr std::size_t _get_bitset_index() {
		if constexpr (std::is_same_v<Component, std::tuple_element_t<I, std::tuple<Components...>>>) {
			return I;
		} else {
			return _get_bitset_index<Component, I + 1>();
		}
	}

	template <typename... QueryComponents> static constexpr _component_bitset _get_query_bitset() {
		_component_bitset bits;
		(bits.set(_get_bitset_index<QueryComponents>()), ...);
		return bits;
	}

	template <typename Component> Component &_get_component_unchecked(uint32_t entity_idx) noexcept {
		return std::get<detail::sparse_set<Component>>(_components).get(entity_idx);
	}

	template <std::size_t... Is>
	void _erase_components_impl(uint32_t entity_idx, const _component_bitset &mask,
	                            std::index_sequence<Is...>) {
		((mask.test(Is) ? std::get<Is>(_components).erase(entity_idx) : void()), ...);
	}
};
} // namespace ecs