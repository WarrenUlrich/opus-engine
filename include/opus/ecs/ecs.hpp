#pragma once

#include <bitset>
#include <cassert>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <optional>
#include <tuple>
#include <vector>

namespace ecs {
using entity_id = std::size_t;

template <typename... Components> class context {
public:
  entity_id new_entity() noexcept {
    if (!_inactive_ids.empty()) {
      auto reused = _inactive_ids.back();
      _inactive_ids.pop_back();
      return reused;
    }

    entity_id new_id = _component_masks.size();
    _component_masks.push_back(_component_bitset());
    _resize_component_vectors(new_id + 1);
    return new_id;
  }

  void remove_entity(entity_id entity) noexcept {
    assert(entity < _component_masks.size());

    auto &mask = _component_masks[entity];
    mask.reset();

    _inactive_ids.emplace_back(entity);
  }

  template <typename Component>
  void add_component(entity_id entity, Component &&component) noexcept {
    assert(entity < _component_masks.size());

    auto &component_vector = std::get<std::vector<Component>>(_components);

    component_vector[entity] = std::move(component);

    _component_masks[entity].set(_get_bitset_index<Component>());
  }

  template <typename Component>
  bool remove_component(entity_id entity) noexcept {
    assert(entity < _component_masks.size());

    auto &component_vector = std::get<std::vector<Component>>(_components);

    assert(entity <= component_vector.size());

    _component_masks[entity].reset(_get_bitset_index<Component>());

    return true;
  }

  template <typename Component>
  bool has_component(entity_id entity) const noexcept {
    assert(entity < _component_masks.size());

    constexpr auto bitset_idx = _get_bitset_index<Component>();
    return _component_masks[entity].test(bitset_idx);
  }

  template <typename... QueryComponents>
  bool has_components(entity_id entity) const noexcept {
    assert(entity < _component_masks.size());

    constexpr auto query_bitset = _get_query_bitset<QueryComponents...>();
    return (_component_masks[entity] & query_bitset) == query_bitset;
  }

  template <typename... QueryComponents>
  std::optional<std::tuple<QueryComponents *...>>
  try_get_components(entity_id id) noexcept {
    if (!has_components<QueryComponents...>(id))
      return std::nullopt;

    return std::tuple<QueryComponents &...>(
        &_get_component_unchecked<QueryComponents>(id)...);
  }

  template <typename Component>
  Component *try_get_component(entity_id id) const noexcept {
    if (!has_component<Component>(id))
      return nullptr;

    return &_get_component_unchecked<Component>(id);
  }

  template <typename... QueryComponents> void for_each_entity(auto &&fn) {
    using fn_t = std::decay_t<decltype(fn)>;

    static_assert(std::is_invocable_v<fn_t, entity_id, QueryComponents &...> ||
                      std::is_invocable_v<fn_t, QueryComponents &...>,
                  "fn must be invocable with either (entity_id, "
                  "QueryComponents &...) or (QueryComponents &...) "
                  "arguments");

    constexpr auto query_bitset = _get_query_bitset<QueryComponents...>();

    #pragma omp parallel for schedule(static)
    for (entity_id i = 0; i < _component_masks.size(); ++i) {
      if ((_component_masks[i] & query_bitset) == query_bitset) {
        if constexpr (std::is_invocable_v<fn_t, entity_id,
                                          QueryComponents &...>) {
          fn(i, _get_component_unchecked<QueryComponents>(i)...);
        } else if constexpr (std::is_invocable_v<fn_t, QueryComponents &...>) {
          fn(_get_component_unchecked<QueryComponents>(i)...);
        }
      }
    }
  }

private:
  using _component_bitset = std::bitset<sizeof...(Components)>;

  std::vector<_component_bitset> _component_masks;

  std::tuple<std::vector<Components>...> _components;

  std::vector<entity_id> _inactive_ids;

  void _resize_component_vectors(std::size_t size) {
    (
        [&] {
          auto &vec = std::get<std::vector<Components>>(_components);
          vec.resize(size);
        }(),
        ...);
  }

  template <typename Component, std::size_t I = 0>
  static constexpr std::size_t _get_bitset_index() {
    using component_tuple = std::tuple<Components...>;

    if constexpr (I == sizeof...(Components)) {
      return I;
    } else if constexpr (std::is_same_v<Component, std::tuple_element_t<
                                                       I, component_tuple>>) {
      return I;
    } else {
      return _get_bitset_index<Component, I + 1>();
    }
  }

  template <typename... QueryComponents>
  static constexpr _component_bitset _get_query_bitset() {
    return (... | (1ULL << _get_bitset_index<QueryComponents>()));
  }

  template <typename Component>
  Component &_get_component_unchecked(entity_id id) noexcept {
    return std::get<std::vector<Component>>(_components)[id];
  }
};
} // namespace ecs