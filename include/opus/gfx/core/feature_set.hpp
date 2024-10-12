#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <variant>

namespace gfx {
class feature_set {
public:
  using feature_value =
      std::variant<bool, int, float, std::string>;

  feature_set() : _features() {};

  feature_set(const feature_set &) = default;

  void set_feature(const std::string &name,
                   const feature_value &value) noexcept {
    _features[name] = value;
  }

  void set_feature(const std::string &name,
                   feature_value &&value) noexcept {
    _features[name] = value;
  }

  const feature_value *
  get_feature(const std::string &name) const noexcept {
    auto it = _features.find(name);
    if (it == _features.end()) {
      return nullptr;
    }

    return &it->second;
  }

private:
  std::unordered_map<std::string, feature_value> _features;
};
} // namespace gfx