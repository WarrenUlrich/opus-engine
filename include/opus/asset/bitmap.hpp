#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <optional>
#include <string_view>
#include <vector>

namespace asset {
class bitmap {
public:
  static std::optional<bitmap> load(std::fstream &file) noexcept {
    const auto file_header = _read_file_header(file);
    if (!file_header)
      return std::nullopt;

    const auto info_header = _read_info_header(file);
    if (!info_header)
      return std::nullopt;

    if (info_header->compression) {
      std::cerr << "compression not supported\n";
      return std::nullopt;
    }

    file.seekg(file_header->offset); // move to pixel data

    std::size_t row_size =
        ((info_header->width * info_header->bitcount + 31) / 32) * 4;
    
    const auto row_padding = (row_size % 4 == 0 ? 0 : 4 - row_size % 4);

    return std::nullopt;
  }

  std::int32_t width() const noexcept { return _info.width; }

  std::int32_t height() const noexcept { return _info.height; }

private:
  struct _file_header {
    std::uint16_t type;
    std::uint32_t size;
    std::uint16_t reserved1;
    std::uint16_t reserved2;
    std::uint32_t offset;
  } _header;

  struct _info_header {
    std::uint32_t size;
    std::int32_t width;
    std::int32_t height;
    std::uint16_t planes;
    std::uint16_t bitcount;
    std::uint32_t compression;
    std::uint32_t sizeimage;
    std::int32_t xpelspermeter;
    std::int32_t ypelspermeter;
    std::uint32_t colorsused;
    std::uint32_t colorsimportant;
  } _info;

  std::vector<std::uint8_t> _data;

  constexpr bitmap(const _file_header &header, const _info_header &info,
                   const std::vector<std::uint8_t> &data)
      : _header(header), _info(info), _data(data) {};

  static std::optional<_file_header>
  _read_file_header(std::fstream &file) noexcept {
    _file_header header{};
    if (!file.read((char *)&header, sizeof(_file_header)))
      return std::nullopt;

    if (header.type != 0x4D42) // 'BM' in little endian
      return std::nullopt;

    return header;
  }

  static std::optional<_info_header>
  _read_info_header(std::fstream &file) noexcept {
    _info_header info{};
    if (!file.read((char *)&info, sizeof(_info_header)))
      return std::nullopt;

    return info;
  }

  static std::optional<std::vector<std::uint8_t>>
  _read_pixel_data(const _file_header &header, const _info_header &info,
                   std::fstream &file) noexcept {
    return std::nullopt;
  }
};
} // namespace image