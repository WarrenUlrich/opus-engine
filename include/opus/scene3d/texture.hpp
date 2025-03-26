#pragma once
#include <array>
#include <vector>
#include <webgpu/webgpu_cpp.h>

#include "../math/vector4.hpp"

namespace scene3d {
class texture {
public:
  enum class type { texture_2d, cubemap };

  texture() = default;
  ~texture() = default;

  // Create a texture from a solid color
  static texture create_from_color(wgpu::Device &device,
                                   const math::vector4<float> &color,
                                   uint32_t width = 1, uint32_t height = 1,
                                   wgpu::FilterMode filter_mode = wgpu::FilterMode::Linear,
                                   wgpu::AddressMode address_mode = wgpu::AddressMode::ClampToEdge) {
    texture result;
    result._width = width;
    result._height = height;
    result._type = type::texture_2d;

    // Create texture data - each pixel is the same color
    std::vector<uint8_t> data(width * height * 4);
    for (size_t i = 0; i < width * height; ++i) {
      // Convert float color [0.0-1.0] to uint8_t [0-255]
      data[i * 4 + 0] = static_cast<uint8_t>(color.x * 255.0f);
      data[i * 4 + 1] = static_cast<uint8_t>(color.y * 255.0f);
      data[i * 4 + 2] = static_cast<uint8_t>(color.z * 255.0f);
      data[i * 4 + 3] = static_cast<uint8_t>(color.w * 255.0f);
    }

    // Create the texture
    wgpu::TextureDescriptor textureDesc = {};
    textureDesc.dimension = wgpu::TextureDimension::e2D;
    textureDesc.size.width = width;
    textureDesc.size.height = height;
    textureDesc.size.depthOrArrayLayers = 1;
    textureDesc.sampleCount = 1;
    textureDesc.format = wgpu::TextureFormat::RGBA8Unorm;
    textureDesc.mipLevelCount = 1;
    textureDesc.usage =
        wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;
    result._texture = device.CreateTexture(&textureDesc);

    // Upload data to the texture
    wgpu::ImageCopyTexture destination = {};
    destination.texture = result._texture;
    destination.mipLevel = 0;
    destination.origin = {0, 0, 0};
    destination.aspect = wgpu::TextureAspect::All;

    wgpu::TextureDataLayout dataLayout = {};
    dataLayout.offset = 0;
    dataLayout.bytesPerRow = width * 4;
    dataLayout.rowsPerImage = height;

    wgpu::Extent3D writeSize = {width, height, 1};

    device.GetQueue().WriteTexture(&destination, data.data(), data.size(),
                                   &dataLayout, &writeSize);

    // Create texture view
    wgpu::TextureViewDescriptor viewDesc = {};
    viewDesc.format = wgpu::TextureFormat::RGBA8Unorm;
    viewDesc.dimension = wgpu::TextureViewDimension::e2D;
    viewDesc.baseMipLevel = 0;
    viewDesc.mipLevelCount = 1;
    viewDesc.baseArrayLayer = 0;
    viewDesc.arrayLayerCount = 1;
    viewDesc.aspect = wgpu::TextureAspect::All;
    result._view = result._texture.CreateView(&viewDesc);

    // Create sampler
    wgpu::SamplerDescriptor samplerDesc = {};
    
    // Set filter modes directly using wgpu::FilterMode
    samplerDesc.minFilter = filter_mode;
    samplerDesc.magFilter = filter_mode;
    
    // Set mipmapFilter based on filter mode
    // Note: You may want to add a separate parameter for mipmap filter if needed
    samplerDesc.mipmapFilter = (filter_mode == wgpu::FilterMode::Linear) ? 
                                wgpu::MipmapFilterMode::Linear : 
                                wgpu::MipmapFilterMode::Nearest;

    // Set address modes directly using wgpu::AddressMode
    samplerDesc.addressModeU = address_mode;
    samplerDesc.addressModeV = address_mode;
    samplerDesc.addressModeW = address_mode;

    result._sampler = device.CreateSampler(&samplerDesc);

    return result;
  }

  // Getters for the private members
  wgpu::TextureView get_view() const { return _view; }
  wgpu::Sampler get_sampler() const { return _sampler; }
  uint32_t get_width() const { return _width; }
  uint32_t get_height() const { return _height; }
  type get_type() const { return _type; }

private:
  wgpu::Texture _texture;
  wgpu::TextureView _view;
  wgpu::Sampler _sampler;
  uint32_t _width = 0;
  uint32_t _height = 0;
  type _type = type::texture_2d;
};
} // namespace scene3d