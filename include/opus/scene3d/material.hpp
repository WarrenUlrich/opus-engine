#pragma once
#include <array>
#include <memory>
#include <optional>
#include <string>
#include <webgpu/webgpu_cpp.h>
#include "texture.hpp" // Include our texture class

namespace scene3d {

class material {
public:
    // Default constructor
    material() = default;
    ~material() = default;

    // PBR material properties
    struct pbr_parameters {
        std::array<float, 4> base_color;    // Base color (RGB) + Alpha
        float metallic;                      // 0 = dielectric, 1 = metallic
        float roughness;                     // 0 = smooth, 1 = rough
        float reflectance;                   // Specular reflectance for dielectrics (0-1)
        float emissive_intensity;            // Emissive intensity multiplier
        std::array<float, 3> emissive_color; // Emissive color (RGB)
        float ao_strength;                   // Ambient occlusion strength
        float normal_scale;                  // Normal map intensity
        std::array<float, 3> padding;        // Padding to ensure proper alignment

        // Default constructor to initialize all values
        pbr_parameters() 
            : base_color{1.0f, 1.0f, 1.0f, 1.0f}
            , metallic(0.0f)
            , roughness(0.5f)
            , reflectance(0.5f)
            , emissive_intensity(0.0f)
            , emissive_color{1.0f, 1.0f, 1.0f}
            , ao_strength(1.0f)
            , normal_scale(1.0f)
            , padding{0.0f, 0.0f, 0.0f}
        {}
    };

    // Create a basic PBR material with default parameters
    static material create_basic(wgpu::Device &device, const std::string &name = "DefaultMaterial") {
        material result;
        result._name = name;
        
        // Create a default white texture
        result._albedo_texture = std::make_shared<texture>(
            texture::create_from_color(device, {1.0f, 1.0f, 1.0f, 1.0f}));
        
        // Create a default black texture with alpha = roughness for metallic-roughness workflow
        result._metallic_roughness_texture = std::make_shared<texture>(
            texture::create_from_color(device, {0.0f, 0.5f, 0.0f, 1.0f}));
        
        // Create a default normal map (flat surface, pointing up: RGB = 0.5, 0.5, 1.0)
        result._normal_texture = std::make_shared<texture>(
            texture::create_from_color(device, {0.5f, 0.5f, 1.0f, 1.0f}));
        
        // Create a default white occlusion texture
        result._occlusion_texture = std::make_shared<texture>(
            texture::create_from_color(device, {1.0f, 1.0f, 1.0f, 1.0f}));
        
        // Create a default black emissive texture
        result._emissive_texture = std::make_shared<texture>(
            texture::create_from_color(device, {0.0f, 0.0f, 0.0f, 1.0f}));
        
        // Initialize parameters with default constructor
        result._parameters = pbr_parameters();
        
        wgpu::BufferDescriptor bufferDesc = {};
        bufferDesc.size = sizeof(pbr_parameters);
        bufferDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        bufferDesc.mappedAtCreation = false;
        result._uniform_buffer = device.CreateBuffer(&bufferDesc);
        
        // Update the uniform buffer with the default parameters
        device.GetQueue().WriteBuffer(result._uniform_buffer, 0, 
                                     &result._parameters, sizeof(pbr_parameters));
        
        // Create the bind group layout for the material
        wgpu::BindGroupLayoutEntry entries[10] = {};
        
        // Uniform buffer
        entries[0].binding = 0;
        entries[0].visibility = wgpu::ShaderStage::Fragment;
        entries[0].buffer.type = wgpu::BufferBindingType::Uniform;
        entries[0].buffer.minBindingSize = sizeof(pbr_parameters);
        
        // Albedo texture and sampler
        entries[1].binding = 1;
        entries[1].visibility = wgpu::ShaderStage::Fragment;
        entries[1].texture.sampleType = wgpu::TextureSampleType::Float;
        entries[1].texture.viewDimension = wgpu::TextureViewDimension::e2D;
        
        entries[2].binding = 2;
        entries[2].visibility = wgpu::ShaderStage::Fragment;
        entries[2].sampler.type = wgpu::SamplerBindingType::Filtering;
        
        // Metallic-roughness texture and sampler
        entries[3].binding = 3;
        entries[3].visibility = wgpu::ShaderStage::Fragment;
        entries[3].texture.sampleType = wgpu::TextureSampleType::Float;
        entries[3].texture.viewDimension = wgpu::TextureViewDimension::e2D;
        
        entries[4].binding = 4;
        entries[4].visibility = wgpu::ShaderStage::Fragment;
        entries[4].sampler.type = wgpu::SamplerBindingType::Filtering;
        
        // Normal texture and sampler
        entries[5].binding = 5;
        entries[5].visibility = wgpu::ShaderStage::Fragment;
        entries[5].texture.sampleType = wgpu::TextureSampleType::Float;
        entries[5].texture.viewDimension = wgpu::TextureViewDimension::e2D;
        
        entries[6].binding = 6;
        entries[6].visibility = wgpu::ShaderStage::Fragment;
        entries[6].sampler.type = wgpu::SamplerBindingType::Filtering;
        
        // Occlusion texture and sampler
        entries[7].binding = 7;
        entries[7].visibility = wgpu::ShaderStage::Fragment;
        entries[7].texture.sampleType = wgpu::TextureSampleType::Float;
        entries[7].texture.viewDimension = wgpu::TextureViewDimension::e2D;
        
        entries[8].binding = 8;
        entries[8].visibility = wgpu::ShaderStage::Fragment;
        entries[8].sampler.type = wgpu::SamplerBindingType::Filtering;
        
        // Emissive texture and sampler
        entries[9].binding = 9;
        entries[9].visibility = wgpu::ShaderStage::Fragment;
        entries[9].texture.sampleType = wgpu::TextureSampleType::Float;
        entries[9].texture.viewDimension = wgpu::TextureViewDimension::e2D;
        
        wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc = {};
        bindGroupLayoutDesc.entryCount = 10;
        bindGroupLayoutDesc.entries = entries;
        result._bind_group_layout = device.CreateBindGroupLayout(&bindGroupLayoutDesc);
        
        // Create the bind group
        result._update_bind_group(device);
        
        return result;
    }
    
    // Create a material with custom textures and parameters
    static material create(
        wgpu::Device &device,
        const std::string &name,
        std::shared_ptr<texture> albedo,
        std::shared_ptr<texture> metallic_roughness = nullptr,
        std::shared_ptr<texture> normal = nullptr,
        std::shared_ptr<texture> occlusion = nullptr,
        std::shared_ptr<texture> emissive = nullptr,
        const pbr_parameters &params = pbr_parameters())
    {
        material result = create_basic(device, name);
        
        // Set textures if provided
        if (albedo) result._albedo_texture = albedo;
        if (metallic_roughness) result._metallic_roughness_texture = metallic_roughness;
        if (normal) result._normal_texture = normal;
        if (occlusion) result._occlusion_texture = occlusion;
        if (emissive) result._emissive_texture = emissive;
        
        // Set parameters
        result._parameters = params;
        
        // Update uniform buffer
        device.GetQueue().WriteBuffer(result._uniform_buffer, 0, 
                                     &result._parameters, sizeof(pbr_parameters));
        
        // Update bind group with new textures
        result._update_bind_group(device);
        
        return result;
    }
    
    // Setters for individual material properties
    void set_base_color(wgpu::Device &device, const std::array<float, 4> &color) {
        _parameters.base_color = color;
        _update_parameter_buffer(device);
    }
    
    void set_metallic(wgpu::Device &device, float metallic) {
        _parameters.metallic = std::clamp(metallic, 0.0f, 1.0f);
        _update_parameter_buffer(device);
    }
    
    void set_roughness(wgpu::Device &device, float roughness) {
        _parameters.roughness = std::clamp(roughness, 0.0f, 1.0f);
        _update_parameter_buffer(device);
    }
    
    void set_reflectance(wgpu::Device &device, float reflectance) {
        _parameters.reflectance = std::clamp(reflectance, 0.0f, 1.0f);
        _update_parameter_buffer(device);
    }
    
    void set_emissive(wgpu::Device &device, const std::array<float, 3> &color, float intensity) {
        _parameters.emissive_color = color;
        _parameters.emissive_intensity = intensity;
        _update_parameter_buffer(device);
    }
    
    void set_normal_scale(wgpu::Device &device, float scale) {
        _parameters.normal_scale = scale;
        _update_parameter_buffer(device);
    }
    
    void set_ao_strength(wgpu::Device &device, float strength) {
        _parameters.ao_strength = std::clamp(strength, 0.0f, 1.0f);
        _update_parameter_buffer(device);
    }
    
    // Texture setters
    void set_albedo_texture(wgpu::Device &device, std::shared_ptr<texture> tex) {
        if (tex) {
            _albedo_texture = tex;
            _update_bind_group(device);
        }
    }
    
    void set_metallic_roughness_texture(wgpu::Device &device, std::shared_ptr<texture> tex) {
        if (tex) {
            _metallic_roughness_texture = tex;
            _update_bind_group(device);
        }
    }
    
    void set_normal_texture(wgpu::Device &device, std::shared_ptr<texture> tex) {
        if (tex) {
            _normal_texture = tex;
            _update_bind_group(device);
        }
    }
    
    void set_occlusion_texture(wgpu::Device &device, std::shared_ptr<texture> tex) {
        if (tex) {
            _occlusion_texture = tex;
            _update_bind_group(device);
        }
    }
    
    void set_emissive_texture(wgpu::Device &device, std::shared_ptr<texture> tex) {
        if (tex) {
            _emissive_texture = tex;
            _update_bind_group(device);
        }
    }
    
    // Getters
    const std::string& get_name() const { return _name; }
    const pbr_parameters& get_parameters() const { return _parameters; }
    wgpu::BindGroup get_bind_group() const { return _bind_group; }
    wgpu::BindGroupLayout get_bind_group_layout() const { return _bind_group_layout; }
    
    std::shared_ptr<texture> get_albedo_texture() const { return _albedo_texture; }
    std::shared_ptr<texture> get_metallic_roughness_texture() const { return _metallic_roughness_texture; }
    std::shared_ptr<texture> get_normal_texture() const { return _normal_texture; }
    std::shared_ptr<texture> get_occlusion_texture() const { return _occlusion_texture; }
    std::shared_ptr<texture> get_emissive_texture() const { return _emissive_texture; }

private:
    std::string _name = "Material";
    pbr_parameters _parameters;
    
    // Textures
    std::shared_ptr<texture> _albedo_texture;
    std::shared_ptr<texture> _metallic_roughness_texture;
    std::shared_ptr<texture> _normal_texture;
    std::shared_ptr<texture> _occlusion_texture;
    std::shared_ptr<texture> _emissive_texture;
    
    // WebGPU resources
    wgpu::Buffer _uniform_buffer;
    wgpu::BindGroupLayout _bind_group_layout;
    wgpu::BindGroup _bind_group;
    
    // Update the parameter uniform buffer
    void _update_parameter_buffer(wgpu::Device &device) {
        device.GetQueue().WriteBuffer(_uniform_buffer, 0, &_parameters, sizeof(pbr_parameters));
    }
    
    // Update the bind group with current textures
    void _update_bind_group(wgpu::Device &device) {
        wgpu::BindGroupEntry entries[10] = {};
        
        // Parameter uniform buffer
        entries[0].binding = 0;
        entries[0].buffer = _uniform_buffer;
        entries[0].offset = 0;
        entries[0].size = sizeof(pbr_parameters);
        
        // Albedo texture and sampler
        entries[1].binding = 1;
        entries[1].textureView = _albedo_texture->get_view();
        
        entries[2].binding = 2;
        entries[2].sampler = _albedo_texture->get_sampler();
        
        // Metallic-roughness texture and sampler
        entries[3].binding = 3;
        entries[3].textureView = _metallic_roughness_texture->get_view();
        
        entries[4].binding = 4;
        entries[4].sampler = _metallic_roughness_texture->get_sampler();
        
        // Normal texture and sampler
        entries[5].binding = 5;
        entries[5].textureView = _normal_texture->get_view();
        
        entries[6].binding = 6;
        entries[6].sampler = _normal_texture->get_sampler();
        
        // Occlusion texture and sampler
        entries[7].binding = 7;
        entries[7].textureView = _occlusion_texture->get_view();
        
        entries[8].binding = 8;
        entries[8].sampler = _occlusion_texture->get_sampler();
        
        // Emissive texture and sampler
        entries[9].binding = 9;
        entries[9].textureView = _emissive_texture->get_view();
        
        wgpu::BindGroupDescriptor bindGroupDesc = {};
        bindGroupDesc.layout = _bind_group_layout;
        bindGroupDesc.entryCount = 10;
        bindGroupDesc.entries = entries;
        
        _bind_group = device.CreateBindGroup(&bindGroupDesc);
    }
};

} // namespace scene3d