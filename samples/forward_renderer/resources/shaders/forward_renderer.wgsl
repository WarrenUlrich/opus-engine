struct CameraUniform {
  view_proj: mat4x4<f32>,
  position: vec3<f32>,
};

struct MaterialUniform {
  albedo: vec3<f32>,
  metallic: f32,
  roughness: f32
};

struct ModelUniform {
  transform: mat4x4<f32>,
  material_idx: u32
};

struct LightUniform {
  position: vec3<f32>,
  intensity: f32,
  color: vec3<f32>,
  radius: f32,
  direction: vec3<f32>,
  cutoff_angle: f32,
  light_type: u32,
};

struct VertexInput {
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
};

struct VertexOutput {
  @builtin(position) Position: vec4<f32>,
  @location(0) fragUV: vec2<f32>,
  @location(1) fragNormal: vec3<f32>,
  @location(2) fragPos: vec3<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<storage, read> lights: array<LightUniform, 256>;
@group(1) @binding(0) var<uniform> model: ModelUniform;
@group(2) @binding(0) var<storage, read> materials: array<MaterialUniform, 64>;

const PI = 3.14159265359;

fn calculateMVP(pos: vec3<f32>) -> vec4<f32> {
    return camera.view_proj * model.transform * vec4<f32>(pos, 1.0);
}

fn calculateNormal(normal: vec3<f32>) -> vec3<f32> {
    return normalize((model.transform * vec4<f32>(normal, 0.0)).xyz);
}

fn calculateFragPosition(pos: vec3<f32>) -> vec3<f32> {
    return (model.transform * vec4<f32>(pos, 1.0)).xyz;
}

fn calculateCheckerPattern(uv: vec2<f32>) -> f32 {
  let scaledUV = floor(30.0 * uv);
  return 0.2 + 0.5 * ((scaledUV.x + scaledUV.y) - 2.0 * floor((scaledUV.x + scaledUV.y) / 2.0));
}

// PBR Functions
fn DistributionGGX(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
  let a = roughness * roughness;
  let a2 = a * a;
  let NdotH = max(dot(N, H), 0.0);
  let NdotH2 = NdotH * NdotH;
  let denom = NdotH2 * (a2 - 1.0) + 1.0;
  return a2 / (PI * denom * denom);
}

fn GeometrySchlickGGX(NdotV: f32, roughness: f32) -> f32 {
  let r = (roughness + 1.0);
  let k = (r * r) / 8.0;
  return NdotV / (NdotV * (1.0 - k) + k);
}

fn GeometrySmith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
  let NdotV = max(dot(N, V), 0.0);
  let NdotL = max(dot(N, L), 0.0);
  let ggx2 = GeometrySchlickGGX(NdotV, roughness);
  let ggx1 = GeometrySchlickGGX(NdotL, roughness);
  return ggx1 * ggx2;
}

fn fresnelSchlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
  return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

fn computeLightingContribution(
  light: LightUniform,
  N: vec3<f32>,
  V: vec3<f32>,
  baseColor: vec3<f32>,
  metallic: f32,
  roughness: f32,
  fragPos: vec3<f32>
) -> vec3<f32> {
  var L: vec3<f32>;
  var radiance: vec3<f32>;

  if (light.light_type == 0u) { // Point Light
      let lightDir = light.position - fragPos;
      let distance = length(lightDir);
      L = normalize(lightDir);
      let attenuation = light.intensity / (1.0 + light.radius * distance * distance);
      radiance = light.color * attenuation;
  } else if (light.light_type == 1u) { // Directional Light
      L = normalize(-light.direction);
      radiance = light.color * (light.intensity * 0.3);
  } else if (light.light_type == 2u) { // Spotlight
      let lightDir = light.position - fragPos;
      let distance = length(lightDir);
      L = normalize(lightDir);
      let attenuation = light.intensity / (1.0 + light.radius * distance * distance);
      let theta = dot(L, normalize(-light.direction));
      let cutoff = cos(radians(light.cutoff_angle));
      let intensity = clamp((theta - cutoff) / (1.0 - cutoff), 0.0, 1.0);
      radiance = light.color * attenuation * intensity;
  } else {
      return vec3<f32>(0.0);
  }

  let NdotL = max(dot(N, L), 0.0);
  if (NdotL > 0.0) {
      let H = normalize(V + L);
      let NDF = DistributionGGX(N, H, roughness);
      let G = GeometrySmith(N, V, L, roughness);
      let F0 = mix(vec3<f32>(0.04), baseColor, metallic);
      let F = fresnelSchlick(max(dot(H, V), 0.0), F0);

      let numerator = NDF * G * F;
      let denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + 0.001;
      let specular = numerator / denominator;

      var kD = vec3<f32>(1.0) - F;
      kD *= 1.0 - metallic;

      let diffuse = kD * baseColor / PI;
      return (diffuse + specular) * radiance * NdotL;
  }

  return vec3<f32>(0.0);
}

// Vertex Shader
@vertex
fn vs(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  output.Position = calculateMVP(input.pos);
  output.fragUV = input.uv;
  output.fragNormal = calculateNormal(input.normal);
  output.fragPos = calculateFragPosition(input.pos);
  return output;
}

// Fragment Shader
@fragment
fn fs(input: VertexOutput) -> @location(0) vec4<f32> {
  // Get the material for this model
  let material = materials[model.material_idx];
  
  // Use material properties instead of checker pattern and hardcoded values
  let baseColor = material.albedo;
  let metallic = material.metallic;
  let roughness = material.roughness;
  
  let N = normalize(input.fragNormal);
  let V = normalize(camera.position - input.fragPos);

  var Lo = vec3<f32>(0.0);
  for (var i = 0u; i < 256u; i = i + 1u) {
      Lo += computeLightingContribution(
          lights[i],
          N,
          V,
          baseColor,
          metallic,
          roughness,
          input.fragPos
      );
  }

  let mapped = Lo / (Lo + vec3<f32>(1.0)); // Simple Reinhard tone mapping
  let gammaCorrectedColor = pow(mapped, vec3<f32>(1.0 / 2.2));

  return vec4<f32>(gammaCorrectedColor, 1.0);
}