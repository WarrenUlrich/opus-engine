
struct CameraUniform {
  mvp: mat4x4<f32>,
  model: mat4x4<f32>,
  position: vec3<f32>,
};

struct LightUniform {
  position: vec3<f32>,    // Position for point/spot lights, ignored for directional lights
  intensity: f32,         // Intensity of the light
  color: vec3<f32>,       // Light color
  radius: f32,            // Radius for point/spot lights, 0 for directional lights
  direction: vec3<f32>,   // Light direction for directional/spotlights, ignored for point lights
  cutoff_angle: f32,      // Cutoff angle for spotlights, 0 for other lights
  light_type: u32,        // 0 = Point light, 1 = Directional light, 2 = Spotlight
};

@group(0) @binding(0) var<uniform> uniforms: CameraUniform;
@group(0) @binding(1) var<storage, read> lights: array<LightUniform, 256>;

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

@vertex
fn vs(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  output.Position = calculateMVP(input.pos);
  output.fragUV = input.uv;
  output.fragNormal = calculateNormal(input.normal);
  output.fragPos = calculateFragPosition(input.pos);
  return output;
}

fn calculateMVP(pos: vec3<f32>) -> vec4<f32> {
  return uniforms.mvp * vec4<f32>(pos, 1.0);
}

fn calculateNormal(normal: vec3<f32>) -> vec3<f32> {
  return normalize((uniforms.model * vec4<f32>(normal, 0.0)).xyz);
}

fn calculateFragPosition(pos: vec3<f32>) -> vec3<f32> {
  return (uniforms.model * vec4<f32>(pos, 1.0)).xyz;
}

@fragment
fn fs(input: VertexOutput) -> @location(0) vec4<f32> {
  let checkerColor = calculateCheckerPattern(input.fragUV);
  let normal = normalize(input.fragNormal);
  var finalColor = vec3<f32>(0.1); // Ambient light base

  for (var i = 0u; i < 256u; i = i + 1u) {
    let light = lights[i];
    finalColor += calculateLighting(light, input.fragPos, normal);
  }

  let baseColor = vec3<f32>(checkerColor, checkerColor, checkerColor);
  finalColor = baseColor * finalColor;

  return vec4<f32>(finalColor, 1.0); // Output color with full opacity
}

fn calculateCheckerPattern(uv: vec2<f32>) -> f32 {
  let scaledUV = floor(30.0 * uv);
  return 0.2 + 0.5 * ((scaledUV.x + scaledUV.y) - 2.0 * floor((scaledUV.x + scaledUV.y) / 2.0));
}

fn calculateLighting(light: LightUniform, fragPos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
  var lightColor = vec3<f32>(0.0);

  if (light.light_type == 0u) {
    lightColor = calculatePointLight(light, fragPos, normal);
  } else if (light.light_type == 1u) {
    lightColor = calculateDirectionalLight(light, normal);
  } else if (light.light_type == 2u) {
    lightColor = calculateSpotLight(light, fragPos, normal);
  }

  return lightColor;
}

fn calculatePointLight(light: LightUniform, fragPos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
  let lightDir = normalize(light.position - fragPos);
  let distance = length(light.position - fragPos);
  let attenuation = light.intensity / (1.0 + light.radius * distance * distance);
  let diffuse = max(dot(normal, lightDir), 0.0);
  return light.color * diffuse * attenuation;
}

fn calculateDirectionalLight(light: LightUniform, normal: vec3<f32>) -> vec3<f32> {
  let lightDir = normalize(light.direction);
  let diffuse = max(dot(normal, lightDir), 0.0);
  return light.color * diffuse * light.intensity;
}

fn calculateSpotLight(light: LightUniform, fragPos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
  let lightDir = normalize(light.position - fragPos);
  let distance = length(light.position - fragPos);
  let attenuation = light.intensity / (1.0 + light.radius * distance * distance);
  let spotEffect = dot(normalize(light.direction), -lightDir);

  if (spotEffect > cos(radians(light.cutoff_angle))) {
    let diffuse = max(dot(normal, lightDir), 0.0);
    return light.color * diffuse * attenuation * spotEffect;
  }

  return vec3<f32>(0.0);
}