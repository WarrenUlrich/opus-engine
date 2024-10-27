// compute_shader.wgsl
[[group(0), binding(1)]] var<storage, read> lightBuffer: array<LightUniform>;
[[group(0), binding(2)]] var<storage, read_write> tileBuffer: array<TileUniform>;

[[stage(compute), workgroup_size(16, 16)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
  
}
