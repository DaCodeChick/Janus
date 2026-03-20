// Element-wise tensor addition: output = a + b
//
// This shader performs element-wise addition of two tensors.
// Used for residual connections in transformer blocks.

struct Uniforms {
    size: u32,  // Number of elements in each tensor
}

@group(0) @binding(0) var<storage, read> tensor_a: array<f32>;
@group(0) @binding(1) var<storage, read> tensor_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= uniforms.size) {
        return;
    }
    
    output[idx] = tensor_a[idx] + tensor_b[idx];
}
