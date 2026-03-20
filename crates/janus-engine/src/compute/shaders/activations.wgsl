// Activation functions for neural networks
// All data is f32 (unquantized)

// ============================================================================
// SiLU (Swish) Activation: f(x) = x * sigmoid(x) = x / (1 + exp(-x))
// ============================================================================

struct SiluUniforms {
    size: u32,    // Number of elements
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> silu_uniforms: SiluUniforms;

// Sigmoid function: 1 / (1 + exp(-x))
fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

// SiLU activation: x * sigmoid(x)
@compute @workgroup_size(256, 1, 1)
fn silu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= silu_uniforms.size) {
        return;
    }
    
    let x = input[idx];
    output[idx] = x * sigmoid(x);
}

// ============================================================================
// RMSNorm (Root Mean Square Normalization)
// ============================================================================
// Formula: output[i] = input[i] / sqrt(mean(input^2) + epsilon)
//
// This requires two passes:
// 1. Compute sum of squares (with reduction)
// 2. Normalize each element

struct RmsNormUniforms {
    size: u32,      // Number of elements
    epsilon: f32,   // Small constant for numerical stability (typically 1e-6)
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> rms_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> rms_output: array<f32>;
@group(0) @binding(2) var<uniform> rms_uniforms: RmsNormUniforms;

// Shared memory for parallel reduction
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn rmsnorm(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let size = rms_uniforms.size;
    
    // Phase 1: Compute sum of squares using parallel reduction
    var local_sum: f32 = 0.0;
    
    // Each thread accumulates sum of squares for its stride
    var idx = global_id.x;
    while (idx < size) {
        let val = rms_input[idx];
        local_sum = local_sum + val * val;
        idx = idx + 256u;  // workgroup_size
    }
    
    // Store in shared memory
    shared_sum[tid] = local_sum;
    workgroupBarrier();
    
    // Parallel reduction in shared memory
    // Reduce 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
    var stride = 128u;
    while (stride > 0u) {
        if (tid < stride && tid + stride < 256u) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    
    // Thread 0 has the final sum of squares
    var rms: f32;
    if (tid == 0u) {
        let mean_square = shared_sum[0] / f32(size);
        rms = sqrt(mean_square + rms_uniforms.epsilon);
        // Store RMS in shared memory for all threads to access
        shared_sum[0] = rms;
    }
    workgroupBarrier();
    
    // Phase 2: Normalize each element
    rms = shared_sum[0];
    idx = global_id.x;
    while (idx < size) {
        rms_output[idx] = rms_input[idx] / rms;
        idx = idx + 256u;
    }
}
