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
// RMSNorm (Root Mean Square Normalization) - Batched per-sequence
// ============================================================================
// Formula: output[i] = (input[i] / sqrt(mean(input^2) + epsilon)) * gamma[i]
//
// For batched processing: Each sequence in the batch is normalized independently
// Input/Output: [batch_size, hidden_dim]
// Gamma weights: [hidden_dim] (shared across batch)
//
// This requires two passes PER SEQUENCE:
// 1. Compute sum of squares (with reduction) within each sequence
// 2. Normalize each element and apply gamma weights

struct RmsNormUniforms {
    batch_size: u32,  // Number of sequences in batch
    hidden_dim: u32,  // Hidden dimension (size per sequence)
    epsilon: f32,     // Small constant for numerical stability (typically 1e-6)
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> rms_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> rms_output: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<uniform> rms_uniforms: RmsNormUniforms;

// Shared memory for parallel reduction (one per workgroup/sequence)
var<workgroup> shared_sum: array<f32, 256>;

// Each workgroup processes one sequence
// Workgroup size: 256 threads
@compute @workgroup_size(256, 1, 1)
fn rmsnorm(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let batch_idx = workgroup_id.x;  // Each workgroup handles one sequence
    
    let is_active = batch_idx < rms_uniforms.batch_size;
    
    let hidden_dim = rms_uniforms.hidden_dim;
    let base_idx = batch_idx * hidden_dim;
    
    // Phase 1: Compute sum of squares using parallel reduction
    var local_sum: f32 = 0.0;
    
    // Each thread accumulates sum of squares for its stride within this sequence
    var idx = tid;
    if (is_active) {
        while (idx < hidden_dim) {
            let global_idx = base_idx + idx;
            let val = rms_input[global_idx];
            local_sum = local_sum + val * val;
            idx = idx + 256u;  // workgroup_size
        }
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
    
    // Thread 0 has the final sum of squares for this sequence
    var rms: f32;
    if (tid == 0u && is_active) {
        let mean_square = shared_sum[0] / f32(hidden_dim);
        rms = sqrt(mean_square + rms_uniforms.epsilon);
        // Store RMS in shared memory for all threads to access
        shared_sum[0] = rms;
    } else if (tid == 0u) {
        shared_sum[0] = 1.0;
    }
    workgroupBarrier();
    
    // Phase 2: Normalize each element and apply gamma weights
    rms = shared_sum[0];
    idx = tid;
    if (is_active) {
        while (idx < hidden_dim) {
            let global_idx = base_idx + idx;
            let normalized = rms_input[global_idx] / rms;
            // Gamma weights are shared across batch, indexed by position in hidden_dim
            rms_output[global_idx] = normalized * gamma[idx];
            idx = idx + 256u;
        }
    }
}
