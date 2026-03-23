// GPU-side argmax for greedy sampling (optimized single-pass version)
//
// This shader finds the index of the maximum value in the logits array,
// avoiding the need to transfer 128KB of logits from GPU to CPU.
//
// Algorithm: Single-pass workgroup reduction
// - All 256 threads cooperate to scan the entire vocab in parallel
// - Use shared memory for intra-workgroup reduction
// - Much simpler and faster than multi-phase approaches

struct ArgmaxUniforms {
    vocab_size: u32,
    batch_size: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> logits: array<f32>;        // Input logits [batch_size * vocab_size]
@group(0) @binding(1) var<storage, read_write> output: array<u32>;  // Output token IDs [batch_size]
@group(0) @binding(2) var<uniform> uniforms: ArgmaxUniforms;

// Shared memory for workgroup-level reduction
var<workgroup> shared_val: array<f32, 256>;
var<workgroup> shared_idx: array<u32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let batch_idx = workgroup_id.y;
    let tid = local_id.x;
    let vocab_size = uniforms.vocab_size;
    
    // Each batch processes independently
    let logits_offset = batch_idx * vocab_size;
    
    // Step 1: Each thread scans a portion of the vocab
    var max_val: f32 = -1e38; // Very negative number
    var max_idx: u32 = 0u;
    
    // Stride through vocab with 256-thread steps
    var i = tid;
    while (i < vocab_size) {
        let val = logits[logits_offset + i];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
        i += 256u; // workgroup_size
    }
    
    // Step 2: Write thread-local max to shared memory
    shared_val[tid] = max_val;
    shared_idx[tid] = max_idx;
    
    workgroupBarrier();
    
    // Step 3: Tree reduction in shared memory
    // Unrolled for better performance on smaller reductions
    if (tid < 128u) {
        if (shared_val[tid + 128u] > shared_val[tid]) {
            shared_val[tid] = shared_val[tid + 128u];
            shared_idx[tid] = shared_idx[tid + 128u];
        }
    }
    workgroupBarrier();
    
    if (tid < 64u) {
        if (shared_val[tid + 64u] > shared_val[tid]) {
            shared_val[tid] = shared_val[tid + 64u];
            shared_idx[tid] = shared_idx[tid + 64u];
        }
    }
    workgroupBarrier();
    
    if (tid < 32u) {
        if (shared_val[tid + 32u] > shared_val[tid]) {
            shared_val[tid] = shared_val[tid + 32u];
            shared_idx[tid] = shared_idx[tid + 32u];
        }
    }
    workgroupBarrier();
    
    // Final warp reduction (no barrier needed within warp)
    if (tid < 16u) {
        if (shared_val[tid + 16u] > shared_val[tid]) {
            shared_val[tid] = shared_val[tid + 16u];
            shared_idx[tid] = shared_idx[tid + 16u];
        }
    }
    workgroupBarrier();
    
    if (tid < 8u) {
        if (shared_val[tid + 8u] > shared_val[tid]) {
            shared_val[tid] = shared_val[tid + 8u];
            shared_idx[tid] = shared_idx[tid + 8u];
        }
    }
    workgroupBarrier();
    
    if (tid < 4u) {
        if (shared_val[tid + 4u] > shared_val[tid]) {
            shared_val[tid] = shared_val[tid + 4u];
            shared_idx[tid] = shared_idx[tid + 4u];
        }
    }
    workgroupBarrier();
    
    if (tid < 2u) {
        if (shared_val[tid + 2u] > shared_val[tid]) {
            shared_val[tid] = shared_val[tid + 2u];
            shared_idx[tid] = shared_idx[tid + 2u];
        }
    }
    workgroupBarrier();
    
    // Thread 0 does final comparison and writes result
    if (tid == 0u) {
        var final_idx = shared_idx[0];
        if (shared_val[1] > shared_val[0]) {
            final_idx = shared_idx[1];
        }
        output[batch_idx] = final_idx;
    }
}
