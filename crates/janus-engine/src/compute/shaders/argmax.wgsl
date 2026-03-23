// GPU-side argmax for greedy sampling
//
// This shader finds the index of the maximum value in the logits array,
// avoiding the need to transfer 128KB of logits from GPU to CPU.
//
// Algorithm: Two-phase reduction
// Phase 1: Each workgroup finds local max in parallel
// Phase 2: Single workgroup finds global max from local maxes

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
var<workgroup> local_max_val: array<f32, 256>;
var<workgroup> local_max_idx: array<u32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let batch_idx = workgroup_id.y;
    let tid = local_id.x;
    let vocab_size = uniforms.vocab_size;
    
    // Each batch processes independently
    let logits_offset = batch_idx * vocab_size;
    
    // Phase 1: Each thread finds max in its stride
    var max_val: f32 = -1e38; // Very negative number
    var max_idx: u32 = 0u;
    
    // Stride through the vocab with workgroup_size steps
    var i = tid;
    while (i < vocab_size) {
        let val = logits[logits_offset + i];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
        i += 256u; // workgroup_size
    }
    
    // Store local max in shared memory
    local_max_val[tid] = max_val;
    local_max_idx[tid] = max_idx;
    
    workgroupBarrier();
    
    // Phase 2: Parallel reduction within workgroup
    // Only the first workgroup does the reduction (for each batch)
    if (workgroup_id.x == 0u) {
        // Reduce in shared memory (binary tree reduction)
        var stride = 128u;
        while (stride > 0u) {
            if (tid < stride && tid + stride < 256u) {
                let other_val = local_max_val[tid + stride];
                if (other_val > local_max_val[tid]) {
                    local_max_val[tid] = other_val;
                    local_max_idx[tid] = local_max_idx[tid + stride];
                }
            }
            workgroupBarrier();
            stride = stride / 2u;
        }
        
        // Thread 0 writes the final result
        if (tid == 0u) {
            output[batch_idx] = local_max_idx[0];
        }
    }
}
