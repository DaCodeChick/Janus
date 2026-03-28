// General Matrix-Matrix Multiplication (GEMM): C = A * B^T
// A: [batch_size, M, K] matrix (dynamic activations, f32)
// B: [N, K] matrix (static weights, f32)
// C: [batch_size, M, N] matrix (output, f32)

const TILE_SIZE: u32 = 16u;

struct GemmUniforms {
    batch_size: u32,
    M: u32,
    K: u32,
    N: u32,
}

@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(1) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> matrix_c: array<f32>;
@group(0) @binding(3) var<uniform> uniforms: GemmUniforms;

var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let batch_idx = workgroup_id.z;
    let row = global_id.y;
    let col = global_id.x;

    let local_row = local_id.y;
    let local_col = local_id.x;

    var sum: f32 = 0.0;
    let num_tiles = (uniforms.K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var tile_idx = 0u; tile_idx < num_tiles && tile_idx < 10000u; tile_idx = tile_idx + 1u) {
        let k_start = tile_idx * TILE_SIZE;

        let a_row = workgroup_id.y * TILE_SIZE + local_row;
        let a_col = k_start + local_col;
        let batch_offset_a = batch_idx * uniforms.M * uniforms.K;

        if (a_row < uniforms.M && a_col < uniforms.K) {
            tile_a[local_row * TILE_SIZE + local_col] = matrix_a[batch_offset_a + a_row * uniforms.K + a_col];
        } else {
            tile_a[local_row * TILE_SIZE + local_col] = 0.0;
        }

        let b_n = workgroup_id.x * TILE_SIZE + local_col;
        let b_k = k_start + local_row;

        if (b_n < uniforms.N && b_k < uniforms.K) {
            let global_idx = b_n * uniforms.K + b_k;
            tile_b[local_col * TILE_SIZE + local_row] = matrix_b[global_idx];
        } else {
            tile_b[local_col * TILE_SIZE + local_row] = 0.0;
        }

        workgroupBarrier();

        for (var k = 0u; k < TILE_SIZE; k = k + 1u) {
            sum = sum + tile_a[local_row * TILE_SIZE + k] * tile_b[local_col * TILE_SIZE + k];
        }

        workgroupBarrier();
    }

    if (batch_idx < uniforms.batch_size && row < uniforms.M && col < uniforms.N) {
        let batch_offset_c = batch_idx * uniforms.M * uniforms.N;
        matrix_c[batch_offset_c + row * uniforms.N + col] = sum;
    }
}
