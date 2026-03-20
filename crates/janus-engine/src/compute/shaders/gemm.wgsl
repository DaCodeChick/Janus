// General Matrix-Matrix Multiplication (GEMM): C = A * B
// A: M x K matrix
// B: K x N matrix
// C: M x N matrix (output)
//
// All data is f32 (unquantized).
// Naive implementation without shared memory tiling.

struct GemmUniforms {
    M: u32,    // Rows of A
    K: u32,    // Cols of A / Rows of B
    N: u32,    // Cols of B
    _pad: u32, // Padding for alignment
}

@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;      // Matrix A (M * K elements, row-major)
@group(0) @binding(1) var<storage, read> matrix_b: array<f32>;      // Matrix B (K * N elements, row-major)
@group(0) @binding(2) var<storage, read_write> matrix_c: array<f32>; // Output matrix C (M * N elements, row-major)
@group(0) @binding(3) var<uniform> uniforms: GemmUniforms;

// Each thread computes one element of the output matrix C
// Workgroup size: 16x16 = 256 threads
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;  // Row index in output matrix C
    let col = global_id.x;  // Column index in output matrix C
    
    // Bounds check
    if (row >= uniforms.M || col >= uniforms.N) {
        return;
    }
    
    // Compute dot product of A[row, :] with B[:, col]
    var sum: f32 = 0.0;
    
    for (var k: u32 = 0u; k < uniforms.K; k = k + 1u) {
        let a_val = matrix_a[row * uniforms.K + k];
        // CRITICAL: PyTorch/HuggingFace stores weights as [out_features, in_features]
        // This means B is physically [N, K], so we must transpose on read
        let b_val = matrix_b[col * uniforms.K + k];
        sum = sum + a_val * b_val;
    }
    
    // Write result to output matrix
    matrix_c[row * uniforms.N + col] = sum;
}
