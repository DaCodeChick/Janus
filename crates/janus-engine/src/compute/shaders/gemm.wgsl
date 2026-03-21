// General Matrix-Matrix Multiplication (GEMM): C = A * B
// A: M x K matrix
// B: K x N matrix
// C: M x N matrix (output)
//
// All data is f32 (unquantized).
// Optimized implementation using tiled matrix multiplication with shared memory.

// Tile size for shared memory blocking
const TILE_SIZE: u32 = 16u;

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

// Shared memory tiles for collaborative loading (L1 cache)
// Each tile is TILE_SIZE x TILE_SIZE = 16 x 16 = 256 elements
var<workgroup> tile_a: array<f32, 256>;  // Tile of matrix A
var<workgroup> tile_b: array<f32, 256>;  // Tile of matrix B

// Tiled GEMM using shared memory for better memory bandwidth utilization
// Each workgroup computes a TILE_SIZE x TILE_SIZE block of the output matrix
// Workgroup size: 16x16 = 256 threads
@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    // Global output position
    let row = global_id.y;  // Global row in output matrix C
    let col = global_id.x;  // Global column in output matrix C
    
    // Local position within the tile
    let local_row = local_id.y;
    let local_col = local_id.x;
    
    // Accumulator for the dot product
    var sum: f32 = 0.0;
    
    // Iterate over tiles along the K dimension
    let num_tiles = (uniforms.K + TILE_SIZE - 1u) / TILE_SIZE;
    
    for (var tile_idx = 0u; tile_idx < num_tiles; tile_idx = tile_idx + 1u) {
        // Calculate the starting K index for this tile
        let k_start = tile_idx * TILE_SIZE;
        
        // ===== COLLABORATIVE LOAD: Each thread loads ONE element into tile_a =====
        let a_row = workgroup_id.y * TILE_SIZE + local_row;
        let a_col = k_start + local_col;
        
        if (a_row < uniforms.M && a_col < uniforms.K) {
            tile_a[local_row * TILE_SIZE + local_col] = matrix_a[a_row * uniforms.K + a_col];
        } else {
            tile_a[local_row * TILE_SIZE + local_col] = 0.0;
        }
        
        // ===== COLLABORATIVE LOAD: Each thread loads ONE element into tile_b =====
        // CRITICAL: PyTorch/HuggingFace stores weights as [out_features, in_features]
        // This means B is physically [N, K], so we must transpose on read
        let b_row = workgroup_id.x * TILE_SIZE + local_col;  // Note: using local_col for row
        let b_col = k_start + local_row;                      // Note: using local_row for col
        
        if (b_row < uniforms.N && b_col < uniforms.K) {
            // Transposed read: B[col, k] instead of B[k, col]
            tile_b[local_row * TILE_SIZE + local_col] = matrix_b[b_row * uniforms.K + b_col];
        } else {
            tile_b[local_row * TILE_SIZE + local_col] = 0.0;
        }
        
        // ===== BARRIER: Wait for all threads to finish loading the tiles =====
        workgroupBarrier();
        
        // ===== COMPUTE: Accumulate dot product using shared memory =====
        for (var k = 0u; k < TILE_SIZE; k = k + 1u) {
            sum = sum + tile_a[local_row * TILE_SIZE + k] * tile_b[k * TILE_SIZE + local_col];
        }
        
        // ===== BARRIER: Wait for all threads to finish computing before loading next tile =====
        workgroupBarrier();
    }
    
    // ===== WRITE OUTPUT: Store the accumulated result to global memory =====
    if (row < uniforms.M && col < uniforms.N) {
        matrix_c[row * uniforms.N + col] = sum;
    }
}
