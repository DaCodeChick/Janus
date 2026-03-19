// Matrix multiplication kernel (C = A * B)
// A: M x K matrix
// B: K x N matrix  
// C: M x N matrix (output)

struct MatMulUniforms {
    M: u32,  // Rows of A
    K: u32,  // Cols of A / Rows of B
    N: u32,  // Cols of B
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(1) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> matrix_c: array<f32>;
@group(0) @binding(3) var<uniform> uniforms: MatMulUniforms;

const TILE_SIZE: u32 = 16u;

var<workgroup> tile_a: array<array<f32, TILE_SIZE>, TILE_SIZE>;
var<workgroup> tile_b: array<array<f32, TILE_SIZE>, TILE_SIZE>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let row = global_id.y;
    let col = global_id.x;
    let local_row = local_id.y;
    let local_col = local_id.x;
    
    // Check bounds
    if (row >= uniforms.M || col >= uniforms.N) {
        return;
    }
    
    var sum: f32 = 0.0;
    
    // Number of tiles needed
    let num_tiles = (uniforms.K + TILE_SIZE - 1u) / TILE_SIZE;
    
    // Iterate over tiles
    for (var tile: u32 = 0u; tile < num_tiles; tile = tile + 1u) {
        // Load tile from A into shared memory
        let a_col = tile * TILE_SIZE + local_col;
        if (row < uniforms.M && a_col < uniforms.K) {
            tile_a[local_row][local_col] = matrix_a[row * uniforms.K + a_col];
        } else {
            tile_a[local_row][local_col] = 0.0;
        }
        
        // Load tile from B into shared memory
        let b_row = tile * TILE_SIZE + local_row;
        if (b_row < uniforms.K && col < uniforms.N) {
            tile_b[local_row][local_col] = matrix_b[b_row * uniforms.N + col];
        } else {
            tile_b[local_row][local_col] = 0.0;
        }
        
        // Synchronize workgroup
        workgroupBarrier();
        
        // Compute partial dot product for this tile
        for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
            sum = sum + tile_a[local_row][k] * tile_b[k][local_col];
        }
        
        // Synchronize before loading next tile
        workgroupBarrier();
    }
    
    // Write result
    if (row < uniforms.M && col < uniforms.N) {
        matrix_c[row * uniforms.N + col] = sum;
    }
}
