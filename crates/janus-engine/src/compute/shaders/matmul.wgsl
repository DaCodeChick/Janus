// Matrix-Vector Multiplication (y = M * x)
// M: rows x cols matrix (weights)
// x: cols-element vector (input)
// y: rows-element vector (output)
//
// All data is f32 (unquantized).

struct MatVecUniforms {
    rows: u32,    // Number of rows in matrix
    cols: u32,    // Number of columns in matrix
    _pad0: u32,   // Padding for alignment
    _pad1: u32,   // Padding for alignment
}

@group(0) @binding(0) var<storage, read> matrix: array<f32>;      // Matrix M (rows * cols elements)
@group(0) @binding(1) var<storage, read> vector: array<f32>;      // Input vector x (cols elements)
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // Output vector y (rows elements)
@group(0) @binding(3) var<uniform> uniforms: MatVecUniforms;

// Workgroup size: 256 threads per workgroup
// Each thread computes one output element (one dot product)
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    
    // Bounds check
    if (row >= uniforms.rows) {
        return;
    }
    
    // Compute dot product for this row
    var sum: f32 = 0.0;
    let row_offset = row * uniforms.cols;
    
    for (var col: u32 = 0u; col < uniforms.cols; col = col + 1u) {
        sum = sum + matrix[row_offset + col] * vector[col];
    }
    
    // Write result
    output[row] = sum;
}
