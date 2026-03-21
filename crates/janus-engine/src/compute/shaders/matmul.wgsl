// Matrix-Vector Multiplication (y = M * x)
// M: rows x cols matrix (weights, packed f16)
// x: cols-element vector (input, f32)
// y: rows-element vector (output, f32)
//
// Matrix M is packed f16 (2 f16s per u32) for 50% VRAM reduction.
// Input vector x and output y are f32.
// All computation happens in f32 precision (mixed-precision inference).

struct MatVecUniforms {
    rows: u32,    // Number of rows in matrix
    cols: u32,    // Number of columns in matrix
    _pad0: u32,   // Padding for alignment
    _pad1: u32,   // Padding for alignment
}

@group(0) @binding(0) var<storage, read> matrix: array<u32>;      // Matrix M (rows * cols / 2 elements, packed f16)
@group(0) @binding(1) var<storage, read> vector: array<f32>;      // Input vector x (cols elements, f32)
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // Output vector y (rows elements, f32)
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
        // Calculate global index in the weight matrix
        let global_idx = row_offset + col;
        
        // Read packed u32 (contains 2 f16 values)
        let packed = matrix[global_idx / 2u];
        
        // Unpack using WebGPU builtin: returns vec2<f32> with converted values
        let unpacked = unpack2x16float(packed);
        
        // Select the correct f16 value (low 16 bits = even index, high 16 bits = odd index)
        let is_odd = (global_idx % 2u) != 0u;
        let weight = select(unpacked.x, unpacked.y, is_odd);
        
        sum = sum + weight * vector[col];
    }
    
    // Write result
    output[row] = sum;
}
