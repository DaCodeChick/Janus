// Q8_0 Quantized Matrix-Vector Multiplication
// Matrix A: Quantized in Q8_0 format (M x K)
// Vector B: F32 format (K elements)
// Vector C: F32 output (M elements)
//
// Q8_0 Block Structure (from llama.cpp ggml-quants.h):
// Block of 32 elements (QK8_0)
// 
// struct block_q8_0 {
//     half d;             // Delta (scale factor, f16, 2 bytes)
//     int8_t qs[32];      // Quantized values (32 bytes)
// };
// Total: 34 bytes per block
//
// Memory layout (as u32 words for WGSL):
// Word 0: d (f16, lower 16 bits) + unused padding (upper 16 bits)
// Words 1-8: 32 bytes of int8 quantized weights (8 u32 words, 4 bytes each)
// Total: 36 bytes = 9 u32 words (padded to align)

struct Q8_0Uniforms {
    M: u32,           // Number of rows in matrix A
    K: u32,           // Number of columns in matrix A (must be multiple of 32)
    num_blocks: u32,  // K / 32
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> matrix_a_q8_0: array<u32>;  // Quantized matrix A (Q8_0 blocks)
@group(0) @binding(1) var<storage, read> vector_b: array<f32>;       // Input vector B (K elements)
@group(0) @binding(2) var<storage, read_write> vector_c: array<f32>; // Output vector C (M elements)
@group(0) @binding(3) var<uniform> uniforms: Q8_0Uniforms;

// Convert f16 bits to f32
fn f16_to_f32(bits: u32) -> f32 {
    let sign = (bits >> 15u) & 1u;
    let exponent = (bits >> 10u) & 0x1Fu;
    let mantissa = bits & 0x3FFu;
    
    // Handle special cases
    if (exponent == 0u) {
        if (mantissa == 0u) {
            // Zero
            return select(0.0, -0.0, sign == 1u);
        } else {
            // Denormalized number
            let f32_mantissa = f32(mantissa) / 1024.0;
            let value = f32_mantissa * pow(2.0, -14.0);
            return select(value, -value, sign == 1u);
        }
    } else if (exponent == 31u) {
        // Infinity or NaN
        return select(1e38, -1e38, sign == 1u); // Treat as large number
    }
    
    // Normalized number
    let f32_exponent = i32(exponent) - 15 + 127;
    let f32_mantissa = mantissa << 13u;
    let f32_bits = (sign << 31u) | (u32(f32_exponent) << 23u) | f32_mantissa;
    
    return bitcast<f32>(f32_bits);
}

// Convert u32 containing bytes to signed int8 values
fn extract_i8(word: u32, byte_idx: u32) -> i32 {
    let byte = (word >> (byte_idx * 8u)) & 0xFFu;
    // Sign extend from 8 bits to 32 bits
    if ((byte & 0x80u) != 0u) {
        return i32(byte | 0xFFFFFF00u);
    } else {
        return i32(byte);
    }
}

// Dequantize and compute dot product for one Q8_0 block (32 elements)
fn process_q8_0_block(
    block_idx: u32,
    vec_offset: u32
) -> f32 {
    // Each block is 9 u32 words = 36 bytes (34 bytes data + 2 bytes padding)
    let base = block_idx * 9u;
    
    // Load scale (delta) - first word, lower 16 bits
    let d_word = matrix_a_q8_0[base];
    let d = f16_to_f32(d_word & 0xFFFFu);
    
    var sum: f32 = 0.0;
    
    // Process 32 quantized int8 values
    // They're stored in words 1-8 (8 u32 words = 32 bytes)
    for (var i: u32 = 0u; i < 8u; i++) {
        let word = matrix_a_q8_0[base + 1u + i];
        
        // Each word contains 4 int8 values
        for (var byte_idx: u32 = 0u; byte_idx < 4u; byte_idx++) {
            let weight_i8 = extract_i8(word, byte_idx);
            let weight_f32 = d * f32(weight_i8);
            
            let elem_idx = i * 4u + byte_idx;
            let vec_val = vector_b[vec_offset + elem_idx];
            
            sum += weight_f32 * vec_val;
        }
    }
    
    return sum;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    
    // Bounds check
    if (row >= uniforms.M) {
        return;
    }
    
    var sum: f32 = 0.0;
    
    // Process all Q8_0 blocks in this row
    for (var block: u32 = 0u; block < uniforms.num_blocks; block++) {
        let block_idx = row * uniforms.num_blocks + block;
        let vec_offset = block * 32u;
        
        sum += process_q8_0_block(block_idx, vec_offset);
    }
    
    // Write result
    vector_c[row] = sum;
}
