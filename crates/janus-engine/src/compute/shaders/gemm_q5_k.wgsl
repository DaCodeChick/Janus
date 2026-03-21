// Q5_K Quantized Matrix-Vector Multiplication
// Matrix A: Quantized in Q5_K format (M x K)
// Vector B: F32 format (K elements)
// Vector C: F32 output (M elements)
//
// Q5_K Block Structure (from llama.cpp ggml-quants.h):
// Superblock of 256 elements (QK_K) divided into 8 blocks of 32 elements
// 
// struct block_q5_K {
//     uint8_t scales[12];      // Scales and mins packed as 6-bit values
//     uint8_t qh[32];          // High bits for all 5-bit quants
//     uint8_t qs[128];         // Low 4-bit quants (256 values, 4 bits each = 128 bytes)
//     half d;                  // Super-scale delta (f16)
//     half dmin;               // Super-min (f16)
// };
// Total: 176 bytes per block
//
// Memory layout (as u32 words since WGSL lacks u8):
// Words 0-2:    12 bytes of packed 6-bit scales and mins (3 u32)
// Words 3-10:   32 bytes of high bits (8 u32)
// Words 11-42:  128 bytes of low 4-bit quantized weights (32 u32)
// Word 43:      4 bytes (d: f16 + dmin: f16)
// Total: 176 bytes = 44 u32 words

struct Q5KUniforms {
    M: u32,           // Number of rows in matrix A
    K: u32,           // Number of columns in matrix A (must be multiple of 256)
    num_blocks: u32,  // K / 256
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> matrix_a_q5k: array<u32>;  // Quantized matrix A (Q5_K blocks)
@group(0) @binding(1) var<storage, read> vector_b: array<f32>;      // Input vector B (K elements)
@group(0) @binding(2) var<storage, read_write> vector_c: array<f32>; // Output vector C (M elements)
@group(0) @binding(3) var<uniform> uniforms: Q5KUniforms;

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

// Extract 6-bit value from packed scales/mins
fn extract_6bit_from_packed(w0: u32, w1: u32, w2: u32, idx: u32) -> u32 {
    // 16 6-bit values packed into 96 bits (3 u32 words)
    let bit_offset = idx * 6u;
    let word_idx = bit_offset / 32u;
    let bit_in_word = bit_offset % 32u;
    
    if (word_idx == 0u) {
        if (bit_in_word <= 26u) {
            return (w0 >> bit_in_word) & 0x3Fu;
        } else {
            let bits_in_first = 32u - bit_in_word;
            let first = (w0 >> bit_in_word) & ((1u << bits_in_first) - 1u);
            let second = (w1 & ((1u << (6u - bits_in_first)) - 1u)) << bits_in_first;
            return first | second;
        }
    } else if (word_idx == 1u) {
        if (bit_in_word <= 26u) {
            return (w1 >> bit_in_word) & 0x3Fu;
        } else {
            let bits_in_first = 32u - bit_in_word;
            let first = (w1 >> bit_in_word) & ((1u << bits_in_first) - 1u);
            let second = (w2 & ((1u << (6u - bits_in_first)) - 1u)) << bits_in_first;
            return first | second;
        }
    } else {
        return (w2 >> bit_in_word) & 0x3Fu;
    }
}

// Extract high bit from qh array
fn get_high_bit(qh_base: u32, elem_idx: u32) -> u32 {
    // qh is 32 bytes starting at words 3-10 (8 u32 words)
    let byte_idx = elem_idx / 8u;
    let bit_idx = elem_idx % 8u;
    let word_idx = byte_idx / 4u;
    let byte_in_word = byte_idx % 4u;
    
    let word = matrix_a_q5k[qh_base + word_idx];
    let byte = (word >> (byte_in_word * 8u)) & 0xFFu;
    let bit = (byte >> bit_idx) & 1u;
    
    return bit;
}

// Dequantize and compute dot product for one Q5_K block (256 elements)
fn process_q5k_block(
    block_idx: u32,
    vec_offset: u32
) -> f32 {
    // Each block is 44 u32 words = 176 bytes
    let base = block_idx * 44u;
    
    // Load scales/mins (first 12 bytes = 3 u32 words at base+0)
    let scales_mins_0 = matrix_a_q5k[base + 0u];
    let scales_mins_1 = matrix_a_q5k[base + 1u];
    let scales_mins_2 = matrix_a_q5k[base + 2u];
    
    // Load super-scale (d) and super-min (dmin) - last word
    let metadata_word = matrix_a_q5k[base + 43u];
    let d = f16_to_f32(metadata_word & 0xFFFFu);
    let dmin = f16_to_f32((metadata_word >> 16u) & 0xFFFFu);
    
    // qh (high bits) starts at word 3
    let qh_base = base + 3u;
    
    var sum: f32 = 0.0;
    
    // Process 8 groups of 32 elements
    for (var group: u32 = 0u; group < 8u; group++) {
        // Extract scale and min for this group (6 bits each)
        let scale_bits = extract_6bit_from_packed(scales_mins_0, scales_mins_1, scales_mins_2, group);
        let min_bits = extract_6bit_from_packed(scales_mins_0, scales_mins_1, scales_mins_2, group + 8u);
        
        let scale = d * f32(scale_bits);
        let min_val = dmin * f32(min_bits);
        
        // Low 4-bit quants start at word 11 (after 3 words scales + 8 words qh)
        // Each group has 32 elements = 16 bytes = 4 u32 words
        let group_base = base + 11u + group * 4u;
        
        // Process 32 elements in this group
        for (var i: u32 = 0u; i < 4u; i++) {
            let word = matrix_a_q5k[group_base + i];
            
            // Each word has 8 nibbles (4-bit low values)
            for (var nibble: u32 = 0u; nibble < 8u; nibble++) {
                let low_4bit = (word >> (nibble * 4u)) & 0xFu;
                let elem_idx_in_group = i * 8u + nibble;
                let elem_idx_global = group * 32u + elem_idx_in_group;
                
                // Get high bit from qh array
                let high_bit = get_high_bit(qh_base, elem_idx_global);
                
                // Combine into 5-bit value
                let weight_5bit = low_4bit | (high_bit << 4u);
                
                // Dequantize
                let weight_f32 = scale * f32(weight_5bit) - min_val;
                
                let vec_val = vector_b[vec_offset + elem_idx_global];
                
                sum += weight_f32 * vec_val;
            }
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
    
    // Process all Q5_K blocks in this row
    for (var block: u32 = 0u; block < uniforms.num_blocks; block++) {
        let block_idx = row * uniforms.num_blocks + block;
        let vec_offset = block * 256u;
        
        sum += process_q5k_block(block_idx, vec_offset);
    }
    
    // Write result
    vector_c[row] = sum;
}
