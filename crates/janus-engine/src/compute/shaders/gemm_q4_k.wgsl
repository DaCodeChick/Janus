// Q4_K Quantized Matrix-Vector Multiplication
// Matrix A: Quantized in Q4_K format (M x K)
// Vector B: F32 format (K elements)
// Vector C: F32 output (M elements)
//
// Q4_K Block Structure (from llama.cpp ggml-quants.h):
// Superblock of 256 elements (QK_K) divided into 8 blocks of 32 elements
// 
// struct block_q4_K {
//     uint8_t scales[12];      // Scales and mins packed as 6-bit values
//     uint8_t qs[128];         // 4-bit quants (256 values packed into 128 bytes)
//     half d;                  // Super-scale delta (f16)
//     half dmin;               // Super-min (f16)
// };
// Total: 144 bytes per block
//
// Memory layout (as u32 words since WGSL lacks u8):
// Words 0-2:    12 bytes of packed 6-bit scales and mins (3 u32)
// Words 3-34:   128 bytes of 4-bit quantized weights (256 weights, 32 u32)
// Words 35:     4 bytes (d: f16 + dmin: f16)
// Total: 144 bytes = 36 u32 words

struct Q4KUniforms {
    M: u32,           // Number of rows in matrix A
    K: u32,           // Number of columns in matrix A (must be multiple of 256)
    num_blocks: u32,  // K / 256
    _pad: u32,
}

fn f16_to_f32_safe(h: u32) -> f32 {
    let s = (h & 0x8000u) << 16u;
    let e = (h & 0x7C00u) >> 10u;
    let m = h & 0x03FFu;

    if (e == 0u) {
        return bitcast<f32>(s);
    }
    if (e == 31u) {
        return bitcast<f32>(s | 0x7F800000u | (m << 13u));
    }
    let exp = e + 112u;
    return bitcast<f32>(s | (exp << 23u) | (m << 13u));
}

fn manual_unpack(val: u32) -> vec2<f32> {
    return vec2<f32>(f16_to_f32_safe(val & 0xFFFFu), f16_to_f32_safe(val >> 16u));
}

@group(0) @binding(0) var<storage, read> matrix_a_q4k: array<u32>;  // Quantized matrix A (Q4_K blocks)
@group(0) @binding(1) var<storage, read> vector_b: array<f32>;      // Input vector B (K elements)
@group(0) @binding(2) var<storage, read_write> vector_c: array<f32>; // Output vector C (M elements)
@group(0) @binding(3) var<uniform> uniforms: Q4KUniforms;

// Dequantize and compute dot product for one Q4_K block (256 elements)
fn process_q4k_block(
    block_idx: u32,
    vec_offset: u32
) -> f32 {
    // Each block is 36 u32 words = 144 bytes
    let base = block_idx * 36u;
    
    // Load scales/mins (first 12 bytes = 3 u32 words at base+0)
    // These contain 8 scales and 8 mins, each 6 bits
    let scales_mins_0 = matrix_a_q4k[base + 0u];
    let scales_mins_1 = matrix_a_q4k[base + 1u];
    let scales_mins_2 = matrix_a_q4k[base + 2u];
    
    // Load super-scale (d) and super-min (dmin) - last word
    let metadata_word = matrix_a_q4k[base + 35u];
    let unpacked_meta = manual_unpack(metadata_word);
    let d = unpacked_meta.x;
    let dmin = unpacked_meta.y;
    
    var sum: f32 = 0.0;
    
    // Process 8 groups of 32 elements
    for (var group: u32 = 0u; group < 8u; group++) {
        // Extract scale and min for this group (6 bits each)
        let scale_bits = extract_6bit_from_packed(scales_mins_0, scales_mins_1, scales_mins_2, group);
        let min_bits = extract_6bit_from_packed(scales_mins_0, scales_mins_1, scales_mins_2, group + 8u);
        
        let scale = d * f32(scale_bits);
        let min_val = dmin * f32(min_bits);
        
        // Quantized weights start at word 3, 32 u32 words (128 bytes)
        // Each u32 contains 8 nibbles (4-bit values)
        let group_base = base + 3u + group * 4u;
        
        // Process 32 elements in this group (4 u32 words)
        for (var i: u32 = 0u; i < 4u; i++) {
            let word = matrix_a_q4k[group_base + i];
            
            // Each word has 8 nibbles (32 bits / 4 bits = 8)
            for (var nibble: u32 = 0u; nibble < 8u; nibble++) {
                let weight_4bit = (word >> (nibble * 4u)) & 0xFu;
                let weight_f32 = scale * f32(weight_4bit) - min_val;
                
                let elem_idx = group * 32u + i * 8u + nibble;
                let vec_val = vector_b[vec_offset + elem_idx];
                
                sum += weight_f32 * vec_val;
            }
        }
    }
    
    return sum;
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

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    
    // Bounds check
    if (row >= uniforms.M) {
        return;
    }
    
    var sum: f32 = 0.0;
    
    // Process all Q4_K blocks in this row
    for (var block: u32 = 0u; block < uniforms.num_blocks; block++) {
        let block_idx = row * uniforms.num_blocks + block;
        let vec_offset = block * 256u;
        
        sum += process_q4k_block(block_idx, vec_offset);
    }
    
    // Write result
    vector_c[row] = sum;
}
