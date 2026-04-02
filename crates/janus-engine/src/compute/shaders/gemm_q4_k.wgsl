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

    // Load super-scale (d) and super-min (dmin) - FIRST word (Word 0)
    let metadata_word = matrix_a_q4k[base + 0u];
    let unpacked_meta = manual_unpack(metadata_word);
    let d = unpacked_meta.x;
    let dmin = unpacked_meta.y;

    // Load scales/mins (12 bytes = 3 u32 words starting at Word 1)
    let scales_mins_0 = matrix_a_q4k[base + 1u];
    let scales_mins_1 = matrix_a_q4k[base + 2u];
    let scales_mins_2 = matrix_a_q4k[base + 3u];
    
    var sum: f32 = 0.0;
    
    // Process 8 groups of 32 elements
    for (var group: u32 = 0u; group < 8u; group++) {
        // Extract scale and min for this group (6 bits each)
        let scale_bits = extract_6bit_from_packed(scales_mins_0, scales_mins_1, scales_mins_2, group);
        let min_bits = extract_6bit_from_packed(scales_mins_0, scales_mins_1, scales_mins_2, group + 8u);
        
        let scale = d * f32(scale_bits);
        let min_val = dmin * f32(min_bits);
        
        // Quantized weights start at Word 4, 32 u32 words (128 bytes)
        // Each u32 contains 8 nibbles (4-bit values)
        let group_base = base + 4u + group * 4u;
        
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
    if (idx < 4u) {
        // sc[0..3]: lower 6 bits of bytes 0..3 (in w0)
        let b = (w0 >> (idx * 8u)) & 0xFFu;
        return b & 63u;
    } else if (idx < 8u) {
        // sc[4..7]: 4 bits from bytes 8..11 (w2) + 2 bits from bytes 0..3 (w0)
        let i = idx - 4u;
        let b_i8 = (w2 >> (i * 8u)) & 0xFFu;
        let b_i = (w0 >> (i * 8u)) & 0xFFu;
        let lower_4 = b_i8 & 15u;
        let upper_2 = b_i >> 6u;
        return lower_4 | (upper_2 << 4u);
    } else if (idx < 12u) {
        // m[0..3]: lower 6 bits of bytes 4..7 (in w1)
        let i = idx - 8u;
        let b = (w1 >> (i * 8u)) & 0xFFu;
        return b & 63u;
    } else {
        // m[4..7]: 4 bits from bytes 8..11 (w2) + 2 bits from bytes 4..7 (w1)
        let i = idx - 12u;
        let b_i8 = (w2 >> (i * 8u)) & 0xFFu;
        let b_i4 = (w1 >> (i * 8u)) & 0xFFu;
        let lower_4 = b_i8 >> 4u;
        let upper_2 = b_i4 >> 6u;
        return lower_4 | (upper_2 << 4u);
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
