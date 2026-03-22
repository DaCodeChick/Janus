# Shader Implementation Guide

This document provides detailed documentation for all WGSL (WebGPU Shading Language) shaders in the Janus Engine. Understanding these shaders is crucial for performance optimization, debugging, and extending the engine.

## Table of Contents

- [Overview](#overview)
- [Core Concepts](#core-concepts)
- [Shader Catalog](#shader-catalog)
  - [Matrix Operations](#matrix-operations)
  - [Attention Mechanisms](#attention-mechanisms)
  - [Activations & Normalization](#activations--normalization)
  - [Position Embeddings](#position-embeddings)
  - [Cache Management](#cache-management)
  - [Quantization](#quantization)
- [Performance Optimization](#performance-optimization)
- [Debugging Shaders](#debugging-shaders)

---

## Overview

Janus uses WebGPU compute shaders (WGSL) for all GPU operations. Each shader is optimized for:

- **Memory Bandwidth**: Minimize VRAM transfers via shared memory tiling
- **Compute Efficiency**: Maximize GPU utilization with proper workgroup sizing
- **Mixed Precision**: FP16 weights + FP32 compute for speed/accuracy balance
- **Batch Processing**: Process multiple sequences in parallel

### Shader Pipeline

```
┌─────────────────────────────────────────────────┐
│          Shader Compilation (at load)           │
├─────────────────────────────────────────────────┤
│  1. Parse WGSL source                           │
│  2. Validate syntax/semantics                   │
│  3. Compile to GPU-specific code               │
│  4. Cache compiled pipelines                    │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│       Runtime Execution (per token)             │
├─────────────────────────────────────────────────┤
│  1. Bind uniforms (parameters)                  │
│  2. Bind buffers (input/output)                 │
│  3. Dispatch workgroups                         │
│  4. Execute in parallel on GPU                  │
│  5. Synchronize if needed                       │
└─────────────────────────────────────────────────┘
```

---

## Core Concepts

### Workgroups and Threads

WGSL organizes work into a 3D grid:

```
@compute @workgroup_size(X, Y, Z)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Each thread executes this function
}
```

- **Workgroup Size**: Number of threads per workgroup (e.g., 16×16×1 = 256 threads)
- **Global ID**: Unique thread identifier across all workgroups
- **Local ID**: Thread position within its workgroup
- **Workgroup ID**: Which workgroup this thread belongs to

**Example**: For a 1024×1024 matrix with 16×16 workgroups:
- Total workgroups: (1024/16) × (1024/16) = 64 × 64 = 4096
- Total threads: 1024 × 1024 = 1,048,576
- Each workgroup has 256 threads

### Memory Hierarchy

```
┌─────────────────────────────────────────┐
│  Register Memory (fastest)              │  Per-thread variables
│  - Thread-local variables               │
│  - Extremely fast access                │
└─────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│  Shared/Workgroup Memory                │  Per-workgroup variables
│  - var<workgroup> declarations          │
│  - Shared within workgroup              │
│  - Requires barriers for sync           │
└─────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│  Global Memory (VRAM)                   │  All threads
│  - var<storage> declarations            │
│  - Highest capacity, slowest access     │
│  - Primary data storage                 │
└─────────────────────────────────────────┘
```

### Synchronization

```wgsl
workgroupBarrier();  // Wait for all threads in workgroup
storageBarrier();    // Ensure memory writes are visible
```

Use barriers when:
- Loading data into shared memory
- After computing with shared memory
- Before loading next tile of data

---

## Shader Catalog

### Matrix Operations

#### 1. `gemm.wgsl` - General Matrix Multiplication

**Purpose**: Compute `C = A × B` for transformer linear layers

**Algorithm**: Tiled matrix multiplication with shared memory

```
Input:
  A: [batch, M, K]  - Activations (FP32)
  B: [K, N]         - Weights (packed FP16)

Output:
  C: [batch, M, N]  - Result (FP32)

Workgroup Size: 16×16×1 (256 threads)
Tile Size: 16×16

Performance:
  - Uses shared memory for A and B tiles
  - Each workgroup computes 16×16 output elements
  - Memory bandwidth: ~80% of theoretical max
```

**Code Structure**:

```wgsl
@compute @workgroup_size(16, 16, 1)
fn main(...) {
    // 1. Collaborative load: Each thread loads 1 element
    for each tile in K dimension:
        Load tile_a from global to shared memory
        Load tile_b from global to shared memory
        workgroupBarrier()
        
        // 2. Compute: Each thread accumulates partial dot product
        for k in tile:
            sum += tile_a[row][k] * tile_b[k][col]
        
        workgroupBarrier()
    
    // 3. Write output
    matrix_c[batch][row][col] = sum
}
```

**Optimizations**:
- Weights packed as FP16 (2 values per u32) for 50% memory savings
- Transpose B matrix on read to match PyTorch weight layout
- Batch dimension parallelized via Z workgroup coordinate

---

#### 2. `gemm_q4_k.wgsl` - Quantized Matrix Multiplication (Q4_K)

**Purpose**: Compute `C = A × B` with 4-bit quantized weights

**Quantization Format**: K-quants (GGUF format)
- 4 bits per weight (16:1 compression vs FP32)
- Block-wise quantization (32 weights per block)
- Separate scale and min per block

```
Block Layout (Q4_K):
  [scale: f16][min: f16][weights: 16×u8]
  Each u8 contains 2 4-bit weights
  
Dequantization:
  weight_fp32 = scale * weight_int4 + min
```

**Performance**:
- Memory bandwidth: 4× better than FP16
- Compute: Slightly slower due to dequantization
- Overall: 2-3× faster inference for memory-bound ops

---

### Attention Mechanisms

#### 3. `attention.wgsl` - Scaled Dot-Product Attention

**Purpose**: Implement `Attention(Q, K, V) = softmax(QK^T / √d) V`

**Three-Stage Pipeline**:

```
Stage 1: compute_qk_scores
├─ Compute Q @ K^T for all key positions
├─ Scale by 1/√head_dim
└─ Output: scores [batch, heads, seq_len]

Stage 2: softmax (separate shader)
├─ Normalize scores to probabilities
└─ Output: probs [batch, heads, seq_len]

Stage 3: compute_attn_output
├─ Compute probs @ V
└─ Output: attention output [batch, heads, head_dim]
```

**Grouped Query Attention (GQA)**:

```
num_heads = 32      (Query heads)
num_kv_heads = 4    (Key/Value heads)
ratio = 32 / 4 = 8  (Each KV head shared by 8 Q heads)

Mapping:
  Q_head_0..7   → KV_head_0
  Q_head_8..15  → KV_head_1
  Q_head_16..23 → KV_head_2
  Q_head_24..31 → KV_head_3
```

**Memory Layout**:

```
Query:  [batch][num_heads][head_dim]
Keys:   [batch][layers][seq_len][num_kv_heads][head_dim]
Values: [batch][layers][seq_len][num_kv_heads][head_dim]
Output: [batch][num_heads][head_dim]
```

**Workgroup Strategy**:
- Each workgroup handles one (batch_item, head) pair
- Parallelizes across keys/sequence length
- Workgroup size: 256 threads

**Code Flow (Stage 1 - QK^T)**:

```wgsl
fn compute_qk_scores(...) {
    // Map head to KV head (for GQA)
    kv_head = query_head / (num_heads / num_kv_heads)
    
    // Each thread processes one key position
    for pos in seq_len:
        // Dot product: Q · K[pos]
        dot_product = 0.0
        for d in head_dim:
            dot_product += query[d] * key[pos][d]
        
        // Scale
        scores[pos] = dot_product / sqrt(head_dim)
}
```

---

#### 4. `softmax.wgsl` - Softmax Normalization

**Purpose**: Convert attention scores to probabilities

**Algorithm**: Numerically stable softmax

```
Given scores x[i] for i in 0..N:

1. Find maximum: max_val = max(x)
2. Compute exp: exp_vals[i] = exp(x[i] - max_val)
3. Compute sum: sum = Σ exp_vals[i]
4. Normalize: probs[i] = exp_vals[i] / sum
```

**Why subtract max?**
- Prevents overflow: exp(large_number) → inf
- Mathematically equivalent: exp(x - max) / Σ exp(x - max) = exp(x) / Σ exp(x)

**Implementation**:

```wgsl
@compute @workgroup_size(256)
fn softmax(...) {
    // Phase 1: Find max (parallel reduction)
    var<workgroup> shared_max: array<f32, 256>;
    
    // Each thread finds local max
    local_max = scores[thread_id]
    shared_max[local_id] = local_max
    
    // Reduce to find global max
    for stride in [128, 64, 32, 16, 8, 4, 2, 1]:
        workgroupBarrier()
        if local_id < stride:
            shared_max[local_id] = max(
                shared_max[local_id],
                shared_max[local_id + stride]
            )
    
    global_max = shared_max[0]
    workgroupBarrier()
    
    // Phase 2: Compute exp and sum
    exp_val = exp(scores[thread_id] - global_max)
    
    // Parallel reduction for sum (similar to max)
    ...
    
    // Phase 3: Normalize
    probs[thread_id] = exp_val / sum
}
```

---

### Activations & Normalization

#### 5. `activations.wgsl` - Neural Network Activations

**SiLU (Swish) Activation**:

```
SiLU(x) = x * sigmoid(x)
        = x / (1 + exp(-x))

Properties:
  - Smooth, non-monotonic
  - Better gradient flow than ReLU
  - Used in modern LLMs (LLaMA, Mistral)
```

**Implementation**:

```wgsl
@compute @workgroup_size(256)
fn silu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= size {
        return;
    }
    
    let x = input[idx];
    
    // SiLU(x) = x * σ(x) where σ(x) = 1 / (1 + e^(-x))
    let sigmoid = 1.0 / (1.0 + exp(-x));
    output[idx] = x * sigmoid;
}
```

**RMSNorm (Root Mean Square Normalization)**:

```
RMSNorm(x) = x / RMS(x) * γ

where RMS(x) = √(Σ x_i² / n)

Properties:
  - Simpler than LayerNorm (no mean centering)
  - Faster to compute
  - Used in LLaMA architecture
```

**Implementation**:

```wgsl
@compute @workgroup_size(256)
fn rmsnorm(...) {
    // Phase 1: Compute sum of squares
    var<workgroup> shared_sum: array<f32, 256>;
    
    sum_squares = 0.0
    for d in hidden_dim:
        sum_squares += input[d] * input[d]
    
    // Parallel reduction to get total
    ...
    
    // Phase 2: Compute RMS
    rms = sqrt(sum_squares / hidden_dim + eps)
    
    // Phase 3: Normalize and scale
    for d in hidden_dim:
        output[d] = (input[d] / rms) * gamma[d]
}
```

---

### Position Embeddings

#### 6. `rope.wgsl` - Rotary Position Embeddings

**Purpose**: Encode position information via rotation in embedding space

**LLaMA Half-Split Variant**:

```
For dimension pair (i, i + head_dim/2):
  
  angle = position / (10000 ^ (2i / head_dim))
  
  x'[i]              = x[i] * cos(angle) - x[i + d/2] * sin(angle)
  x'[i + head_dim/2] = x[i] * sin(angle) + x[i + d/2] * cos(angle)
```

**Visualization**:

```
Input: [x0, x1, x2, x3, x4, x5, x6, x7]  (head_dim = 8)
       └────┬────┘ └────┬────┘
         pair 0      pair 1
         
Rotate pair 0: (x0, x4) → (x0', x4')
Rotate pair 1: (x1, x5) → (x1', x5')
Rotate pair 2: (x2, x6) → (x2', x6')
Rotate pair 3: (x3, x7) → (x3', x7')

Output: [x0', x1', x2', x3', x4', x5', x6', x7']
```

**Optimization - Precomputed Cache**:

Instead of computing `sin(angle)` and `cos(angle)` for every forward pass:

```
At initialization:
  For position p in 0..max_seq_len:
    For dim i in 0..(head_dim/2):
      angle = p / (10000 ^ (2i / head_dim))
      rope_cache[p][i] = (cos(angle), sin(angle))

During inference:
  For position p:
    (cos_val, sin_val) = rope_cache[p][dim]
    x' = x * cos_val - x_paired * sin_val
    x_paired' = x * sin_val + x_paired * cos_val
```

**Performance**:
- Without cache: ~1000 exp() and trig ops per token
- With cache: 2 memory reads per dimension pair
- Speedup: ~10-20× faster

**Implementation**:

```wgsl
@compute @workgroup_size(256)
fn main(...) {
    // Calculate which dimension pair this thread handles
    let pair_idx = global_id.x;
    let batch_idx = pair_idx / (num_heads * half_dim);
    let head_idx = (pair_idx % (num_heads * half_dim)) / half_dim;
    let dim_pair = pair_idx % half_dim;
    
    // Read from cache
    let cache_idx = position * half_dim + dim_pair;
    let cos_val = rope_cache[cache_idx * 2];
    let sin_val = rope_cache[cache_idx * 2 + 1];
    
    // Read input pair
    let idx_first = batch_idx * (num_heads * head_dim) 
                  + head_idx * head_dim 
                  + dim_pair;
    let idx_second = idx_first + half_dim;
    
    let x_first = input[idx_first];
    let x_second = input[idx_second];
    
    // Apply rotation
    output[idx_first]  = x_first * cos_val - x_second * sin_val;
    output[idx_second] = x_first * sin_val + x_second * cos_val;
}
```

---

### Cache Management

#### 7. `update_cache.wgsl` - KV Cache Update

**Purpose**: Store new K/V projections in cache for future tokens

**Cache Layout**:

```
KV Cache: [batch][layers][max_seq_len][num_kv_heads][head_dim]

For each new token:
  - Store K projection at cache[batch][layer][position]
  - Store V projection at cache[batch][layer][position]
  - Increment position
```

**Write Operation**:

```wgsl
@compute @workgroup_size(256)
fn main(...) {
    // Calculate which cache element to update
    let elem_idx = global_id.x;
    
    // Decompose into batch, head, dimension
    let batch_idx = elem_idx / (num_kv_heads * head_dim);
    let head_idx = (elem_idx % (num_kv_heads * head_dim)) / head_dim;
    let dim = elem_idx % head_dim;
    
    // Read new K/V value
    let new_value = new_kv[elem_idx];
    
    // Calculate cache position
    let cache_idx = batch_idx * (num_layers * max_seq_len * num_kv_heads * head_dim)
                  + layer_idx * (max_seq_len * num_kv_heads * head_dim)
                  + position * (num_kv_heads * head_dim)
                  + head_idx * head_dim
                  + dim;
    
    // Write to cache
    kv_cache[cache_idx] = new_value;
}
```

---

#### 8. `compress_cache.wgsl` - KV Cache Compression

**Purpose**: Extend effective context length by compressing old cache entries

**Compression Strategy**:

```
Original cache (seq_len = 1000):
  [Token 0] [Token 1] ... [Token 999]

After compression (2:1 ratio):
  [Avg(0,1)] [Avg(2,3)] ... [Avg(598,599)] [Token 600] ... [Token 999]
  └────── Compressed (300) ──────┘        └─ Uncompressed (400) ─┘
```

**Algorithm**:

```wgsl
@compute @workgroup_size(256)
fn compress(...) {
    let compressed_pos = global_id.x;
    
    // This compressed position represents averaging N original positions
    let start_pos = compressed_pos * compression_ratio;
    let end_pos = start_pos + compression_ratio;
    
    // Average over compression window
    var sum = 0.0;
    for pos in start_pos..end_pos:
        sum += cache[pos];
    
    let avg = sum / f32(compression_ratio);
    
    // Write to output (will be copied back to main cache)
    compressed_cache[compressed_pos] = avg;
}
```

**Memory Management**:

```
1. Create temporary compressed buffer
2. Compress old entries → temp buffer
3. Copy compressed entries to cache start
4. Copy recent uncompressed entries after compressed
5. Free temporary buffer

Layout after compression:
  [Compressed Old][Uncompressed Recent][Free Space]
```

**Performance**:
- 2:1 compression: 2× effective context length, ~5% quality loss
- 4:1 compression: 4× effective context length, ~10-15% quality loss

---

### Quantization

#### 9. `gemm_q4_k.wgsl`, `gemm_q5_k.wgsl`, `gemm_q8_0.wgsl`

**Purpose**: Multiply with quantized weights for faster inference

**Quantization Levels**:

```
Q8_0: 8-bit quantization
  - 1 byte per weight
  - Scale per 32-weight block
  - 4:1 compression vs FP32

Q5_K: 5-bit K-quants
  - 5 bits per weight  
  - Per-block scale + min
  - 6.4:1 compression vs FP32

Q4_K: 4-bit K-quants
  - 4 bits per weight
  - Per-block scale + min
  - 8:1 compression vs FP32
  - Sweet spot: best size/quality trade-off
```

**Dequantization**:

```wgsl
// Q4_K block structure (32 weights)
struct Q4KBlock {
    scale: f16,    // 2 bytes
    min: f16,      // 2 bytes
    weights: [16]u8  // 16 bytes (2 weights per byte)
}

fn dequantize_q4k(block: Q4KBlock, idx: u32) -> f32 {
    // Extract 4-bit value
    let byte = block.weights[idx / 2];
    let is_high = (idx % 2) == 1;
    let nibble = select(byte & 0x0F, (byte >> 4) & 0x0F, is_high);
    
    // Convert to FP32
    let scale = f32(block.scale);
    let min_val = f32(block.min);
    return scale * f32(nibble) + min_val;
}
```

**GEMM with Quantized Weights**:

```wgsl
@compute @workgroup_size(16, 16, 1)
fn gemm_q4k(...) {
    var sum = 0.0;
    
    for k in 0..K:
        let a_val = matrix_a[k];  // FP32 activation
        
        // Dequantize weight on-the-fly
        let block_idx = k / 32;
        let in_block_idx = k % 32;
        let b_val = dequantize_q4k(matrix_b_blocks[block_idx], in_block_idx);
        
        sum += a_val * b_val;  // FP32 computation
    
    matrix_c[row * N + col] = sum;
}
```

**Performance Analysis**:

| Format | Memory | Bandwidth | Compute | Overall Speedup |
|--------|--------|-----------|---------|-----------------|
| FP32 | 100% | 1.0× | 1.0× | 1.0× (baseline) |
| FP16 | 50% | 2.0× | 1.0× | 1.8× |
| Q8_0 | 25% | 4.0× | 0.9× | 3.2× |
| Q5_K | 15.6% | 6.4× | 0.85× | 4.5× |
| Q4_K | 12.5% | 8.0× | 0.8× | 5.5× |

Note: Memory-bound operations (GEMM) benefit most from quantization.

---

## Performance Optimization

### 1. Memory Access Patterns

**Coalesced Access** (Fast):

```wgsl
// Good: Sequential threads access sequential memory
for thread_id in 0..256:
    data[thread_id]  // Adjacent threads → adjacent memory
```

**Strided Access** (Slow):

```wgsl
// Bad: Threads access memory with large strides
for thread_id in 0..256:
    data[thread_id * 1024]  // Cache misses, poor bandwidth
```

### 2. Shared Memory Tiling

**Without Tiling** (Poor):

```wgsl
// Each thread loads from global memory multiple times
for k in 0..K:
    sum += A[row * K + k] * B[k * N + col]  // K global reads per thread
```

**With Tiling** (Optimized):

```wgsl
// Load tile into shared memory once
for tile in 0..(K/TILE_SIZE):
    Load A_tile from global to workgroup memory  // Collaborative load
    Load B_tile from global to workgroup memory
    workgroupBarrier()
    
    // Compute using shared memory (fast)
    for k in 0..TILE_SIZE:
        sum += A_tile[k] * B_tile[k]
    
    workgroupBarrier()
```

**Bandwidth Comparison**:
- Without tiling: K × (M×N) global reads = K×M×N
- With tiling: K/T × (M×T + T×N) global reads ≈ K×(M+N)
- Speedup: (M×N) / (M+N) ≈ 10-100× for typical sizes

### 3. Occupancy Optimization

**Workgroup Size Selection**:

```
GPU: 64 compute units, 1024 max threads per workgroup

Option 1: workgroup_size(1024)
  - 64 workgroups active (64 × 1024 = 65,536 threads)
  - Requires more shared memory per workgroup
  - May limit occupancy

Option 2: workgroup_size(256)
  - 256 workgroups active (256 × 256 = 65,536 threads)
  - Less shared memory per workgroup
  - Better occupancy

Recommendation: 256-512 threads per workgroup
```

### 4. Reduce Synchronization

**Excessive Barriers** (Slow):

```wgsl
for i in 0..100:
    shared[local_id] = compute_something()
    workgroupBarrier()  // 100 barriers!
```

**Optimized** (Fast):

```wgsl
// Accumulate in registers
var sum = 0.0
for i in 0..100:
    sum += compute_something()

// Single barrier at the end
shared[local_id] = sum
workgroupBarrier()
```

---

## Debugging Shaders

### 1. Validation Errors

**Common Issues**:

```wgsl
// Error: Binding index already used
@group(0) @binding(0) var<storage> a: array<f32>;
@group(0) @binding(0) var<storage> b: array<f32>;  // ❌ Duplicate

// Fix: Use unique binding indices
@group(0) @binding(0) var<storage> a: array<f32>;
@group(0) @binding(1) var<storage> b: array<f32>;  // ✅
```

```wgsl
// Error: Workgroup size exceeds limits
@compute @workgroup_size(2048)  // ❌ Too large

// Fix: Use smaller workgroups
@compute @workgroup_size(256)   // ✅
```

### 2. Out-of-Bounds Access

**Debug Pattern**:

```wgsl
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Always bounds check!
    if idx >= array_size {
        return;
    }
    
    // Safe to access
    output[idx] = input[idx];
}
```

### 3. Incorrect Results

**Debug Techniques**:

```wgsl
// 1. Write intermediate results to separate buffer
debug_buffer[idx] = intermediate_value;

// 2. Use simple test cases
// Input: [1, 2, 3, 4]
// Expected: [2, 4, 6, 8]  (multiply by 2)

// 3. Compare with CPU implementation
// Compute same operation on CPU and compare results
```

### 4. Performance Profiling

Use GPU profiler tools:

- **NVIDIA Nsight**: CUDA-based GPUs
- **RenderDoc**: Vulkan/DirectX
- **Metal Debugger**: macOS

Key metrics:
- Memory bandwidth utilization
- Compute occupancy
- Workgroup efficiency
- Cache hit rate

---

## Best Practices

### 1. Memory Layout

```wgsl
// ✅ Good: Structure of Arrays (SoA)
var<storage> positions_x: array<f32>;
var<storage> positions_y: array<f32>;
var<storage> positions_z: array<f32>;

// ❌ Bad: Array of Structures (AoS) - poor access patterns
struct Position { x: f32, y: f32, z: f32 }
var<storage> positions: array<Position>;
```

### 2. Uniform Buffer Size

```wgsl
// Uniforms should be small and frequently used
struct Uniforms {
    batch_size: u32,
    seq_len: u32,
    hidden_dim: u32,
    // ... other small parameters
}  // Aim for < 256 bytes

// Large data → storage buffers
var<storage> weights: array<f32>;  // Can be GBs
```

### 3. Avoid Divergent Branching

```wgsl
// ❌ Bad: Different threads take different paths
if thread_id % 2 == 0:
    expensive_operation_a()
else:
    expensive_operation_b()
// GPU executes BOTH paths for all threads!

// ✅ Good: All threads execute same code
result = select(operation_b(), operation_a(), thread_id % 2 == 0)
```

---

## Further Reading

- [WebGPU Specification](https://www.w3.org/TR/webgpu/)
- [WGSL Language Spec](https://www.w3.org/TR/WGSL/)
- [GPU Optimization Guide](https://developer.nvidia.com/blog/cuda-optimization/)
- Janus source: `crates/janus-engine/src/compute/shaders/`

---

*This shader guide is maintained alongside the implementation. For latest changes, see the shader source files.*
