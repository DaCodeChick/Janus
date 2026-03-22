# Janus Engine Architecture

This document provides a comprehensive overview of the Janus Engine architecture, including component diagrams, data flow, and design decisions.

## Table of Contents

- [High-Level Overview](#high-level-overview)
- [Component Architecture](#component-architecture)
- [Compute Engine](#compute-engine)
- [Model Architecture](#model-architecture)
- [Memory Management](#memory-management)
- [Inference Pipeline](#inference-pipeline)
- [Plugin System](#plugin-system)
- [Router Architecture](#router-architecture)

---

## High-Level Overview

Janus is a modular, GPU-accelerated LLM inference engine with a plugin-based architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                       Janus Engine Ecosystem                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │ janus-server │      │ janus-router │      │   Plugins    │  │
│  │              │◄─────┤              │◄─────┤              │  │
│  │  HTTP API    │      │   Routing    │      │  Dynamic     │  │
│  │   (SSE)      │      │   Logic      │      │  Loading     │  │
│  └──────┬───────┘      └──────┬───────┘      └──────────────┘  │
│         │                     │                                  │
│         └─────────────────────┘                                  │
│                    │                                             │
│         ┌──────────▼──────────┐                                 │
│         │   janus-engine      │                                 │
│         │                     │                                 │
│         │  ┌──────────────┐  │                                 │
│         │  │    Model     │  │  Core inference engine          │
│         │  │  Transformer │  │  - GPU compute ops              │
│         │  │   Sampler    │  │  - Memory management            │
│         │  └──────┬───────┘  │  - Static compute graph         │
│         │         │           │                                 │
│         │  ┌──────▼───────┐  │                                 │
│         │  │ComputeEngine │  │                                 │
│         │  │              │  │                                 │
│         │  │  WGPU/WebGPU │  │                                 │
│         │  └──────┬───────┘  │                                 │
│         └─────────┼──────────┘                                 │
│                   │                                             │
│         ┌─────────▼──────────┐                                 │
│         │   GPU Hardware     │                                 │
│         │  Vulkan/Metal/DX12 │                                 │
│         └────────────────────┘                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### Directory Structure

```
janus/
├── janus-api/              # ABI-stable plugin interface
│   ├── types.rs            # FFI-safe types
│   ├── callback.rs         # Streaming callbacks
│   └── plugin.rs           # Plugin trait
│
├── janus-engine/           # Core inference engine
│   ├── compute/            # GPU compute operations
│   │   ├── engine.rs       # ComputeEngine (device/queue)
│   │   ├── cache.rs        # KV cache with compression
│   │   ├── pipeline_cache.rs # Pre-compiled shaders
│   │   ├── ops/            # Tensor operations
│   │   │   ├── matmul.rs   # GEMM operations
│   │   │   ├── attention.rs # Attention mechanisms
│   │   │   ├── activation.rs # SiLU, RMSNorm
│   │   │   └── rope.rs     # Rotary Position Embedding
│   │   └── shaders/        # WGSL compute shaders
│   │
│   ├── model/              # Transformer implementation
│   │   ├── model.rs        # Full model with static graph
│   │   ├── block.rs        # TransformerBlock
│   │   ├── output.rs       # LM head
│   │   ├── sampler.rs      # Token sampling
│   │   └── speculative.rs  # Speculative decoding
│   │
│   ├── loaders/            # Model loading
│   │   ├── gguf.rs         # GGUF format
│   │   └── safetensors.rs  # Safetensors format
│   │
│   └── examples/           # Usage examples
│
├── janus-router/           # Intelligent routing
│   └── lib.rs              # Local vs Cloud routing
│
├── janus-server/           # HTTP API server
│   └── main.rs             # OpenAI-compatible API
│
└── plugins/                # Plugin implementations
    ├── janus-instruct-plugin/
    └── janus-roleplay-plugin/
```

---

## Compute Engine

The `ComputeEngine` is the foundation of GPU operations:

```
┌───────────────────────────────────────────────────────────┐
│                     ComputeEngine                          │
├───────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐         ┌─────────────┐                  │
│  │   Device    │         │    Queue    │                  │
│  │             │         │             │                  │
│  │  - GPU init │         │ - Submit    │                  │
│  │  - Limits   │         │ - Poll      │                  │
│  │  - Adapter  │         │ - Sync      │                  │
│  └─────────────┘         └─────────────┘                  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              Buffer Management                       │  │
│  │                                                       │  │
│  │  • Zero-copy allocation                             │  │
│  │  • Usage flags: STORAGE | COPY_SRC | COPY_DST      │  │
│  │  • Memory alignment                                  │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │           Pipeline Management                        │  │
│  │                                                       │  │
│  │  • Bind group layouts                                │  │
│  │  • Compute pipelines                                 │  │
│  │  • Shader modules                                    │  │
│  └─────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Static Computation Graph**: All buffers pre-allocated at model load time
2. **Single Submission Per Token**: Entire forward pass encoded in one command buffer
3. **Pipeline Caching**: Shaders compiled once and reused
4. **Zero Dynamic Allocations**: No memory allocation during inference

---

## Model Architecture

### Transformer Model Flow

```
Input Token IDs
      │
      ▼
┌─────────────────┐
│   Embedding     │  token_id → hidden_state [hidden_dim]
└────────┬────────┘
         │
         ▼
┌────────────────────────────────────────────────┐
│           Transformer Layers (x N)              │
│                                                 │
│  ┌──────────────────────────────────────────┐  │
│  │       Layer i (TransformerBlock)         │  │
│  │                                           │  │
│  │  ┌─────────────────────────────────────┐ │  │
│  │  │      Self-Attention Block           │ │  │
│  │  │                                      │ │  │
│  │  │  Input: x                            │ │  │
│  │  │    │                                 │ │  │
│  │  │    ▼                                 │ │  │
│  │  │  RMSNorm (pre-norm)                 │ │  │
│  │  │    │                                 │ │  │
│  │  │    ├──────┬──────┬──────┐           │ │  │
│  │  │    ▼      ▼      ▼      ▼           │ │  │
│  │  │    Q      K      V    (projections) │ │  │
│  │  │    │      │      │                   │ │  │
│  │  │    │      └──────┴─────► KV Cache   │ │  │
│  │  │    │            (reuse)  (store)    │ │  │
│  │  │    │             │                   │ │  │
│  │  │    └─────────────┘                   │ │  │
│  │  │           │                           │ │  │
│  │  │           ▼                           │ │  │
│  │  │    RoPE (Rotary Embedding)           │ │  │
│  │  │           │                           │ │  │
│  │  │           ▼                           │ │  │
│  │  │    Attention(Q, K, V)                │ │  │
│  │  │      scores = Q @ K^T / √d           │ │  │
│  │  │      probs = softmax(scores)         │ │  │
│  │  │      out = probs @ V                 │ │  │
│  │  │           │                           │ │  │
│  │  │           ▼                           │ │  │
│  │  │    Output projection (O)             │ │  │
│  │  │           │                           │ │  │
│  │  │           ▼                           │ │  │
│  │  │    Residual: x + out                 │ │  │
│  │  └──────────┬────────────────────────── │ │  │
│  │             │                             │  │
│  │             ▼                             │  │
│  │  ┌─────────────────────────────────────┐ │  │
│  │  │      Feed-Forward Network (FFN)     │ │  │
│  │  │                                      │ │  │
│  │  │  Input: x                            │ │  │
│  │  │    │                                 │ │  │
│  │  │    ▼                                 │ │  │
│  │  │  RMSNorm (pre-norm)                 │ │  │
│  │  │    │                                 │ │  │
│  │  │    ├────────┬─────────┐             │ │  │
│  │  │    ▼        ▼         ▼             │ │  │
│  │  │  Gate      Up      (projections)    │ │  │
│  │  │    │        │                        │ │  │
│  │  │    ▼        │                        │ │  │
│  │  │  SiLU       │                        │ │  │
│  │  │    │        │                        │ │  │
│  │  │    └────┬───┘                        │ │  │
│  │  │         ▼                            │ │  │
│  │  │    Element-wise multiply            │ │  │
│  │  │         │                            │ │  │
│  │  │         ▼                            │ │  │
│  │  │    Down projection                   │ │  │
│  │  │         │                            │ │  │
│  │  │         ▼                            │ │  │
│  │  │    Residual: x + out                │ │  │
│  │  └──────────┬────────────────────────── │ │  │
│  └─────────────┼──────────────────────────┘  │
│                │                              │
│                ▼                              │
│           Next Layer                          │
└───────────────┬────────────────────────────────┘
                │
                ▼
┌───────────────────────────┐
│      Output Head          │
│                           │
│  RMSNorm (final)          │
│         │                 │
│         ▼                 │
│  LM Head projection       │
│  [hidden_dim → vocab_size]│
│         │                 │
│         ▼                 │
│    Logits [vocab_size]    │
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│       Sampler             │
│                           │
│  • Temperature scaling    │
│  • Top-p (nucleus)        │
│  • Top-k filtering        │
│  • Repetition penalty     │
│         │                 │
│         ▼                 │
│   Next token ID           │
└───────────────────────────┘
```

### Buffer Ping-Pong Pattern

To avoid buffer conflicts, Janus uses a ping-pong pattern:

```
Layer 0:  Input → Buffer_A → Output
          ────────────────────
Layer 1:  Buffer_A → Buffer_B → Output
          ────────────────────
Layer 2:  Buffer_B → Buffer_A → Output
          ────────────────────
...
```

This eliminates the need for intermediate buffer copies.

---

## Memory Management

### Pre-allocated Buffers

```
┌─────────────────────────────────────────────────┐
│              Model Struct                        │
├─────────────────────────────────────────────────┤
│                                                  │
│  Weight Buffers (read-only):                    │
│  ┌────────────────────────────────────────┐    │
│  │ • embedding_weights                     │    │
│  │ • layer_weights[0..N]                   │    │
│  │ • output_norm_weights                   │    │
│  │ • lm_head_weights                       │    │
│  └────────────────────────────────────────┘    │
│                                                  │
│  Scratch Buffers (read-write):                  │
│  ┌────────────────────────────────────────┐    │
│  │ • hidden_a  [batch × seq × hidden]     │    │
│  │ • hidden_b  [batch × seq × hidden]     │    │
│  │ • q_proj    [batch × seq × heads × d]  │    │
│  │ • k_proj    [batch × seq × heads × d]  │    │
│  │ • v_proj    [batch × seq × heads × d]  │    │
│  │ • attn_out  [batch × seq × hidden]     │    │
│  │ • scores    [batch × heads × seq × seq]│    │
│  │ • probs     [batch × heads × seq × seq]│    │
│  │ • gate_proj [batch × seq × inter]      │    │
│  │ • up_proj   [batch × seq × inter]      │    │
│  │ • ffn_out   [batch × seq × hidden]     │    │
│  │ • logits    [batch × seq × vocab]      │    │
│  │ • rope_cache [max_seq × heads × d]     │    │
│  └────────────────────────────────────────┘    │
│                                                  │
│  KV Cache:                                       │
│  ┌────────────────────────────────────────┐    │
│  │ • key_cache   [batch][layers][seq][..] │    │
│  │ • value_cache [batch][layers][seq][..] │    │
│  │                                         │    │
│  │ With compression (optional):            │    │
│  │ • compression_config                    │    │
│  │ • actual_tokens_stored                  │    │
│  │ • compressed_tokens                     │    │
│  └────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

### Memory Allocation Strategy

1. **At Model Load**:
   - Allocate all weight buffers from model file
   - Allocate all scratch buffers based on max sequence length
   - Allocate KV cache based on batch size and max sequence length

2. **During Inference**:
   - Zero new allocations
   - Reuse scratch buffers with ping-pong pattern
   - Update KV cache in-place

3. **Memory Footprint** (example: 7B model, batch=1, seq=2048):
   - Weights: ~7GB (FP16) or ~3.5GB (Q4_K)
   - Scratch buffers: ~500MB
   - KV cache: ~1GB
   - **Total**: ~8.5GB VRAM (FP16) or ~5GB (Q4_K)

---

## Inference Pipeline

### Single Token Generation

```
┌────────────────────────────────────────────────────────────┐
│              Single Command Encoder (per token)            │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Embedding Lookup                                        │
│     ├─ Bind: token_ids, embedding_weights                  │
│     └─ Dispatch: embed_lookup kernel                       │
│                                                             │
│  2. For each transformer layer (0..N):                      │
│     │                                                       │
│     ├─ RMSNorm (attention input)                           │
│     │  ├─ Bind: input, attn_norm_weights                   │
│     │  └─ Dispatch: rmsnorm kernel                         │
│     │                                                       │
│     ├─ QKV Projections (parallel)                          │
│     │  ├─ Bind: normed_input, q_weight → q_proj           │
│     │  ├─ Bind: normed_input, k_weight → k_proj           │
│     │  └─ Bind: normed_input, v_weight → v_proj           │
│     │  └─ Dispatch: 3x gemm kernels                        │
│     │                                                       │
│     ├─ RoPE (rotary position embedding)                    │
│     │  ├─ Bind: q_proj, k_proj, rope_cache, position      │
│     │  └─ Dispatch: rope kernel                            │
│     │                                                       │
│     ├─ Update KV Cache                                      │
│     │  ├─ Bind: k_proj, v_proj → kv_cache[layer]          │
│     │  └─ Dispatch: update_cache kernel                    │
│     │                                                       │
│     ├─ (Optional) Compress KV Cache                        │
│     │  ├─ Check: if should_compress()                      │
│     │  ├─ Bind: kv_cache → compressed                      │
│     │  └─ Dispatch: compress_cache kernel                  │
│     │                                                       │
│     ├─ Attention                                            │
│     │  ├─ Compute scores: Q @ K^T / √d                     │
│     │  ├─ Softmax: scores → probs                          │
│     │  ├─ Compute output: probs @ V                        │
│     │  └─ Dispatch: attention kernel                       │
│     │                                                       │
│     ├─ Output Projection                                    │
│     │  ├─ Bind: attn_out, o_weight → hidden               │
│     │  └─ Dispatch: gemm kernel                            │
│     │                                                       │
│     ├─ Residual Add                                         │
│     │  └─ Dispatch: add kernel                             │
│     │                                                       │
│     ├─ RMSNorm (FFN input)                                 │
│     │  ├─ Bind: hidden, ffn_norm_weights                   │
│     │  └─ Dispatch: rmsnorm kernel                         │
│     │                                                       │
│     ├─ FFN                                                  │
│     │  ├─ Gate projection + SiLU                           │
│     │  ├─ Up projection                                    │
│     │  ├─ Element-wise multiply                            │
│     │  └─ Down projection                                  │
│     │  └─ Dispatch: ffn kernel                             │
│     │                                                       │
│     └─ Residual Add                                         │
│        └─ Dispatch: add kernel                             │
│                                                             │
│  3. Final RMSNorm                                           │
│     ├─ Bind: hidden, output_norm_weights                   │
│     └─ Dispatch: rmsnorm kernel                            │
│                                                             │
│  4. LM Head                                                 │
│     ├─ Bind: normed, lm_head_weights → logits             │
│     └─ Dispatch: gemm kernel                               │
│                                                             │
│  5. Submit Command Buffer                                   │
│     └─ queue.submit([encoder.finish()])                    │
│                                                             │
│  6. Read Logits                                             │
│     ├─ Map buffer (async)                                  │
│     └─ Copy to CPU                                         │
│                                                             │
│  7. Sample Next Token                                       │
│     ├─ Apply temperature                                   │
│     ├─ Top-p / Top-k filtering                             │
│     ├─ Multinomial sampling                                │
│     └─ Return token_id                                     │
└────────────────────────────────────────────────────────────┘
                           │
                           ▼
                   Next iteration (append to KV cache)
```

### Performance Optimization Summary

| Optimization | Impact | Description |
|-------------|--------|-------------|
| Static Computation Graph | 97-98% reduction | Pre-allocated buffers, single encoder |
| Pipeline Caching | ~20% speedup | Pre-compiled shaders |
| Zero Dynamic Allocations | Predictable perf | No GC pauses, consistent latency |
| Ping-Pong Buffers | Eliminates copies | Alternate between two buffers |
| KV Cache Compression | 2-4x context length | Extend context with minimal quality loss |
| Speculative Decoding | 1.5-3x speedup | Draft model + verification |

---

## Plugin System

```
┌────────────────────────────────────────────────────┐
│                 Plugin Architecture                 │
├────────────────────────────────────────────────────┤
│                                                     │
│  ┌────────────────────────────────────────┐        │
│  │         Plugin Loading                  │        │
│  │                                         │        │
│  │  1. Discover .so/.dll files             │        │
│  │  2. dlopen() / LoadLibrary()            │        │
│  │  3. Find create_plugin() symbol         │        │
│  │  4. Call factory function               │        │
│  └────────────────────────────────────────┘        │
│                       │                             │
│                       ▼                             │
│  ┌────────────────────────────────────────┐        │
│  │      JanusPlugin Interface (FFI)        │        │
│  │                                         │        │
│  │  • init(config_json)                   │        │
│  │  • info() → PluginInfo                 │        │
│  │  • analyze(context) → RoutingPreference│        │
│  │  • infer_stream(context, callback)     │        │
│  │  • infer_blocking(context) → response  │        │
│  │  • shutdown()                           │        │
│  └────────────────────────────────────────┘        │
│                       │                             │
│                       ▼                             │
│  ┌────────────────────────────────────────┐        │
│  │      ABI Stability (abi_stable)         │        │
│  │                                         │        │
│  │  • RStr<'_>  (FFI-safe string)         │        │
│  │  • RVec<T>   (FFI-safe vector)         │        │
│  │  • RResult<T, E> (FFI-safe Result)     │        │
│  │  • RBox<T>   (FFI-safe Box)            │        │
│  └────────────────────────────────────────┘        │
└────────────────────────────────────────────────────┘
```

### Plugin Lifecycle

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  Create  │────▶│   Init   │────▶│   Use    │────▶│ Shutdown │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
     │                │                 │                 │
     ▼                ▼                 ▼                 ▼
Load plugin     Parse config    Process requests   Clean up
from .so/.dll   Initialize      • analyze()        Free resources
                resources       • infer_stream()
                                • infer_blocking()
```

---

## Router Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  DeterministicRouter                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input: RoutingRequest                                   │
│  ├─ prompt: String                                       │
│  ├─ estimated_tokens: usize                              │
│  └─ system_state: SystemState                            │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │            Routing Decision Tree                │    │
│  │                                                  │    │
│  │  1. Check Local Engine Availability              │    │
│  │     ├─ If NOT available → CLOUD                  │    │
│  │     └─ If available → Continue                   │    │
│  │                                                  │    │
│  │  2. Check VRAM Exhaustion                        │    │
│  │     ├─ If exhausted → CLOUD                      │    │
│  │     └─ If OK → Continue                          │    │
│  │                                                  │    │
│  │  3. Check Token Threshold                        │    │
│  │     ├─ If > MAX_LOCAL_TOKENS → CLOUD            │    │
│  │     └─ If ≤ threshold → Continue                │    │
│  │                                                  │    │
│  │  4. Check Complexity Keywords                    │    │
│  │     ├─ If complex → CLOUD                        │    │
│  │     └─ If simple → LOCAL                         │    │
│  └────────────────────────────────────────────────┘    │
│                                                          │
│  Output: RoutingDecision                                 │
│  └─ destination: Local | Cloud                           │
└─────────────────────────────────────────────────────────┘
```

---

## Design Principles

### 1. **GPU-First Architecture**
- All compute operations run on GPU
- Minimize CPU ↔ GPU transfers
- Maximize GPU occupancy

### 2. **Static Computation Graph**
- Pre-allocate all buffers at load time
- Single command encoder per token
- Predictable performance

### 3. **Zero-Copy Memory**
- Direct GPU buffer access
- No intermediate copies
- Memory-mapped when possible

### 4. **Fail-Safe Error Handling**
- No `unwrap()` or `expect()` in production
- Proper `Result<T, E>` propagation
- Graceful degradation

### 5. **Modularity**
- Clean separation of concerns
- Plugin-based extensibility
- Testable components

---

## Performance Characteristics

### GPU Submissions Per Token

```
Before Optimization:  30-50+ submissions
                      ├─ Embedding: 1
                      ├─ Layer 0: 15-20
                      ├─ Layer 1: 15-20
                      ├─ ...
                      └─ Output: 2-3

After Optimization:   1 submission
                      └─ Entire forward pass
```

### Memory Footprint (7B Model)

| Component | FP16 | Q4_K |
|-----------|------|------|
| Weights | 7.0 GB | 3.5 GB |
| Scratch Buffers | 0.5 GB | 0.5 GB |
| KV Cache | 1.0 GB | 1.0 GB |
| **Total** | **8.5 GB** | **5.0 GB** |

### Typical Performance (RTX 4090, 7B model)

- **Prompt processing**: 200-300 tok/s
- **Token generation**: 40-60 tok/s
- **Latency per token**: 16-25ms
- **Memory bandwidth**: ~450 GB/s utilized

---

## Future Enhancements

1. **Multi-GPU Support**: Tensor parallelism across GPUs
2. **Mixed Precision**: FP16/FP32 hybrid computation
3. **Flash Attention**: More efficient attention kernels
4. **Model Quantization**: INT8/INT4 quantization support
5. **Continuous Batching**: Dynamic batch size adjustment

---

*This architecture document is maintained alongside the codebase. For implementation details, see the source code in `crates/janus-engine/`.*
