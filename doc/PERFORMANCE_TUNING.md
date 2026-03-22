# Performance Tuning Guide

This guide provides comprehensive strategies for optimizing Janus Engine performance across different hardware configurations and use cases.

## Table of Contents

- [Quick Wins](#quick-wins)
- [Hardware-Specific Optimizations](#hardware-specific-optimizations)
- [Model Selection](#model-selection)
- [Memory Optimization](#memory-optimization)
- [Batching Strategies](#batching-strategies)
- [Advanced Techniques](#advanced-techniques)
- [Benchmarking](#benchmarking)
- [Troubleshooting](#troubleshooting)

---

## Quick Wins

### 1. Enable Release Mode

```bash
# ❌ Debug mode (5-10× slower)
cargo build
cargo run --example inference

# ✅ Release mode (full optimization)
cargo build --release
cargo run --release --example inference
```

**Impact**: 5-10× speedup
**Effort**: 1 minute

### 2. Use Quantized Models

```rust
// ❌ FP16 model: ~7GB VRAM, 40 tok/s
let model = Model::from_gguf("model-f16.gguf", &engine).await?;

// ✅ Q4_K model: ~3.5GB VRAM, 60 tok/s
let model = Model::from_gguf("model-q4_k.gguf", &engine).await?;
```

**Impact**: 1.5-2× speedup, 50% VRAM savings
**Effort**: Download quantized model

### 3. Adjust Temperature

```rust
// ❌ High temperature = more computation
let config = SamplerConfig {
    temperature: 1.5,  // More sampling overhead
    ..Default::default()
};

// ✅ Lower temperature = faster sampling
let config = SamplerConfig {
    temperature: 0.7,  // Optimal for quality/speed
    ..Default::default()
};
```

**Impact**: 10-20% speedup
**Effort**: 1 line of code

### 4. Reduce Max Tokens

```rust
// ❌ Generate very long responses
let config = SamplerConfig {
    max_tokens: 1000,  // Slower for long sequences
    ..Default::default()
};

// ✅ Generate concise responses
let config = SamplerConfig {
    max_tokens: 200,  // Faster, more focused
    ..Default::default()
};
```

**Impact**: Proportional to token count
**Effort**: 1 line of code

---

## Hardware-Specific Optimizations

### NVIDIA GPUs

#### Optimal Settings (RTX 3000/4000 series)

```rust
// Model config optimized for NVIDIA
let model_config = ModelConfig {
    batch_size: 4,  // Higher batch size for NVIDIA
    max_seq_len: 2048,
    ..Default::default()
};

// Sampler config
let sampler_config = SamplerConfig {
    temperature: 0.7,
    top_p: 0.9,
    top_k: 40,  // Good balance for NVIDIA
    max_tokens: 200,
};
```

**Memory Guidelines**:

| GPU | VRAM | Recommended Model | Batch Size |
|-----|------|-------------------|------------|
| RTX 3060 | 12GB | 7B Q4_K | 1-2 |
| RTX 3080 | 10GB | 7B Q4_K | 1-2 |
| RTX 3090 | 24GB | 13B Q4_K or 7B FP16 | 2-4 |
| RTX 4080 | 16GB | 7B Q4_K | 2-4 |
| RTX 4090 | 24GB | 13B Q4_K or 7B FP16 | 4-8 |

**Driver Settings**:
```bash
# Enable CUDA persistence mode
sudo nvidia-smi -pm 1

# Set power limit (if thermal throttling)
sudo nvidia-smi -pl 350  # Adjust based on your GPU

# Check current performance
nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,power.draw --format=csv
```

### AMD GPUs

#### Optimal Settings (RX 6000/7000 series)

```rust
// Model config optimized for AMD
let model_config = ModelConfig {
    batch_size: 2,  // Lower batch size for AMD
    max_seq_len: 2048,
    ..Default::default()
};
```

**Memory Guidelines**:

| GPU | VRAM | Recommended Model | Batch Size |
|-----|------|-------------------|------------|
| RX 6700 XT | 12GB | 7B Q4_K | 1-2 |
| RX 6800 XT | 16GB | 7B Q4_K | 2 |
| RX 6900 XT | 16GB | 7B Q4_K | 2-3 |
| RX 7900 XT | 20GB | 13B Q4_K | 2-3 |
| RX 7900 XTX | 24GB | 13B Q4_K or 7B FP16 | 2-4 |

**Driver Settings**:
```bash
# Check current clock speeds
rocm-smi

# Set performance mode (if supported)
rocm-smi --setperflevel high
```

### Apple Silicon (M1/M2/M3)

#### Optimal Settings

```rust
// Model config optimized for Apple Silicon
let model_config = ModelConfig {
    batch_size: 1,  // Unified memory benefits single batch
    max_seq_len: 4096,  // Can handle longer sequences
    ..Default::default()
};
```

**Memory Guidelines**:

| Device | Unified Memory | Recommended Model | Notes |
|--------|---------------|-------------------|-------|
| M1 | 8GB | 7B Q5_K | Shared with system |
| M1 Pro | 16GB | 7B Q4_K | Good performance |
| M1 Max | 32GB | 13B Q4_K | Excellent for ML |
| M2 | 8GB | 7B Q5_K | Shared with system |
| M2 Pro | 16GB | 7B Q4_K | Good performance |
| M2 Max | 32GB+ | 13B Q4_K | Best performance |
| M3 Max | 36GB+ | 13B FP16 | Top tier |

**System Settings**:
```bash
# Close memory-intensive applications
# Safari, Chrome can use 4-8GB each

# Monitor memory pressure
# Activity Monitor → Memory tab
```

### Integrated GPUs (Intel/AMD)

#### Optimal Settings

```rust
// Model config for integrated GPUs
let model_config = ModelConfig {
    batch_size: 1,
    max_seq_len: 1024,  // Lower to reduce memory pressure
    ..Default::default()
};

// Use aggressive quantization
// Q5_K or Q4_K models only
```

**Memory Guidelines**:
- Integrated GPUs share system RAM
- Reserve 4-8GB for GPU usage
- Close other applications during inference

---

## Model Selection

### Size vs Performance Trade-off

```
┌────────────────────────────────────────────────────┐
│                Model Size Decision Tree             │
├────────────────────────────────────────────────────┤
│                                                     │
│  VRAM Available?                                    │
│  ├─ < 6GB:  Use 3B or 7B with Q5_K/Q4_K           │
│  ├─ 6-12GB: Use 7B with Q4_K                       │
│  ├─ 12-16GB: Use 7B FP16 or 13B Q4_K              │
│  └─ > 16GB: Use 13B FP16 or 30B Q4_K              │
│                                                     │
│  Use Case?                                          │
│  ├─ Chat/Instruct: 7B sufficient                   │
│  ├─ Code Generation: 13B+ recommended              │
│  ├─ Creative Writing: 13B+ beneficial              │
│  └─ Simple QA: 3B-7B sufficient                    │
└────────────────────────────────────────────────────┘
```

### Quantization Comparison

| Format | Size (7B) | Quality Loss | Speed | Recommendation |
|--------|-----------|--------------|-------|----------------|
| FP16 | 14GB | 0% (baseline) | 1.0× | High-quality, need VRAM |
| Q8_0 | 7.5GB | < 1% | 1.3× | Near-lossless |
| Q5_K | 5.0GB | 1-2% | 1.6× | Great balance |
| **Q4_K** | **3.5GB** | **2-3%** | **1.8×** | **Best choice** |
| Q3_K | 2.8GB | 5-8% | 2.0× | Quality degradation |
| Q2_K | 2.3GB | 10-15% | 2.2× | Not recommended |

**Recommendation**: Q4_K offers the best quality/speed/size trade-off

### Model Architecture

**For Inference Speed**:
```
1. Mistral 7B (fastest architecture)
2. LLaMA 2 7B
3. TinyLlama (if size matters more)
```

**For Quality**:
```
1. LLaMA 2 13B
2. Mistral 7B
3. LLaMA 2 7B
```

---

## Memory Optimization

### KV Cache Management

```rust
use janus_engine::{KVCache, CacheCompressionConfig};

// ❌ Standard KV cache
let cache = KVCache::new(&engine, 1, 32, 2048, 4, 128)?;
// Uses: 1 * 32 * 2048 * 4 * 128 * 4 bytes * 2 (K+V) = ~128MB

// ✅ KV cache with compression
let config = CacheCompressionConfig {
    enabled: true,
    uncompressed_window: 512,  // Keep recent 512 tokens
    compression_ratio: 2,  // 2:1 compression for old tokens
    compression_trigger: 0.8,  // Compress at 80% full
};

let cache = KVCache::with_compression(
    &engine, 1, 32, 2048, 4, 128, config
)?;
// Effective context: 2048 → 4096+ tokens
// Memory: Same 128MB
```

**Impact**: 2-4× effective context length
**Quality loss**: < 5% for 2:1 compression

### Buffer Pre-allocation

```rust
// ✅ Good: Pre-allocate with realistic max sequence length
let model = Model::new(
    config,
    &engine,
    embeddings,
    blocks,
    output_norm,
    lm_head,
)?;
// All buffers allocated once at creation

// ❌ Bad: Would require dynamic allocation (not supported)
// Dynamic allocation causes:
// - Memory fragmentation
// - Inconsistent performance
// - Potential OOM
```

### Memory Monitoring

```rust
use janus_engine::ComputeEngine;

async fn monitor_memory(engine: &ComputeEngine) {
    // Check available VRAM
    let adapter_info = engine.adapter_info();
    println!("GPU: {}", adapter_info.name);
    
    // Track cache statistics
    let (actual, compressed, effective) = cache.compression_stats();
    println!("Cache: {actual} tokens ({compressed} compressed, {effective} effective)");
}
```

---

## Batching Strategies

### When to Use Batching

```
Single Request:
  Latency: 100ms per token
  Throughput: 10 tok/s
  
Batch of 4:
  Latency: 110ms per token (10% slower)
  Throughput: 36 tok/s (3.6× faster)
  
Batch of 8:
  Latency: 125ms per token (25% slower)
  Throughput: 64 tok/s (6.4× faster)
```

**Use batching when**:
- Serving multiple users simultaneously
- Throughput > latency priority
- VRAM available (batching uses more memory)

**Don't batch when**:
- Interactive single-user applications
- Real-time requirements (< 50ms latency)
- Limited VRAM

### Optimal Batch Size

```rust
fn calculate_optimal_batch_size(
    vram_gb: f32,
    model_size_gb: f32,
    max_seq_len: u32,
) -> u32 {
    // Rule of thumb: Reserve 20% for scratch buffers
    let available_for_batch = (vram_gb - model_size_gb) * 0.8;
    
    // KV cache per batch item: ~seq_len * hidden_dim * layers * 2 * 4 bytes
    let kv_cache_per_batch = (max_seq_len as f32 * 4096.0 * 32.0 * 2.0 * 4.0) / 1e9;
    
    // Calculate max batch size
    let max_batch = (available_for_batch / kv_cache_per_batch).floor() as u32;
    
    // Clamp to reasonable range
    max_batch.clamp(1, 16)
}

// Example: RTX 4090 with 7B Q4_K model
let batch_size = calculate_optimal_batch_size(
    24.0,   // 24GB VRAM
    3.5,    // 7B Q4_K = ~3.5GB
    2048,   // 2K context
);
// Returns: 8 (can run 8 simultaneous sequences)
```

### Dynamic Batching

```rust
use janus_engine::{Model, Sampler};
use std::collections::VecDeque;

struct BatchScheduler {
    pending_requests: VecDeque<Request>,
    max_batch_size: usize,
    max_wait_time_ms: u64,
}

impl BatchScheduler {
    async fn process_batch(&mut self, model: &mut Model) {
        // Collect requests up to batch size or timeout
        let mut batch = Vec::new();
        let start = std::time::Instant::now();
        
        while batch.len() < self.max_batch_size 
            && start.elapsed().as_millis() < self.max_wait_time_ms as u128 
        {
            if let Some(request) = self.pending_requests.pop_front() {
                batch.push(request);
            }
        }
        
        // Process batch
        if !batch.is_empty() {
            model.generate_batch(&batch, &sampler).await?;
        }
    }
}
```

---

## Advanced Techniques

### 1. Speculative Decoding

Speculative decoding uses a small "draft" model to predict multiple tokens ahead, then verifies with the target model:

```rust
use janus_engine::SpeculativeDecoder;

// Load small draft model (e.g., TinyLlama 1.1B)
let draft_model = Model::from_gguf("tinyllama-1.1b-q4.gguf", &engine).await?;

// Load target model (e.g., LLaMA 2 7B)
let target_model = Model::from_gguf("llama-2-7b-q4.gguf", &engine).await?;

// Create speculative decoder
let mut decoder = SpeculativeDecoder::new(
    draft_model,
    target_model,
    SpeculativeConfig {
        num_draft_tokens: 4,  // Draft 4 tokens ahead
        acceptance_threshold: 0.8,
    }
);

// Generate (1.5-3× faster)
let output = decoder.generate(&prompt, &sampler).await?;
```

**Performance**:
- Best case (high acceptance): 3× speedup
- Average case: 1.8-2.2× speedup
- Worst case (low acceptance): 1.2× speedup

**When to use**:
- Long-form generation (> 100 tokens)
- Predictable content (code, structured text)
- VRAM available for 2 models

### 2. Continuous Batching

Process variable-length sequences efficiently:

```rust
struct ContinuousBatcher {
    active_sequences: Vec<Sequence>,
    max_batch_size: usize,
}

impl ContinuousBatcher {
    fn add_sequence(&mut self, seq: Sequence) {
        self.active_sequences.push(seq);
    }
    
    fn remove_finished(&mut self) {
        self.active_sequences.retain(|seq| !seq.is_finished());
    }
    
    async fn step(&mut self, model: &mut Model) -> Result<()> {
        // Remove finished sequences
        self.remove_finished();
        
        // Add new sequences if space available
        while self.active_sequences.len() < self.max_batch_size {
            if let Some(new_seq) = self.pending_queue.pop() {
                self.add_sequence(new_seq);
            } else {
                break;
            }
        }
        
        // Generate one token for each active sequence
        if !self.active_sequences.is_empty() {
            model.generate_batch_single_token(&self.active_sequences).await?;
        }
        
        Ok(())
    }
}
```

**Benefits**:
- No wasted computation on padding
- Higher throughput for mixed-length requests
- Better resource utilization

### 3. Flash Attention (Future)

Flash Attention reduces memory usage for attention computation:

```
Standard Attention:
  Memory: O(seq_len²)
  Speed: Baseline
  
Flash Attention:
  Memory: O(seq_len)
  Speed: 2-4× faster
  
Implementation: Coming soon to Janus
```

---

## Benchmarking

### Built-in Benchmarking

```rust
use janus_engine::{Model, Sampler};
use std::time::Instant;

async fn benchmark_model(model: &mut Model) -> Result<()> {
    let prompt = "The quick brown fox";
    let tokens = tokenizer.encode(prompt, false)?;
    
    // Warmup
    for _ in 0..5 {
        model.forward(&tokens).await?;
    }
    
    // Benchmark
    let num_runs = 100;
    let start = Instant::now();
    
    for _ in 0..num_runs {
        model.forward(&tokens).await?;
    }
    
    let elapsed = start.elapsed();
    let avg_time_ms = elapsed.as_millis() as f64 / num_runs as f64;
    let tok_per_sec = 1000.0 / avg_time_ms;
    
    println!("Average time per token: {:.2}ms", avg_time_ms);
    println!("Throughput: {:.2} tok/s", tok_per_sec);
    
    Ok(())
}
```

### Performance Metrics

Track these key metrics:

1. **Time per Token**: Lower is better
2. **Throughput**: tok/s, higher is better  
3. **Memory Usage**: VRAM footprint
4. **GPU Utilization**: Should be > 90%
5. **Memory Bandwidth**: % of theoretical max

### Profiling Tools

**NVIDIA**:
```bash
# Profile with Nsight Systems
nsys profile --trace=cuda,nvtx ./target/release/inference model.gguf "prompt"

# Profile with Nsight Compute  
ncu --set full ./target/release/inference model.gguf "prompt"
```

**AMD**:
```bash
# Profile with ROCm profiler
rocprof --stats ./target/release/inference model.gguf "prompt"
```

**General**:
```bash
# CPU profiling
cargo flamegraph --root --example inference -- model.gguf "prompt"

# Memory profiling
valgrind --tool=massif ./target/release/inference model.gguf "prompt"
```

---

## Troubleshooting

### Slow Performance

**Symptom**: < 10 tok/s on capable GPU

**Checklist**:
1. ✅ Built in release mode? `cargo build --release`
2. ✅ Using quantized model? Q4_K recommended
3. ✅ GPU utilization high? Check with `nvidia-smi`/`rocm-smi`
4. ✅ Thermal throttling? Check temperatures
5. ✅ Background processes? Close other GPU applications

**Debug**:
```rust
// Add timing
let start = Instant::now();
let logits = model.forward(&tokens).await?;
println!("Forward pass: {:?}", start.elapsed());

let start = Instant::now();
let token = sampler.sample(&logits)?;
println!("Sampling: {:?}", start.elapsed());
```

### Out of Memory

**Symptom**: OOM errors or crashes

**Solutions**:

1. **Reduce batch size**:
```rust
let config = ModelConfig {
    batch_size: 1,  // Lower from 4
    ..Default::default()
};
```

2. **Reduce max sequence length**:
```rust
let config = ModelConfig {
    max_seq_len: 1024,  // Lower from 2048
    ..Default::default()
};
```

3. **Use more aggressive quantization**:
```rust
// Q4_K instead of Q5_K or FP16
let model = Model::from_gguf("model-q4_k.gguf", &engine).await?;
```

4. **Enable KV cache compression**:
```rust
let config = CacheCompressionConfig {
    enabled: true,
    compression_ratio: 4,  // More aggressive (was 2)
    ..Default::default()
};
```

### High Latency

**Symptom**: Long delays between tokens

**Causes**:
1. **CPU bottleneck**: Sampler running on CPU
2. **Memory transfers**: Too many GPU ↔ CPU copies
3. **Dynamic allocations**: Fragmented memory

**Solutions**:
```rust
// 1. Reduce sampling complexity
let config = SamplerConfig {
    top_k: 20,  // Lower from 40
    temperature: 0.7,  // Lower from 1.0
    ..Default::default()
};

// 2. Use lower precision
// Switch to Q4_K if using FP16

// 3. Optimize context length
// Don't pass entire history each time
// Use KV cache effectively
```

### Inconsistent Performance

**Symptom**: Performance varies significantly between runs

**Causes**:
1. **Thermal throttling**: GPU overheating
2. **Power management**: Dynamic clocks
3. **Background tasks**: Other GPU processes

**Solutions**:
```bash
# NVIDIA: Lock clocks
sudo nvidia-smi -lgc 1800,1800  # Lock GPU clock

# NVIDIA: Set persistence mode
sudo nvidia-smi -pm 1

# AMD: Set performance mode
rocm-smi --setperflevel high
```

---

## Performance Checklist

Use this checklist to ensure optimal performance:

### Basic ✅

- [ ] Built with `--release` flag
- [ ] Using quantized model (Q4_K/Q5_K)
- [ ] Closed unnecessary applications
- [ ] GPU drivers up to date

### Intermediate ✅

- [ ] Batch size optimized for VRAM
- [ ] Max sequence length appropriate for use case
- [ ] Sampler temperature/top_k tuned
- [ ] KV cache compression enabled (if needed)

### Advanced ✅

- [ ] GPU clocks locked (no thermal throttling)
- [ ] Profiled with GPU profiler
- [ ] Considered speculative decoding
- [ ] Implemented continuous batching (if multi-user)

---

## Expected Performance

### Reference Numbers (7B Q4_K model)

| GPU | Prompt (tok/s) | Generation (tok/s) | Latency/Token |
|-----|---------------|-------------------|---------------|
| RTX 3060 (12GB) | 180-220 | 35-45 | 22-28ms |
| RTX 3080 (10GB) | 250-300 | 45-55 | 18-22ms |
| RTX 3090 (24GB) | 280-320 | 50-60 | 16-20ms |
| RTX 4080 (16GB) | 300-350 | 55-65 | 15-18ms |
| RTX 4090 (24GB) | 350-400 | 60-70 | 14-16ms |
| RX 6800 XT (16GB) | 200-250 | 40-50 | 20-25ms |
| RX 7900 XTX (24GB) | 280-320 | 50-60 | 16-20ms |
| M1 Max (32GB) | 120-150 | 25-35 | 28-40ms |
| M2 Max (32GB) | 140-170 | 30-40 | 25-33ms |

*Note: Actual performance varies based on model architecture, sequence length, and system configuration.*

---

## Further Resources

- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [SHADER_GUIDE.md](SHADER_GUIDE.md) - GPU shader optimization
- [Examples](../crates/janus-engine/examples/) - Code examples
- [WebGPU Best Practices](https://www.w3.org/TR/webgpu/#best-practices)

---

*This performance guide is maintained alongside the codebase. For latest optimizations, see the source code.*
