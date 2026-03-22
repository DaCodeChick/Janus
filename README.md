# Janus Engine

[![License: LGPL-3.0-or-later](https://img.shields.io/badge/License-LGPL%203.0+-blue.svg)](LICENSE)

**Janus Engine** is a high-performance, GPU-accelerated LLM inference engine built in Rust with WebGPU. It achieves exceptional performance through advanced optimizations including static computation graphs, pipeline caching, and zero-copy memory management.

## 🚀 Key Features

- **GPU-Accelerated Inference**: Cross-platform GPU compute via WebGPU (wgpu)
- **Extreme Performance Optimization**:
  - Static computation graph with **1 GPU submission per token** (97-98% reduction)
  - Zero dynamic buffer allocations during inference
  - Pre-compiled shader pipeline cache (10-20% additional speedup)
- **Multiple Model Format Support**:
  - GGUF models (quantized and FP16)
  - Safetensors models
- **Flexible Architecture**:
  - Plugin system with ABI-stable FFI
  - Intelligent routing between local and cloud inference
  - HTTP API server with streaming support
- **Production-Ready**:
  - Strict error handling (no unwrap/expect in production code)
  - Comprehensive testing
  - Type-safe quantization support (Q4_K)

## 📊 Performance

Recent optimizations have achieved dramatic performance improvements:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| GPU Submissions/Token | 30-50+ | 1 | 97-98% reduction |
| Dynamic Allocations | Hundreds | 0 | 100% reduction |
| Shader Compilations | Per operation | At model load | ~20% speedup |

**Architecture Highlights:**
- Pre-allocated scratch buffers (15 buffers in Model struct)
- Ping-pong buffer pattern for transformer layers
- Single command encoder for entire forward pass
- High-precision benchmarking (millisecond accuracy)

## 🏗️ Architecture

```
janus/
├── janus-api/           # ABI-stable plugin API
├── janus-engine/        # Core GPU inference engine
│   ├── compute/         # GPU compute operations
│   │   ├── ops/         # Tensor operations (GEMM, RoPE, Attention, etc.)
│   │   ├── cache.rs     # KV cache for autoregressive generation
│   │   ├── engine.rs    # ComputeEngine (GPU device/queue)
│   │   └── pipeline_cache.rs  # Pre-compiled shader cache
│   ├── model/           # Transformer model implementation
│   │   ├── block.rs     # TransformerBlock (attention + FFN)
│   │   ├── model.rs     # Full Model with static computation graph
│   │   ├── output.rs    # LM Head for logits generation
│   │   └── sampler.rs   # Token sampling strategies
│   └── loaders/         # Model loading (GGUF, Safetensors)
├── janus-router/        # Intelligent routing logic
├── janus-server/        # HTTP API server
└── plugins/             # Plugin implementations
    ├── janus-instruct-plugin/
    └── janus-roleplay-plugin/
```

## 🛠️ Installation

### Prerequisites

- Rust 1.75+ (2024 edition)
- GPU with Vulkan/Metal/DirectX 12 support
- At least 4GB VRAM (8GB+ recommended for larger models)

### Build from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/janus-engine.git
cd janus-engine

# Build the project
cargo build --release

# Run tests
cargo test --workspace
```

## 📖 Usage

### Basic Inference Example

```rust
use janus_engine::{ComputeEngine, Model, ModelConfig, Tokenizer, Sampler};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize GPU compute engine
    let engine = ComputeEngine::new().await?;
    
    // Load model from GGUF file
    let model = Model::from_gguf("path/to/model.gguf", &engine).await?;
    
    // Load tokenizer
    let tokenizer = Tokenizer::from_file("tokenizer.json")?;
    
    // Generate text
    let prompt = "Hello, how are you?";
    let tokens = tokenizer.encode(prompt)?;
    
    let sampler = Sampler::new();
    let output = model.generate(&tokens, 50, &sampler).await?;
    
    println!("{}", tokenizer.decode(&output)?);
    Ok(())
}
```

### Running the Inference Example

**Directory mode** (auto-discovers model files):
```bash
cargo run --release --example inference path/to/model_dir "Your prompt here"
```

**File mode** (explicit paths):
```bash
cargo run --release --example inference \
    path/to/model.gguf \
    path/to/config.json \
    path/to/tokenizer.json \
    "Your prompt here"
```

### Supported Model Architectures

- **LLaMA/LLaMA 2** (Meta)
- **Mistral** (Mistral AI)
- **TinyLlama** (tiny but powerful)
- Any transformer model following the LLaMA architecture

## 🔌 Plugin System

Janus supports dynamic plugins via an ABI-stable FFI interface:

```rust
use janus_api::{Plugin, PluginCapabilities, ProcessingContext};

#[export_name = "janus_plugin_create"]
pub extern "C" fn create_plugin() -> Box<dyn Plugin> {
    Box::new(MyPlugin::new())
}
```

See [`crates/plugins/`](crates/plugins/) for example implementations.

## 🎯 Intelligent Routing

The router can automatically decide between local GPU inference and cloud APIs:

```rust
use janus_router::{DeterministicRouter, RoutingRequest, SystemState};

let router = DeterministicRouter::new();
let request = RoutingRequest::new(
    prompt.to_string(),
    token_count,
    SystemState::default(),
);

let destination = router.route(&request);
```

Routing heuristics include:
- Local engine availability
- VRAM exhaustion detection
- Token threshold enforcement
- Complexity keyword matching

See [janus-router/README.md](crates/janus-router/README.md) for details.

## 🌐 HTTP API Server

Run a local inference API server:

```bash
cargo run --release -p janus-server
```

The server provides:
- OpenAI-compatible `/v1/completions` endpoint
- Streaming responses via Server-Sent Events (SSE)
- Plugin-based text processing
- Model management

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
cargo test --workspace

# Run specific crate tests
cargo test -p janus-engine
cargo test -p janus-router

# Run with output
cargo test -- --nocapture
```

## 📈 Performance Benchmarking

The engine includes built-in high-precision benchmarking:

```rust
// Benchmark output during generation:
// Tokens generated: 50
// Elapsed time: 1234ms
// Speed: 40.52 tok/s
// GPU submissions: 50 (1.00 per token)
```

## 🔧 Configuration

### Workspace Configuration

Key settings in `Cargo.toml`:

```toml
[workspace.lints.clippy]
unwrap_used = "deny"      # Enforce proper error handling
expect_used = "deny"
panic = "warn"

[profile.release]
opt-level = 3             # Maximum optimization
lto = "thin"              # Link-time optimization
codegen-units = 1         # Single codegen unit for better optimization
```

### Model Configuration

Models can be configured via `config.json` (HuggingFace format):

```json
{
  "architectures": ["LlamaForCausalLM"],
  "hidden_size": 4096,
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 4,
  "intermediate_size": 11008,
  "rms_norm_eps": 1e-05
}
```

## 🤝 Contributing

Contributions are welcome! Please ensure:

1. Code follows the strict error handling policy (no `unwrap`/`expect`)
2. All tests pass: `cargo test --workspace`
3. Code is formatted: `cargo fmt`
4. Clippy is happy: `cargo clippy --workspace -- -D warnings`

## 📝 License

This project is licensed under the **LGPL-3.0-or-later** license. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Built with [wgpu](https://github.com/gfx-rs/wgpu) for cross-platform GPU compute
- Inspired by [llama.cpp](https://github.com/ggerganov/llama.cpp) and modern LLM inference engines
- Uses [HuggingFace](https://huggingface.co/) model formats and tokenizers

## 📚 Documentation

### Comprehensive Guides

- **[Architecture Guide](doc/ARCHITECTURE.md)** - System architecture, component diagrams, and design decisions
- **[Shader Implementation Guide](doc/SHADER_GUIDE.md)** - Detailed WGSL shader documentation and optimization
- **[Performance Tuning Guide](doc/PERFORMANCE_TUNING.md)** - Hardware-specific optimizations and benchmarking
- **[Supported Models](doc/SUPPORTED_MODELS.md)** - Model compatibility and configuration
- **[FP16 Implementation](doc/FP16_IMPLEMENTATION.md)** - Mixed-precision inference details

### Examples

- **[Basic Inference](crates/janus-engine/examples/inference.rs)** - Simple single-sequence generation
- **[Batch Inference](crates/janus-engine/examples/batch_inference.rs)** - Process multiple prompts in parallel
- **[Streaming Generation](crates/janus-engine/examples/streaming.rs)** - Token-by-token streaming output
- **[Plugin Development](crates/janus-engine/examples/plugin_development.rs)** - Create custom inference plugins

### API Reference

- [janus-router README](crates/janus-router/README.md) - Routing logic documentation
- [API Documentation](https://docs.rs/janus-engine) - Coming soon

## 🐛 Troubleshooting

### GPU Not Detected
Ensure you have proper GPU drivers installed:
- **Vulkan**: Install Vulkan SDK
- **Metal**: macOS 10.13+ with Metal support
- **DirectX 12**: Windows 10+ with DX12 support

### Out of Memory
Reduce model size or batch size. For large models, ensure you have sufficient VRAM:
- 7B models: 4-6GB VRAM
- 13B models: 8-12GB VRAM

### Slow Inference
Check that:
- Release mode is enabled: `cargo build --release`
- GPU is being used (check logs)
- No background GPU processes are competing for resources

**See [Performance Tuning Guide](doc/PERFORMANCE_TUNING.md) for detailed optimization strategies.**

---

**Status**: Active development | **Version**: 0.1.0 | **Rust Edition**: 2024
