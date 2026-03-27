# Janus

[![License: LGPL-3.0-or-later](https://img.shields.io/badge/License-LGPL%203.0+-blue.svg)](LICENSE)

Janus is a Rust workspace for local, GPU-accelerated LLM inference with a modular server architecture.
It includes a WebGPU-based engine, an OpenAI-compatible chat API server, deterministic routing, and a set
of composable `janus-mod-*` modules.

## Highlights

- GPU inference via `wgpu` (Vulkan/Metal/DX12 backends)
- Model loading from GGUF and Safetensors
- OpenAI-compatible chat endpoint: `POST /v1/chat/completions`
- Streaming responses via Server-Sent Events (SSE)
- Built-in web chat UI (`GET /chat`) and health check (`GET /health`)
- Modular architecture with in-process module plugins via `JanusPlugin` and pluggable `janus-mod-*` crates

## Workspace Layout

Current workspace members in `Cargo.toml`:

- `crates/janus-engine` - core inference engine, model formats, generation
- `crates/janus-server` - HTTP server and OpenAI-compatible API
- `crates/janus-mod-router` - routing module and deterministic routing primitives
- `crates/janus-mod-*` - modular server plugins (instruct, routing, vision, tts, etc.)

There are also legacy example plugin crates under `crates/plugins/` that are not part of the workspace.

## Prerequisites

- Rust toolchain (workspace uses Rust 2024 edition)
- A supported GPU/runtime for `wgpu` (Vulkan, Metal, or DirectX 12)
- Model assets:
  - `model.gguf` or `model.safetensors`
  - `tokenizer.json`
  - `config.json` (required for Safetensors)

## Build and Test

```bash
# Build all crates
cargo build --workspace --release

# Run tests
cargo test --workspace

# Optional lint/format checks
cargo fmt --all -- --check
cargo clippy --workspace -- -D warnings
```

## Quick Start

### 1) Run local inference example

Directory mode (recommended):

```bash
cargo run -p janus-engine --example inference --release -- path/to/model_dir "Hello from Janus"
```

File mode examples are documented in `crates/janus-engine/examples/inference.rs`.

### 2) Run the chat server

```bash
cargo run -p janus-server --release -- path/to/model_dir --host 0.0.0.0 --port 8080
```

You can also pass a model file directly (`.gguf` or `.safetensors`) instead of a directory.

### 3) Call the OpenAI-compatible API

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is Janus?"}
    ],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

For streaming, set `"stream": true`.

## API Endpoints

- `GET /` - HTML docs/landing page
- `GET /chat` - interactive browser chat UI
- `GET /health` - health + loaded model name
- `POST /v1/chat/completions` - OpenAI-compatible chat completions (streaming and non-streaming)

## Notes on Routing and Plugins

- `janus-mod-router` provides deterministic local/cloud routing heuristics.
- Plugins are module-based and compose through `janus_engine::JanusPlugin` at app startup.
- `janus-server` currently wires in multiple `janus-mod-*` plugins at startup (instruct, router,
  knowledge, lora, rp, tts, vecmem, vision, vismem, voice, imggen).

## Documentation

- `doc/ARCHITECTURE.md`
- `doc/SHADER_GUIDE.md`
- `doc/PERFORMANCE_TUNING.md`
- `doc/SUPPORTED_MODELS.md`
- `doc/FP16_IMPLEMENTATION.md`
- `doc/SERVER_README.md`

## Contributing

Contributions are welcome. Before opening a PR, please run:

```bash
cargo test --workspace
cargo fmt --all
cargo clippy --workspace -- -D warnings
```

## License

LGPL-3.0-or-later. See `LICENSE`.
