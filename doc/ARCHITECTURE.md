# Janus Architecture

This document describes the current Janus workspace architecture after the move to module-based composition.

## Overview

Janus is organized as a Rust workspace with a core inference engine and a server that composes functionality
through in-process modules implementing `JanusPlugin`.

```
┌──────────────────────────────────────────────────────────────┐
│                           Janus                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  janus-server                                                 │
│  ├─ HTTP API (OpenAI-compatible chat endpoint)               │
│  ├─ SSE streaming                                             │
│  └─ Composes janus-mod-* modules                              │
│          │                                                    │
│          ▼                                                    │
│  janus-engine                                                 │
│  ├─ Model loading (GGUF / Safetensors)                       │
│  ├─ Transformer inference and sampling                        │
│  ├─ WebGPU compute backend (`wgpu`)                          │
│  └─ App composition API (`JanusApp`, `JanusPlugin`)          │
│                                                              │
│  janus-router                                                 │
│  └─ DeterministicRouter / RouterConfig / routing types       │
│                                                              │
│  Other janus-mod-* crates                                     │
│  ├─ instruct / knowledge / lora / rp                         │
│  ├─ tts / vecmem / vision / vismem / voice / imggen          │
│  └─ Extend app behavior during startup                        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Workspace Structure

Primary crates:

- `crates/janus-engine`: core compute, model, formats, and app composition abstractions
- `crates/janus-server`: HTTP server and API routes
- `crates/janus-router`: routing logic and deterministic router primitives
- `crates/janus-mod-*`: additional pluggable modules used by the server

Legacy directories:

- `crates/plugins/`: historical dynamic-plugin examples; not part of workspace members

## Module-Based Composition

The current extension mechanism is in-process and trait-based:

- `janus_engine::JanusPlugin` defines a simple `build(&self, app: &mut JanusApp)` hook.
- `janus_server` creates a `JanusApp`, adds selected `janus-mod-*` modules, then finalizes server state.
- Modules can register or mutate app components through `JanusApp` setters.

This replaces the previous external API-crate split and avoids an ABI boundary in normal usage.

## Routing Architecture

Routing primitives now live in `janus-router`:

- `DeterministicRouter`
- `RouterConfig`
- `RoutingRequest`
- `SystemState`
- `RouteDestination`

The decision flow remains heuristic and deterministic:

1. Local engine availability
2. VRAM exhaustion threshold
3. Token threshold limit
4. Prompt complexity keywords
5. Default to local engine

## Inference Pipeline (High Level)

At a high level, token generation in `janus-engine` follows:

1. Load tensors/config from GGUF or Safetensors
2. Build transformer blocks and static buffers
3. Run compute passes through `wgpu`
4. Produce logits and sample next token
5. Repeat until stop condition

The codebase emphasizes static allocation and predictable runtime behavior.

## Serving Flow

`janus-server` startup sequence:

1. Parse CLI arguments (`model`, `host`, `port`, optional template override)
2. Resolve model/tokenizer/config paths
3. Initialize `ComputeEngine`
4. Load tensors + build `Model`
5. Build app + add modules (`janus-mod-*`)
6. Install routes and bind address
7. Start axum server

## Notes

- This architecture intentionally focuses on in-workspace module composition.
- Dynamic loading infrastructure in `janus-engine/src/loader` remains a placeholder.
