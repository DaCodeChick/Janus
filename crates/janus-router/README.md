# janus-router

Intelligent routing logic for the Janus Engine that decides whether to route inference requests to a local GGUF engine or a cloud API based on multiple heuristics.

## Features

- **Deterministic Routing**: Predictable, rule-based routing decisions
- **Multiple Heuristics**: 
  - Complexity detection via keyword matching
  - Token threshold enforcement
  - VRAM availability checking
  - Local engine availability verification
- **Configurable**: All heuristics and thresholds can be customized
- **Well-Tested**: Comprehensive test suite with 26+ unit tests
- **Zero Dependencies**: Pure Rust with no external dependencies

## Usage

### Basic Example

```rust
use janus_router::{DeterministicRouter, RoutingRequest, SystemState};

let router = DeterministicRouter::new();

// Simple query -> LocalEngine
let request = RoutingRequest::new(
    "Hello, how are you?".to_string(),
    50,
    SystemState::default(),
);

let destination = router.route(&request);
println!("Route to: {:?}", destination);
```

### Custom Configuration

```rust
use janus_router::{DeterministicRouter, RouterConfig};

let config = RouterConfig::new()
    .with_max_local_tokens(4096)
    .add_complexity_keyword("translate")
    .add_complexity_keyword("summarize")
    .set_complexity_check(true);

let router = DeterministicRouter::with_config(config);
```

## Routing Heuristics

The router evaluates requests in priority order:

### 1. Local Engine Availability (Highest Priority)
If the local engine is unavailable, **always route to CloudAPI**.

### 2. VRAM Exhaustion
If GPU VRAM is below 10% available, **route to CloudAPI**.

### 3. Token Threshold
If the request exceeds the configured token limit (default: 8192), **route to CloudAPI**.

### 4. Complexity Check
If the prompt contains complexity keywords, **route to CloudAPI**.

Default keywords:
- "analyze"
- "refactor"
- "code"
- "debug"
- "optimize"
- "review"
- "architect"
- "design pattern"
- "algorithm"

### 5. Default
For simple conversational queries, **route to LocalEngine**.

## System State

The router requires system state information to make informed decisions:

```rust
use janus_router::SystemState;

let state = SystemState::new(
    6 * 1024 * 1024 * 1024,  // 6GB VRAM available
    8 * 1024 * 1024 * 1024,  // 8GB VRAM total
    30,                       // 30% GPU utilization
    true,                     // Local engine available
);
```

## Configuration Options

All routing heuristics can be enabled/disabled:

```rust
let config = RouterConfig::new()
    .set_complexity_check(false)  // Disable complexity checking
    .set_token_check(false)        // Disable token threshold
    .set_vram_check(false);        // Disable VRAM checking
```

## Testing

Run the comprehensive test suite:

```bash
cargo test -p janus-router
```

All 26 tests cover:
- Simple queries
- Complexity keyword detection (case-insensitive)
- Token threshold enforcement
- VRAM exhaustion handling
- Local engine availability
- Custom configurations
- Disabled heuristics
- Edge cases (zero tokens, empty prompts, etc.)
- Priority ordering
- Builder pattern functionality

## License

LGPL-3.0-or-later
