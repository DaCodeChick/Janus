//! Example usage of the DeterministicRouter
//!
//! This example demonstrates how to use the router to make routing decisions
//! based on prompt complexity, token count, and system resources.

use janus_router::{DeterministicRouter, RouterConfig, RoutingRequest, SystemState};

fn main() {
    // Create a router with default configuration
    let router = DeterministicRouter::new();

    // Example 1: Simple conversational query -> LocalEngine
    let simple_request = RoutingRequest::new(
        "Hello, how are you today?".to_string(),
        50,
        SystemState::default(),
    );
    println!(
        "Simple query routes to: {:?}",
        router.route(&simple_request)
    );

    // Example 2: Complex code analysis -> CloudAPI
    let complex_request = RoutingRequest::new(
        "Please analyze this code for performance bottlenecks and refactor it".to_string(),
        500,
        SystemState::default(),
    );
    println!(
        "Complex query routes to: {:?}",
        router.route(&complex_request)
    );

    // Example 3: High token count -> CloudAPI
    let large_context_request = RoutingRequest::new(
        "Tell me a long story".to_string(),
        10000, // Exceeds default 8192 token limit
        SystemState::default(),
    );
    println!(
        "Large context query routes to: {:?}",
        router.route(&large_context_request)
    );

    // Example 4: Low VRAM -> CloudAPI
    let low_vram_state = SystemState::new(
        200 * 1024 * 1024,      // 200MB available
        8 * 1024 * 1024 * 1024, // 8GB total
        95,
        true,
    );
    let low_vram_request =
        RoutingRequest::new("What is the weather?".to_string(), 50, low_vram_state);
    println!(
        "Low VRAM query routes to: {:?}",
        router.route(&low_vram_request)
    );

    // Example 5: Custom configuration
    let custom_config = RouterConfig::new()
        .with_max_local_tokens(4096)
        .add_complexity_keyword("translate")
        .add_complexity_keyword("summarize");

    let custom_router = DeterministicRouter::with_config(custom_config);

    let translation_request = RoutingRequest::new(
        "Please translate this to Spanish".to_string(),
        100,
        SystemState::default(),
    );
    println!(
        "Translation query (custom config) routes to: {:?}",
        custom_router.route(&translation_request)
    );

    // Example 6: Disabled heuristics
    let permissive_config = RouterConfig::new()
        .set_complexity_check(false)
        .set_token_check(false);

    let permissive_router = DeterministicRouter::with_config(permissive_config);

    let heavy_request = RoutingRequest::new(
        "Analyze and refactor this massive codebase".to_string(),
        50000,
        SystemState::default(),
    );
    println!(
        "Heavy query (permissive config) routes to: {:?}",
        permissive_router.route(&heavy_request)
    );
}
