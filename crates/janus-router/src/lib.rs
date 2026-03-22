//! Janus Router - Intelligent routing between local and cloud inference
//!
//! This crate provides deterministic routing logic to decide whether to route
//! inference requests to a local GGUF engine or a cloud API based on multiple
//! heuristics including prompt complexity, token count, and system resource availability.

// Enforce error handling best practices (per .opencode/AGENTS.md)
#![warn(clippy::unwrap_used)]
#![warn(clippy::expect_used)]
#![warn(clippy::panic)]
#![warn(clippy::todo)]
#![warn(clippy::unimplemented)]

pub mod router;
pub mod types;

// Re-export main types for convenience
pub use router::{DeterministicRouter, RouterConfig};
pub use types::{RouteDestination, RoutingRequest, SystemState};
