//! OpenAI-compatible chat completion API server
//!
//! This module provides an HTTP server with OpenAI-compatible endpoints for
//! chat completions, enabling drop-in replacement for OpenAI API clients.

pub mod handlers;
pub mod models;
pub mod routes;

pub use handlers::ChatCompletionHandler;
pub use models::{ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk};
pub use routes::create_router;
