//! API routes and router configuration

use crate::handlers::{AppState, ChatCompletionHandler};
use axum::{routing::post, Router};
use std::sync::Arc;

/// Create the application router with all endpoints
pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(ChatCompletionHandler::handle))
        .with_state(state)
}
