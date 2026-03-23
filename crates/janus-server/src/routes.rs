//! API routes and router configuration

use crate::handlers::{AppState, ChatCompletionHandler};
use axum::{routing::{get, post}, Router, Json, response::Html};
use serde_json::json;
use std::sync::Arc;

/// Root endpoint - displays server info
async fn root() -> Html<&'static str> {
    Html(r#"<!DOCTYPE html>
<html>
<head>
    <title>Janus Chat Server</title>
    <style>
        body { font-family: system-ui; max-width: 800px; margin: 40px auto; padding: 0 20px; }
        code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }
        pre { background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }
        h1 { color: #333; }
        .endpoint { margin: 20px 0; padding: 15px; border-left: 4px solid #007bff; background: #f8f9fa; }
    </style>
</head>
<body>
    <h1>🚀 Janus Chat Server</h1>
    <p>OpenAI-compatible chat completion API powered by GPU-accelerated LLM inference.</p>
    
    <h2>Available Endpoints</h2>
    
    <div class="endpoint">
        <h3>POST /v1/chat/completions</h3>
        <p>Chat completion endpoint (OpenAI-compatible)</p>
        <p><strong>Example:</strong></p>
        <pre>curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "stream": false
  }'</pre>
    </div>

    <div class="endpoint">
        <h3>GET /health</h3>
        <p>Health check endpoint</p>
        <p><strong>Example:</strong> <a href="/health">GET /health</a></p>
    </div>

    <h2>Streaming</h2>
    <p>Add <code>"stream": true</code> to your request for server-sent events (SSE) streaming:</p>
    <pre>curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'</pre>
</body>
</html>"#)
}

/// Health check endpoint
async fn health(axum::extract::State(state): axum::extract::State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(json!({
        "status": "ok",
        "model": state.model_name,
    }))
}

/// Create the application router with all endpoints
pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(root))
        .route("/health", get(health))
        .route("/v1/chat/completions", post(ChatCompletionHandler::handle))
        .with_state(state)
}
