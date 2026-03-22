//! OpenAI-compatible API models and types

use janus_engine::ChatMessage;
use serde::{Deserialize, Serialize};

/// Chat completion request (OpenAI-compatible)
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    /// ID of the model to use
    pub model: String,

    /// Messages in the conversation
    pub messages: Vec<ChatMessage>,

    /// Maximum number of tokens to generate
    #[serde(default = "default_max_tokens")]
    pub max_tokens: Option<usize>,

    /// Sampling temperature (0.0 to 2.0)
    #[serde(default = "default_temperature")]
    pub temperature: Option<f32>,

    /// Top-p (nucleus) sampling parameter
    #[serde(default)]
    pub top_p: Option<f32>,

    /// Top-k sampling parameter
    #[serde(default)]
    pub top_k: Option<usize>,

    /// Whether to stream responses using SSE
    #[serde(default)]
    pub stream: bool,

    /// Stop sequences
    #[serde(default)]
    pub stop: Option<Vec<String>>,

    /// Number of chat completion choices to generate
    #[serde(default = "default_n")]
    pub n: usize,

    /// Presence penalty (-2.0 to 2.0)
    #[serde(default)]
    pub presence_penalty: Option<f32>,

    /// Frequency penalty (-2.0 to 2.0)
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
}

fn default_max_tokens() -> Option<usize> {
    Some(128)
}

fn default_temperature() -> Option<f32> {
    Some(0.7)
}

fn default_n() -> usize {
    1
}

/// Chat completion response (OpenAI-compatible)
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: Usage,
}

/// A single chat completion choice
#[derive(Debug, Serialize)]
pub struct ChatCompletionChoice {
    pub index: usize,
    pub message: ResponseMessage,
    pub finish_reason: Option<String>,
}

/// Response message
#[derive(Debug, Serialize)]
pub struct ResponseMessage {
    pub role: String,
    pub content: String,
}

/// Token usage statistics
#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Streaming chat completion chunk (OpenAI-compatible)
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

/// A single streaming choice
#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

/// Delta content in streaming response
#[derive(Debug, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// Error response (OpenAI-compatible)
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

/// Error detail
#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    pub r#type: String,
    pub code: Option<String>,
}

impl ErrorResponse {
    pub fn new(message: String) -> Self {
        Self {
            error: ErrorDetail {
                message,
                r#type: "server_error".to_string(),
                code: None,
            },
        }
    }
}
