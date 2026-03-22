//! Request handlers for chat completion endpoints

use crate::models::*;
use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive},
        IntoResponse, Response, Sse,
    },
    Json,
};
use futures::stream::{self, Stream};
use janus_engine::{ChatFormatter, Model};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;

/// Shared application state
pub struct AppState {
    pub model: Arc<Mutex<Model>>,
    pub chat_formatter: ChatFormatter,
    pub model_name: String,
}

/// Handler for chat completion requests
pub struct ChatCompletionHandler;

impl ChatCompletionHandler {
    /// Handle a chat completion request (streaming or non-streaming)
    pub async fn handle(
        State(state): State<Arc<AppState>>,
        Json(request): Json<ChatCompletionRequest>,
    ) -> Response {
        // Validate request
        if request.messages.is_empty() {
            return Self::error_response(
                StatusCode::BAD_REQUEST,
                "messages array cannot be empty".to_string(),
            );
        }

        if request.n != 1 {
            return Self::error_response(
                StatusCode::BAD_REQUEST,
                "Currently only n=1 is supported".to_string(),
            );
        }

        // Format the conversation into a prompt
        let prompt = state.chat_formatter.format_chat(&request.messages);
        tracing::debug!("Formatted prompt:\n{}", prompt);

        // Determine max tokens
        let max_tokens = request.max_tokens.unwrap_or(128);

        // Get stop strings
        let mut stop_strings: Vec<String> = request.stop.clone().unwrap_or_default();
        
        // Add template-specific stop tokens
        let template_stops = state.chat_formatter.stop_tokens();
        for &stop in template_stops {
            let stop_string = stop.to_string();
            if !stop_strings.contains(&stop_string) {
                stop_strings.push(stop_string);
            }
        }

        // Handle streaming vs non-streaming
        if request.stream {
            Self::handle_streaming(state, prompt, max_tokens, stop_strings).await
        } else {
            Self::handle_non_streaming(state, prompt, max_tokens, stop_strings).await
        }
    }

    /// Handle non-streaming completion
    async fn handle_non_streaming(
        state: Arc<AppState>,
        prompt: String,
        max_tokens: usize,
        stop_strings: Vec<String>,
    ) -> Response {
        let mut model = state.model.lock().await;

        // Generate text
        let generated = match model
            .generate_with_callback(
                &prompt,
                max_tokens,
                Some(&stop_strings),
                None::<fn(&str) -> bool>,
            )
            .await
        {
            Ok(text) => text,
            Err(e) => {
                return Self::error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Generation failed: {}", e),
                );
            }
        };

        // Remove any stop strings from the output
        let mut content = generated.clone();
        for stop_str in &stop_strings {
            if let Some(pos) = content.find(stop_str) {
                content.truncate(pos);
            }
        }

        // Create response
        let response = ChatCompletionResponse {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            model: state.model_name.clone(),
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ResponseMessage {
                    role: "assistant".to_string(),
                    content: content.trim().to_string(),
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Usage {
                prompt_tokens: 0, // TODO: Calculate from tokenizer
                completion_tokens: 0, // TODO: Track actual tokens
                total_tokens: 0,
            },
        };

        Json(response).into_response()
    }

    /// Handle streaming completion with SSE
    async fn handle_streaming(
        state: Arc<AppState>,
        prompt: String,
        max_tokens: usize,
        stop_strings: Vec<String>,
    ) -> Response {
        // Create a channel for streaming tokens
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<String>();

        // Spawn generation task
        let state_clone = state.clone();
        tokio::spawn(async move {
            let mut model = state_clone.model.lock().await;

            let callback = |text: &str| -> bool {
                // Send the text chunk through the channel
                if tx.send(text.to_string()).is_err() {
                    tracing::warn!("Client disconnected");
                    return false;
                }
                true
            };

            if let Err(e) = model
                .generate_with_callback(
                    &prompt,
                    max_tokens,
                    Some(&stop_strings),
                    Some(callback),
                )
                .await
            {
                tracing::error!("Generation error: {}", e);
            }
        });

        // Create SSE stream
        let stream = Self::create_sse_stream(rx, state.model_name.clone());

        Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    }

    /// Create Server-Sent Events stream from token channel
    fn create_sse_stream(
        rx: tokio::sync::mpsc::UnboundedReceiver<String>,
        model_name: String,
    ) -> impl Stream<Item = Result<Event, Infallible>> {
        let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        stream::unfold(
            (rx, id, created, model_name, false, false),
            |(mut rx, id, created, model_name, mut sent_role, mut finished)| async move {
                if finished {
                    return None;
                }

                // Send role in first chunk
                if !sent_role {
                    sent_role = true;
                    let chunk = ChatCompletionChunk {
                        id: id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model_name.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: Delta {
                                role: Some("assistant".to_string()),
                                content: None,
                            },
                            finish_reason: None,
                        }],
                    };

                    let data = match serde_json::to_string(&chunk) {
                        Ok(s) => s,
                        Err(e) => {
                            tracing::error!("Failed to serialize SSE chunk: {}", e);
                            return None;
                        }
                    };
                    let event = Event::default().data(data);
                    return Some((
                        Ok(event),
                        (rx, id, created, model_name, sent_role, finished),
                    ));
                }

                // Receive next token
                match rx.recv().await {
                    Some(content) => {
                        let chunk = ChatCompletionChunk {
                            id: id.clone(),
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model: model_name.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: Delta {
                                    role: None,
                                    content: Some(content),
                                },
                                finish_reason: None,
                            }],
                        };

                        let data = match serde_json::to_string(&chunk) {
                            Ok(s) => s,
                            Err(e) => {
                                tracing::error!("Failed to serialize SSE chunk: {}", e);
                                return None;
                            }
                        };
                        let event = Event::default().data(data);
                        Some((Ok(event), (rx, id, created, model_name, sent_role, finished)))
                    }
                    None => {
                        // Send final chunk with finish_reason
                        finished = true;
                        let chunk = ChatCompletionChunk {
                            id: id.clone(),
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model: model_name.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: Delta {
                                    role: None,
                                    content: None,
                                },
                                finish_reason: Some("stop".to_string()),
                            }],
                        };

                        let data = match serde_json::to_string(&chunk) {
                            Ok(s) => s,
                            Err(e) => {
                                tracing::error!("Failed to serialize final SSE chunk: {}", e);
                                return None;
                            }
                        };
                        let event = Event::default().data(data);
                        Some((Ok(event), (rx, id, created, model_name, sent_role, finished)))
                    }
                }
            },
        )
    }

    /// Create an error response
    fn error_response(status: StatusCode, message: String) -> Response {
        let error = ErrorResponse::new(message);
        (status, Json(error)).into_response()
    }
}
