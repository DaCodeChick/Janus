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
use janus_engine::{ChatFormatter, ChatMessage, ChatRole, Model};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;

enum StreamEvent {
    Chunk(String),
    Done(String),
}

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
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("📨 Received chat completion request");
        println!("   Messages: {}", request.messages.len());
        println!("   Streaming: {}", request.stream);
        println!("   Max tokens: {:?}", request.max_tokens);
        
        // Validate request
        if request.messages.is_empty() {
            println!("❌ Validation failed: messages array is empty");
            return Self::error_response(
                StatusCode::BAD_REQUEST,
                "messages array cannot be empty".to_string(),
            );
        }

        if request.n != 1 {
            println!("❌ Validation failed: n={} (only n=1 supported)", request.n);
            return Self::error_response(
                StatusCode::BAD_REQUEST,
                "Currently only n=1 is supported".to_string(),
            );
        }

        // Ensure a stable system anchor exists for chat templates.
        let mut messages = request.messages.clone();
        if !messages.iter().any(|msg| msg.role == ChatRole::System) {
            messages.insert(
                0,
                ChatMessage {
                    role: ChatRole::System,
                    content: "You are a helpful AI assistant.".to_string(),
                },
            );
            println!("🧭 Injected default system message");
        }

        // Format the conversation into a prompt
        println!("🔄 Formatting conversation...");
        let prompt = state.chat_formatter.format_chat(&messages);
        println!("📝 Formatted prompt ({} chars)", prompt.len());
        tracing::debug!("Formatted prompt:\n{}", prompt);

        // Determine max tokens
        let max_tokens = request.max_tokens.unwrap_or(128);
        println!("🎯 Max tokens: {}", max_tokens);

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
        println!("🛑 Stop strings: {:?}", stop_strings);

        // Handle streaming vs non-streaming
        if request.stream {
            println!("🌊 Starting streaming generation...");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            Self::handle_streaming(state, prompt, max_tokens, stop_strings).await
        } else {
            println!("📦 Starting non-streaming generation...");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
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
        println!("🔒 Acquiring model lock...");
        let mut model = state.model.lock().await;
        println!("✅ Model lock acquired");

        // Generate text
        println!("🤖 Starting generation...");
        let generation = match model
            .generate_with_callback(
                &prompt,
                max_tokens,
                Some(&stop_strings),
                None::<fn(&str) -> bool>,
            )
            .await
        {
            Ok(result) => {
                println!("✅ Generation complete ({} chars)", result.text.len());
                result
            }
            Err(e) => {
                println!("❌ Generation failed: {}", e);
                return Self::error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Generation failed: {}", e),
                );
            }
        };

        // Remove any stop strings from the output
        let mut content = generation.text.clone();
        for stop_str in &stop_strings {
            if let Some(pos) = content.find(stop_str) {
                content.truncate(pos);
            }
        }

        println!("📤 Sending response ({} chars after stop removal)", content.trim().len());

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
                finish_reason: Some(generation.finish_reason.clone()),
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
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<StreamEvent>();

        // Spawn generation task
        let state_clone = state.clone();
        tokio::spawn(async move {
            println!("🔒 [Stream] Acquiring model lock...");
            let mut model = state_clone.model.lock().await;
            println!("✅ [Stream] Model lock acquired");
            println!("🤖 [Stream] Starting generation...");

            let mut token_count = 0;
            let start_time = std::time::Instant::now();
            let mut last_log_time = start_time;
            
            let callback = |text: &str| -> bool {
                token_count += 1;
                let now = std::time::Instant::now();
                
                if token_count % 10 == 1 {
                    let elapsed = now.duration_since(start_time).as_secs_f64();
                    let tokens_per_sec = token_count as f64 / elapsed;
                    let recent_elapsed = now.duration_since(last_log_time).as_secs_f64();
                    let recent_tps = 10.0 / recent_elapsed;
                    
                    println!("📝 [Stream] {} tokens | {:.1} tok/s overall | {:.1} tok/s recent", 
                             token_count, tokens_per_sec, recent_tps);
                    last_log_time = now;
                }
                
                // Send the text chunk through the channel
                if tx.send(StreamEvent::Chunk(text.to_string())).is_err() {
                    println!("⚠️  [Stream] Client disconnected");
                    return false;
                }
                true
            };

            match model
                .generate_with_callback(
                    &prompt,
                    max_tokens,
                    Some(&stop_strings),
                    Some(callback),
                )
                .await
            {
                Ok(result) => {
                    let elapsed = start_time.elapsed().as_secs_f64();
                    let tokens_per_sec = token_count as f64 / elapsed;
                    println!(
                        "✅ [Stream] Generation complete: {} tokens in {:.1}s ({:.1} tok/s), finish_reason={}",
                        token_count,
                        elapsed,
                        tokens_per_sec,
                        result.finish_reason
                    );
                    let _ = tx.send(StreamEvent::Done(result.finish_reason));
                }
                Err(e) => {
                    println!("❌ [Stream] Generation error: {}", e);
                    let _ = tx.send(StreamEvent::Done("stop".to_string()));
                }
            }
        });

        // Create SSE stream
        println!("🌊 Creating SSE stream...");
        let stream = Self::create_sse_stream(rx, state.model_name.clone());

        Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    }

    /// Create Server-Sent Events stream from token channel
    fn create_sse_stream(
        rx: tokio::sync::mpsc::UnboundedReceiver<StreamEvent>,
        model_name: String,
    ) -> impl Stream<Item = Result<Event, Infallible>> {
        let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        stream::unfold(
            (rx, id, created, model_name, false, false, String::from("stop")),
            |(mut rx, id, created, model_name, mut sent_role, mut finished, mut finish_reason)| async move {
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
                        (rx, id, created, model_name, sent_role, finished, finish_reason),
                    ));
                }

                // Receive next stream event
                match rx.recv().await {
                    Some(StreamEvent::Chunk(content)) => {
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
                        Some((Ok(event), (rx, id, created, model_name, sent_role, finished, finish_reason)))
                    }
                    Some(StreamEvent::Done(reason)) => {
                        finish_reason = reason;

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
                                finish_reason: Some(finish_reason.clone()),
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
                        Some((Ok(event), (rx, id, created, model_name, sent_role, finished, finish_reason)))
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
                                finish_reason: Some(finish_reason),
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
                        Some((Ok(event), (rx, id, created, model_name, sent_role, finished, String::from("stop"))))
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
