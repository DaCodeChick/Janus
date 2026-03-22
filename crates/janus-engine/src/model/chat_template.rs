//! Chat template formatting for converting OpenAI-style messages to model-specific formats
//!
//! This module provides chat template formatters that convert structured conversation
//! messages (system, user, assistant) into the specific prompt format expected by
//! different LLM architectures.

use serde::{Deserialize, Serialize};
use std::fmt;

/// A single message in a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role of the message sender
    pub role: ChatRole,
    /// Content of the message
    pub content: String,
}

/// Role of a message sender
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    /// System prompt that sets context/instructions
    System,
    /// User message (human input)
    User,
    /// Assistant message (model output)
    Assistant,
}

impl fmt::Display for ChatRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChatRole::System => write!(f, "system"),
            ChatRole::User => write!(f, "user"),
            ChatRole::Assistant => write!(f, "assistant"),
        }
    }
}

/// Chat template format used by different model families
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplateFormat {
    /// ChatML format used by models like Mistral-Instruct, Hermes, etc.
    /// Format: `<|im_start|>role\ncontent<|im_end|>`
    ChatML,

    /// Llama 3 format
    /// Format: `<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>`
    Llama3,

    /// Llama 2 format
    /// Format: `[INST] <<SYS>>system<</SYS>>user [/INST] assistant`
    Llama2,

    /// Alpaca format
    /// Format: `### Instruction:\nuser\n\n### Response:\nassistant`
    Alpaca,

    /// Vicuna format
    /// Format: `USER: content\nASSISTANT: content`
    Vicuna,

    /// Zephyr format (similar to ChatML but with different tokens)
    /// Format: `<|system|>\ncontent</s>\n<|user|>\ncontent</s>\n<|assistant|>\n`
    Zephyr,
}

impl ChatTemplateFormat {
    /// Detect the appropriate chat template format from model name or config
    ///
    /// This uses heuristics based on common model naming patterns.
    /// If detection fails, defaults to ChatML as it's widely supported.
    pub fn detect_from_model_name(model_name: &str) -> Self {
        let model_lower = model_name.to_lowercase();

        if model_lower.contains("llama-3") || model_lower.contains("llama3") {
            Self::Llama3
        } else if model_lower.contains("llama-2") || model_lower.contains("llama2") {
            Self::Llama2
        } else if model_lower.contains("mistral")
            || model_lower.contains("hermes")
            || model_lower.contains("openchat")
        {
            Self::ChatML
        } else if model_lower.contains("alpaca") {
            Self::Alpaca
        } else if model_lower.contains("vicuna") {
            Self::Vicuna
        } else if model_lower.contains("zephyr") || model_lower.contains("tinyllama") {
            Self::Zephyr
        } else {
            // Default to ChatML as it's widely supported
            tracing::warn!(
                "Could not detect chat template from model name '{}', defaulting to ChatML",
                model_name
            );
            Self::ChatML
        }
    }

    /// Get the stop tokens for this template format
    pub fn stop_tokens(&self) -> Vec<String> {
        match self {
            Self::ChatML => vec!["<|im_end|>".to_string()],
            Self::Llama3 => vec!["<|eot_id|>".to_string(), "<|end_of_text|>".to_string()],
            Self::Llama2 => vec!["</s>".to_string()],
            Self::Alpaca => vec!["###".to_string()],
            Self::Vicuna => vec!["USER:".to_string()],
            Self::Zephyr => vec!["</s>".to_string()],
        }
    }
}

/// Chat template formatter
pub struct ChatFormatter {
    format: ChatTemplateFormat,
}

impl ChatFormatter {
    /// Create a new chat formatter with the specified format
    pub fn new(format: ChatTemplateFormat) -> Self {
        Self { format }
    }

    /// Create a formatter by detecting the format from a model name
    pub fn from_model_name(model_name: &str) -> Self {
        let format = ChatTemplateFormat::detect_from_model_name(model_name);
        tracing::info!("Detected chat template format: {:?}", format);
        Self::new(format)
    }

    /// Format a conversation into a prompt string
    ///
    /// # Arguments
    /// * `messages` - Array of chat messages in conversation order
    ///
    /// # Returns
    /// A formatted prompt string ready for tokenization
    pub fn format_chat(&self, messages: &[ChatMessage]) -> String {
        match self.format {
            ChatTemplateFormat::ChatML => self.format_chatml(messages),
            ChatTemplateFormat::Llama3 => self.format_llama3(messages),
            ChatTemplateFormat::Llama2 => self.format_llama2(messages),
            ChatTemplateFormat::Alpaca => self.format_alpaca(messages),
            ChatTemplateFormat::Vicuna => self.format_vicuna(messages),
            ChatTemplateFormat::Zephyr => self.format_zephyr(messages),
        }
    }

    /// Get the stop tokens for this formatter
    pub fn stop_tokens(&self) -> Vec<String> {
        self.format.stop_tokens()
    }

    /// Get the template format being used
    pub fn format(&self) -> ChatTemplateFormat {
        self.format
    }

    // Format implementations for each template

    fn format_chatml(&self, messages: &[ChatMessage]) -> String {
        let mut prompt = String::new();

        for msg in messages {
            prompt.push_str(&format!(
                "<|im_start|>{}\n{}<|im_end|>\n",
                msg.role, msg.content
            ));
        }

        // Add the assistant prefix to prompt the model to respond
        prompt.push_str("<|im_start|>assistant\n");

        prompt
    }

    fn format_llama3(&self, messages: &[ChatMessage]) -> String {
        let mut prompt = String::new();

        for msg in messages {
            prompt.push_str(&format!(
                "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                msg.role, msg.content
            ));
        }

        // Add the assistant prefix
        prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");

        prompt
    }

    fn format_llama2(&self, messages: &[ChatMessage]) -> String {
        let mut prompt = String::new();
        let mut system_prompt = String::new();
        let mut conversation = Vec::new();

        // Extract system message and conversation turns
        for msg in messages {
            match msg.role {
                ChatRole::System => {
                    system_prompt = msg.content.clone();
                }
                _ => {
                    conversation.push(msg.clone());
                }
            }
        }

        // Format as Llama 2 chat format
        let mut in_user_turn = true;
        for msg in conversation {
            match msg.role {
                ChatRole::User => {
                    if in_user_turn {
                        prompt.push_str("[INST] ");
                        if !system_prompt.is_empty() {
                            prompt.push_str(&format!("<<SYS>>\n{}\n<</SYS>>\n\n", system_prompt));
                            system_prompt.clear(); // Only include system prompt once
                        }
                        prompt.push_str(&msg.content);
                        prompt.push_str(" [/INST] ");
                        in_user_turn = false;
                    } else {
                        prompt.push_str(&format!("[INST] {} [/INST] ", msg.content));
                    }
                }
                ChatRole::Assistant => {
                    prompt.push_str(&msg.content);
                    prompt.push(' ');
                    in_user_turn = true;
                }
                ChatRole::System => {
                    // Already handled
                }
            }
        }

        prompt
    }

    fn format_alpaca(&self, messages: &[ChatMessage]) -> String {
        let mut prompt = String::new();

        // Alpaca format typically uses the last user message as instruction
        for msg in messages {
            match msg.role {
                ChatRole::System => {
                    prompt.push_str(&format!("### System:\n{}\n\n", msg.content));
                }
                ChatRole::User => {
                    prompt.push_str(&format!("### Instruction:\n{}\n\n", msg.content));
                }
                ChatRole::Assistant => {
                    prompt.push_str(&format!("### Response:\n{}\n\n", msg.content));
                }
            }
        }

        // Add response prefix
        prompt.push_str("### Response:\n");

        prompt
    }

    fn format_vicuna(&self, messages: &[ChatMessage]) -> String {
        let mut prompt = String::new();

        for msg in messages {
            match msg.role {
                ChatRole::System => {
                    prompt.push_str(&format!("SYSTEM: {}\n", msg.content));
                }
                ChatRole::User => {
                    prompt.push_str(&format!("USER: {}\n", msg.content));
                }
                ChatRole::Assistant => {
                    prompt.push_str(&format!("ASSISTANT: {}\n", msg.content));
                }
            }
        }

        // Add assistant prefix
        prompt.push_str("ASSISTANT: ");

        prompt
    }

    fn format_zephyr(&self, messages: &[ChatMessage]) -> String {
        let mut prompt = String::new();

        for msg in messages {
            prompt.push_str(&format!("<|{}|>\n{}</s>\n", msg.role, msg.content));
        }

        // Add the assistant prefix
        prompt.push_str("<|assistant|>\n");

        prompt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chatml_format() {
        let formatter = ChatFormatter::new(ChatTemplateFormat::ChatML);
        let messages = vec![
            ChatMessage {
                role: ChatRole::System,
                content: "You are a helpful assistant.".to_string(),
            },
            ChatMessage {
                role: ChatRole::User,
                content: "Hello!".to_string(),
            },
        ];

        let prompt = formatter.format_chat(&messages);
        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("<|im_start|>user"));
        assert!(prompt.contains("<|im_start|>assistant"));
    }

    #[test]
    fn test_format_detection() {
        assert_eq!(
            ChatTemplateFormat::detect_from_model_name("Llama-3-8B-Instruct"),
            ChatTemplateFormat::Llama3
        );
        assert_eq!(
            ChatTemplateFormat::detect_from_model_name("Mistral-7B-Instruct"),
            ChatTemplateFormat::ChatML
        );
        assert_eq!(
            ChatTemplateFormat::detect_from_model_name("TinyLlama-1.1B-Chat"),
            ChatTemplateFormat::Zephyr
        );
    }

    #[test]
    fn test_stop_tokens() {
        let formatter = ChatFormatter::new(ChatTemplateFormat::ChatML);
        let stop_tokens = formatter.stop_tokens();
        assert_eq!(stop_tokens, vec!["<|im_end|>"]);

        let formatter = ChatFormatter::new(ChatTemplateFormat::Llama3);
        let stop_tokens = formatter.stop_tokens();
        assert!(stop_tokens.contains(&"<|eot_id|>".to_string()));
    }
}
