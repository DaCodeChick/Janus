//! Plugin Development Example
//!
//! This example demonstrates how to create a custom Janus plugin that implements
//! the `JanusPlugin` trait from `janus-api`. Plugins can be dynamically loaded at
//! runtime and provide custom inference logic, preprocessing, or routing decisions.
//!
//! This example creates a "sentiment-aware" plugin that analyzes prompt sentiment
//! and adjusts generation parameters accordingly.
//!
//! ## Building a Plugin
//!
//! To build your own plugin:
//! 1. Create a new library crate with `crate-type = ["cdylib"]`
//! 2. Add dependencies: `janus-api`, `janus-engine`, `abi_stable`
//! 3. Implement the `JanusPlugin` trait
//! 4. Export the plugin creation function
//!
//! ## Usage
//!
//! This is a library example showing plugin structure. To use it:
//!
//! ```bash
//! # Build the plugin as a shared library
//! cargo build --example plugin_development --release
//!
//! # Load it in your application
//! let plugin = load_plugin("target/release/libplugin_development.so")?;
//! ```

use janus_api::{
    JanusPlugin, JanusPlugin_TO, PluginInfo, PluginErrorCode, PromptContext,
    RoutingPreference, StreamCallback, ResponseChunk, PluginCapabilities,
};
use abi_stable::{
    sabi_extern_fn,
    sabi_trait::TD_Opaque,
    std_types::{RResult, RStr, RString, RBox},
};
use std::sync::{Arc, Mutex};

/// Sentiment-aware plugin implementation
///
/// This plugin analyzes prompt sentiment and adjusts generation parameters
/// to produce more contextually appropriate responses.
pub struct SentimentPlugin {
    /// Plugin configuration
    config: PluginConfig,
    
    /// Internal state (if needed)
    state: Arc<Mutex<PluginState>>,
}

#[derive(Debug, Clone)]
struct PluginConfig {
    /// Temperature adjustment based on sentiment
    temperature_boost: f32,
    
    /// Whether to enable sentiment analysis
    enabled: bool,
    
    /// Maximum tokens to generate
    max_tokens: usize,
}

impl Default for PluginConfig {
    fn default() -> Self {
        Self {
            temperature_boost: 0.2,
            enabled: true,
            max_tokens: 200,
        }
    }
}

struct PluginState {
    /// Number of requests processed
    request_count: u64,
    
    /// Average sentiment scores
    avg_sentiment: f32,
}

impl Default for PluginState {
    fn default() -> Self {
        Self {
            request_count: 0,
            avg_sentiment: 0.0,
        }
    }
}

/// Simple sentiment analysis (in a real plugin, use a proper NLP library)
fn analyze_sentiment(text: &str) -> f32 {
    // Positive words
    let positive_words = [
        "happy", "good", "great", "excellent", "wonderful", "amazing",
        "love", "enjoy", "fantastic", "brilliant", "positive",
    ];
    
    // Negative words
    let negative_words = [
        "sad", "bad", "terrible", "awful", "hate", "horrible",
        "angry", "disappointed", "negative", "worst", "poor",
    ];
    
    let text_lower = text.to_lowercase();
    
    let positive_count = positive_words.iter()
        .filter(|word| text_lower.contains(*word))
        .count() as f32;
    
    let negative_count = negative_words.iter()
        .filter(|word| text_lower.contains(*word))
        .count() as f32;
    
    // Return sentiment score (-1.0 to 1.0)
    let total = positive_count + negative_count;
    if total == 0.0 {
        0.0 // Neutral
    } else {
        (positive_count - negative_count) / total
    }
}

impl JanusPlugin for SentimentPlugin {
    fn init(&mut self, config_json: RStr<'_>) -> RResult<(), PluginErrorCode> {
        // Parse configuration JSON
        match serde_json::from_str::<serde_json::Value>(config_json.as_str()) {
            Ok(config_value) => {
                // Extract configuration values
                if let Some(temp_boost) = config_value.get("temperature_boost").and_then(|v| v.as_f64()) {
                    self.config.temperature_boost = temp_boost as f32;
                }
                
                if let Some(enabled) = config_value.get("enabled").and_then(|v| v.as_bool()) {
                    self.config.enabled = enabled;
                }
                
                if let Some(max_tokens) = config_value.get("max_tokens").and_then(|v| v.as_u64()) {
                    self.config.max_tokens = max_tokens as usize;
                }
                
                tracing::info!("SentimentPlugin initialized with config: {:?}", self.config);
                RResult::ROk(())
            }
            Err(e) => {
                tracing::error!("Failed to parse config JSON: {}", e);
                RResult::RErr(PluginErrorCode::InvalidInput)
            }
        }
    }

    fn info(&self) -> PluginInfo {
        PluginInfo {
            name: RStr::from("sentiment-aware-plugin"),
            version: RStr::from("1.0.0"),
            author: RStr::from("Janus Team"),
            description: RStr::from("Adjusts generation parameters based on prompt sentiment"),
            capabilities: PluginCapabilities::STREAMING,
            max_context_tokens: 4096,
            preferred_batch_size: 1,
        }
    }

    fn analyze(&self, context: &PromptContext) -> RoutingPreference {
        if !self.config.enabled {
            return RoutingPreference::NoPreference;
        }
        
        // Analyze sentiment
        let sentiment = analyze_sentiment(context.prompt.as_str());
        
        // Update state
        if let Ok(mut state) = self.state.lock() {
            state.request_count += 1;
            state.avg_sentiment = (state.avg_sentiment * (state.request_count - 1) as f32 + sentiment)
                / state.request_count as f32;
        }
        
        tracing::debug!("Prompt sentiment: {:.2}", sentiment);
        
        // Route based on sentiment complexity
        // Highly emotional prompts (positive or negative) might benefit from cloud models
        if sentiment.abs() > 0.7 {
            RoutingPreference::Cloud
        } else {
            RoutingPreference::Local
        }
    }

    fn infer_stream(
        &mut self,
        context: &PromptContext,
        callback: StreamCallback,
    ) -> RResult<(), PluginErrorCode> {
        // Analyze sentiment and adjust temperature
        let sentiment = analyze_sentiment(context.prompt.as_str());
        let adjusted_temperature = context.temperature + (sentiment.abs() * self.config.temperature_boost);
        
        tracing::info!(
            "Streaming inference with sentiment={:.2}, adjusted_temperature={:.2}",
            sentiment,
            adjusted_temperature
        );
        
        // In a real plugin, you would:
        // 1. Load your model or connect to an API
        // 2. Generate tokens with the adjusted parameters
        // 3. Call the callback for each token
        //
        // For this example, we'll simulate token generation
        
        let tokens = vec![
            "This", " is", " a", " simulated", " response", " with",
            " adjusted", " parameters", " based", " on", " sentiment", "."
        ];
        
        for (i, token) in tokens.iter().enumerate() {
            let chunk = ResponseChunk {
                data: RStr::from(*token),
                is_final: i == tokens.len() - 1,
                token_count: (i + 1) as u32,
            };
            
            // Call the streaming callback
            let result = callback.call(&chunk);
            if result != PluginErrorCode::Success {
                tracing::error!("Streaming callback error: {:?}", result);
                return RResult::RErr(PluginErrorCode::InferenceFailed);
            }
            
            // Simulate token generation delay
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        
        RResult::ROk(())
    }

    fn infer_blocking(
        &mut self,
        context: &PromptContext,
    ) -> RResult<RStr<'static>, PluginErrorCode> {
        // Analyze sentiment
        let sentiment = analyze_sentiment(context.prompt.as_str());
        
        tracing::info!(
            "Blocking inference with sentiment={:.2}",
            sentiment
        );
        
        // In a real plugin, generate the full response here
        // For this example, return a simulated response
        let response = format!(
            "This is a simulated response (sentiment: {:.2}). \
             In a real plugin, this would be generated text based on your prompt.",
            sentiment
        );
        
        // Convert to RString (owned) then to RStr<'static>
        let response_rstring = RString::from(response);
        let response_leaked: &'static str = Box::leak(response_rstring.as_str().to_string().into_boxed_str());
        
        RResult::ROk(RStr::from(response_leaked))
    }

    fn shutdown(&mut self) {
        tracing::info!("SentimentPlugin shutting down");
        
        if let Ok(state) = self.state.lock() {
            tracing::info!(
                "Processed {} requests with average sentiment {:.2}",
                state.request_count,
                state.avg_sentiment
            );
        }
    }
}

impl SentimentPlugin {
    /// Create a new SentimentPlugin instance
    pub fn new() -> Self {
        Self {
            config: PluginConfig::default(),
            state: Arc::new(Mutex::new(PluginState::default())),
        }
    }
}

impl Default for SentimentPlugin {
    fn default() -> Self {
        Self::new()
    }
}

/// Plugin entry point - this function is called when the plugin is loaded
///
/// This must be exported with a specific name for the plugin system to find it.
#[sabi_extern_fn]
pub fn create_plugin() -> JanusPlugin_TO<'static, RBox<()>> {
    let plugin = SentimentPlugin::new();
    JanusPlugin_TO::from_value(plugin, TD_Opaque)
}

// ============================================================================
// Example usage code (this would be in your application, not the plugin)
// ============================================================================

/// Example: Using the plugin in your application
///
/// ```no_run
/// use janus_api::{JanusPlugin, PromptContext};
/// 
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Load the plugin (actual loading mechanism depends on your setup)
///     let mut plugin = create_plugin();
///     
///     // Initialize with config
///     let config = r#"{"temperature_boost": 0.3, "enabled": true}"#;
///     plugin.init(config.into())?;
///     
///     // Get plugin info
///     let info = plugin.info();
///     println!("Loaded plugin: {} v{}", info.name, info.version);
///     
///     // Create a prompt context
///     let context = PromptContext {
///         prompt: "I'm feeling great today!".into(),
///         temperature: 0.7,
///         max_tokens: 100,
///         ..Default::default()
///     };
///     
///     // Analyze routing preference
///     let preference = plugin.analyze(&context);
///     println!("Routing preference: {:?}", preference);
///     
///     // Generate response (blocking)
///     let response = plugin.infer_blocking(&context)?;
///     println!("Response: {}", response);
///     
///     // Shutdown
///     plugin.shutdown();
///     
///     Ok(())
/// }
/// ```

// ============================================================================
// Additional helper functions for plugin developers
// ============================================================================

/// Helper: Parse generation parameters from JSON
pub fn parse_generation_params(json: &str) -> Result<GenerationParams, serde_json::Error> {
    serde_json::from_str(json)
}

#[derive(Debug, serde::Deserialize)]
pub struct GenerationParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub max_tokens: usize,
}

/// Helper: Create a simple streaming callback that prints tokens
/// 
/// Note: StreamCallback requires an `extern "C"` function pointer and cannot
/// capture variables. For collecting tokens, you would need to use global state
/// or thread-local storage.
pub extern "C" fn simple_print_callback(chunk: &ResponseChunk) -> PluginErrorCode {
    print!("{}", chunk.data.as_str());
    if chunk.is_final {
        println!();
    }
    PluginErrorCode::Success
}

/// Helper: Create a streaming callback that returns success
pub fn create_simple_callback() -> StreamCallback {
    StreamCallback::new(simple_print_callback)
}

// ============================================================================
// Testing utilities
// ============================================================================

/// Example main function demonstrating plugin usage
fn main() {
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    println!("=== Janus Plugin Development Example ===\n");
    
    // Create plugin instance
    let mut plugin = SentimentPlugin::new();
    println!("Created SentimentPlugin");
    
    // Initialize with config
    let config = r#"{"temperature_boost": 0.3, "enabled": true, "max_tokens": 200}"#;
    match plugin.init(RStr::from(config)) {
        RResult::ROk(_) => println!("Plugin initialized with config"),
        RResult::RErr(e) => {
            eprintln!("Failed to initialize plugin: {:?}", e);
            return;
        }
    }
    
    // Get plugin info
    let info = plugin.info();
    println!("\nPlugin Info:");
    println!("  Name: {}", info.name);
    println!("  Version: {}", info.version);
    println!("  Author: {}", info.author);
    println!("  Description: {}", info.description);
    println!("  Max context tokens: {}", info.max_context_tokens);
    println!("  Preferred batch size: {}", info.preferred_batch_size);
    
    // Test sentiment analysis
    let test_prompts = [
        "I'm feeling absolutely wonderful today!",
        "This is the worst day ever",
        "The weather seems normal",
    ];
    
    println!("\nTesting sentiment analysis:");
    for prompt in &test_prompts {
        let sentiment = analyze_sentiment(prompt);
        println!("  '{}' -> sentiment: {:.2}", prompt, sentiment);
    }
    
    // Create a prompt context
    let context = PromptContext {
        prompt: RStr::from("I'm feeling great today! Tell me something positive."),
        system_prompt: RStr::from("You are a helpful assistant."),
        max_tokens: 100,
        temperature: 0.7,
        top_p: 0.9,
        context_size: 2048,
        metadata_json: RStr::from("{}"),
    };
    
    // Analyze routing preference
    let preference = plugin.analyze(&context);
    println!("\nRouting preference: {:?}", preference);
    
    // Test blocking inference
    println!("\nTesting blocking inference...");
    match plugin.infer_blocking(&context) {
        RResult::ROk(response) => {
            println!("Response: {}", response);
        }
        RResult::RErr(e) => {
            eprintln!("Inference failed: {:?}", e);
        }
    }
    
    // Test streaming inference
    println!("\nTesting streaming inference...");
    print!("Streamed output: ");
    let callback = create_simple_callback();
    match plugin.infer_stream(&context, callback) {
        RResult::ROk(_) => {
            println!("\nStreaming completed successfully");
        }
        RResult::RErr(e) => {
            eprintln!("\nStreaming failed: {:?}", e);
        }
    }
    
    // Shutdown
    plugin.shutdown();
    println!("\nPlugin shut down");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sentiment_analysis() {
        assert!(analyze_sentiment("I love this!") > 0.5);
        assert!(analyze_sentiment("This is terrible") < -0.5);
        assert!(analyze_sentiment("The weather is normal").abs() < 0.1);
    }
    
    #[test]
    fn test_plugin_info() {
        let plugin = SentimentPlugin::new();
        let info = plugin.info();
        assert_eq!(info.name.as_str(), "sentiment-aware-plugin");
        assert_eq!(info.version.as_str(), "1.0.0");
    }
    
    #[tokio::test]
    async fn test_plugin_initialization() {
        let mut plugin = SentimentPlugin::new();
        let config = r#"{"temperature_boost": 0.5, "enabled": true}"#;
        let result = plugin.init(RStr::from(config));
        assert!(matches!(result, RResult::ROk(_)));
        assert_eq!(plugin.config.temperature_boost, 0.5);
    }
}
