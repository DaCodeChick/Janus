//! Streaming inference example
//!
//! This example demonstrates how to use the Janus Engine for text generation
//! with configuration suitable for interactive streaming applications.
//!
//! Note: The current public API uses `generate()` which returns complete responses.
//! For true token-by-token streaming, you would need to use internal APIs or
//! extend the public API to expose per-token generation.
//!
//! Usage (directory mode):
//!   cargo run --example streaming --release <model_dir> "<prompt>"
//!
//! Usage (file mode):
//!   cargo run --example streaming --release <model.gguf|model.safetensors> <config.json> <tokenizer.json> "<prompt>"
//!
//! Directory mode expects model_dir to contain:
//!   - model.gguf or model.safetensors (model weights)
//!   - config.json (HuggingFace config)
//!   - tokenizer.json (HuggingFace tokenizer)

use std::collections::HashMap;
use std::env;
use std::io::{self, Write};
use std::path::PathBuf;
use janus_engine::{
    ComputeEngine, GgufFile, SafetensorsFile, HuggingFaceConfig,
    Model, ModelConfig, Tokenizer, Sampler, SamplerConfig, TransformerBlock, TransformerBlockConfig
};
use janus_engine::model::block::get_tensor;

/// Build TransformerBlock from tensor map
fn build_transformer_block(
    config: &TransformerBlockConfig,
    layer_idx: u32,
    tensors: &HashMap<String, wgpu::Buffer>,
) -> Result<TransformerBlock, Box<dyn std::error::Error>> {
    let q = get_tensor(
        tensors,
        &format!("model.layers.{}.self_attn.q_proj.weight", layer_idx),
        &format!("blk.{}.attn_q.weight", layer_idx),
    )?;
    let k = get_tensor(
        tensors,
        &format!("model.layers.{}.self_attn.k_proj.weight", layer_idx),
        &format!("blk.{}.attn_k.weight", layer_idx),
    )?;
    let v = get_tensor(
        tensors,
        &format!("model.layers.{}.self_attn.v_proj.weight", layer_idx),
        &format!("blk.{}.attn_v.weight", layer_idx),
    )?;
    let o = get_tensor(
        tensors,
        &format!("model.layers.{}.self_attn.o_proj.weight", layer_idx),
        &format!("blk.{}.attn_output.weight", layer_idx),
    )?;
    let gate = get_tensor(
        tensors,
        &format!("model.layers.{}.mlp.gate_proj.weight", layer_idx),
        &format!("blk.{}.ffn_gate.weight", layer_idx),
    )?;
    let up = get_tensor(
        tensors,
        &format!("model.layers.{}.mlp.up_proj.weight", layer_idx),
        &format!("blk.{}.ffn_up.weight", layer_idx),
    )?;
    let down = get_tensor(
        tensors,
        &format!("model.layers.{}.mlp.down_proj.weight", layer_idx),
        &format!("blk.{}.ffn_down.weight", layer_idx),
    )?;
    let attn_norm = get_tensor(
        tensors,
        &format!("model.layers.{}.input_layernorm.weight", layer_idx),
        &format!("blk.{}.attn_norm.weight", layer_idx),
    )?;
    let ffn_norm = get_tensor(
        tensors,
        &format!("model.layers.{}.post_attention_layernorm.weight", layer_idx),
        &format!("blk.{}.ffn_norm.weight", layer_idx),
    )?;

    Ok(TransformerBlock::new(
        config.clone(),
        q.clone(),
        k.clone(),
        v.clone(),
        o.clone(),
        gate.clone(),
        up.clone(),
        down.clone(),
        attn_norm.clone(),
        ffn_norm.clone(),
    ))
}

/// Generate text with periodic progress updates
async fn generate_with_progress(
    model: &mut Model,
    prompt: &str,
    max_tokens: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    println!("Generating {} tokens...", max_tokens);
    
    let start = std::time::Instant::now();
    
    // Use the public generate API
    // Note: This doesn't provide true token-by-token streaming, but demonstrates
    // how to use the public API for generation with progress tracking
    let output = model.generate(prompt, max_tokens).await?;
    
    let elapsed = start.elapsed();
    
    println!("\n\n=== Statistics ===");
    println!("Time: {:.2}s", elapsed.as_secs_f64());
    println!("Estimated speed: {:.2} tok/s", max_tokens as f64 / elapsed.as_secs_f64());
    
    Ok(output)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();

    // Determine if we're in directory mode or file mode
    let (model_path, config_path, tokenizer_path, prompt) = if args.len() == 3 {
        // Directory mode: <model_dir> "<prompt>"
        let model_dir = PathBuf::from(&args[1]);
        
        // Auto-discover model file
        let model_file = if model_dir.join("model.gguf").exists() {
            model_dir.join("model.gguf")
        } else if model_dir.join("model.safetensors").exists() {
            model_dir.join("model.safetensors")
        } else {
            return Err("No model.gguf or model.safetensors found in directory".into());
        };
        
        (
            model_file,
            model_dir.join("config.json"),
            model_dir.join("tokenizer.json"),
            args[2].clone(),
        )
    } else if args.len() == 5 {
        // File mode: <model> <config> <tokenizer> "<prompt>"
        (
            PathBuf::from(&args[1]),
            PathBuf::from(&args[2]),
            PathBuf::from(&args[3]),
            args[4].clone(),
        )
    } else {
        eprintln!("Usage (directory mode): {} <model_dir> \"<prompt>\"", args[0]);
        eprintln!("Usage (file mode): {} <model.gguf|model.safetensors> <config.json> <tokenizer.json> \"<prompt>\"", args[0]);
        return Err("Invalid arguments".into());
    };

    println!("Loading model from {:?}", model_path);
    println!("Config: {:?}", config_path);
    println!("Tokenizer: {:?}", tokenizer_path);

    // Initialize compute engine
    let engine = ComputeEngine::new().await?;
    println!("Initialized compute engine");

    // Load HuggingFace config
    let hf_config = HuggingFaceConfig::from_file(&config_path)?;
    println!("Loaded config");

    // Convert to ModelConfig
    let model_config: ModelConfig = (&hf_config).into();

    // Load tokenizer
    let tokenizer = Tokenizer::from_file(&tokenizer_path)?;
    println!("Loaded tokenizer with vocab size: {}", tokenizer.vocab_size());

    // Load model weights based on file extension
    let tensors = if model_path.extension().and_then(|s| s.to_str()) == Some("gguf") {
        let loader = GgufFile::from_file(&model_path)?;
        engine.allocate_tensors(&loader)?
    } else {
        let loader = SafetensorsFile::from_file(&model_path)?;
        engine.allocate_tensors(&loader)?
    };

    println!("Loaded {} tensors", tensors.len());

    // Build transformer blocks
    let block_config = TransformerBlockConfig {
        batch_size: model_config.batch_size,
        hidden_dim: model_config.hidden_dim,
        num_heads: model_config.num_heads,
        num_kv_heads: model_config.num_kv_heads,
        head_dim: model_config.head_dim,
        ffn_dim: model_config.ffn_dim,
        rms_norm_eps: model_config.rms_norm_eps,
    };

    let mut blocks = Vec::new();
    for layer_idx in 0..model_config.num_layers {
        let block = build_transformer_block(&block_config, layer_idx, &tensors)?;
        blocks.push(block);
    }

    println!("Built {} transformer blocks", blocks.len());

    // Get embedding and output tensors
    let token_embedding_table =
        get_tensor(&tensors, "model.embed_tokens.weight", "token_embd.weight")?.clone();

    let output_norm_weight =
        get_tensor(&tensors, "model.norm.weight", "output_norm.weight")?.clone();

    let lm_head_weight = get_tensor(&tensors, "lm_head.weight", "output.weight")?.clone();

    // Create sampler with reasonable defaults for streaming/interactive use
    let sampler_config = SamplerConfig {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        repetition_penalty: 1.1,
        beam_width: 1, // Greedy decoding for streaming
        max_tokens: 200,
    };
    let sampler = Sampler::new(sampler_config, model_config.vocab_size);

    // Create model
    let mut model = Model::new(
        model_config,
        engine,
        tokenizer,
        sampler,
        token_embedding_table,
        blocks,
        output_norm_weight,
        lm_head_weight,
    )?;

    println!("Model created successfully");

    println!("\n=== Text Generation ===");
    println!("Prompt: {}", prompt);
    println!("Output:");
    io::stdout().flush()?;

    // Generate text
    // Note: The current public API returns complete responses rather than streaming
    // token-by-token. For true streaming, you would need to extend the Model API.
    let output = generate_with_progress(&mut model, &prompt, 200).await?;
    
    println!("\n{}", output);

    Ok(())
}
