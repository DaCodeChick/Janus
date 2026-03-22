//! Streaming inference example
//!
//! This example demonstrates token-by-token streaming generation, which is useful
//! for building interactive applications where you want to display tokens as they're
//! generated rather than waiting for the full response.
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
    ComputeEngine, GGUFLoader, SafetensorsLoader, ModelLoader, HuggingFaceConfig,
    Model, ModelConfig, Tokenizer, Sampler, SamplerConfig, TransformerBlock, TransformerBlockConfig
};

/// Build TransformerBlock from tensor map
fn build_transformer_block(
    config: &TransformerBlockConfig,
    layer_idx: u32,
    tensors: &HashMap<String, wgpu::Buffer>,
) -> Result<TransformerBlock, Box<dyn std::error::Error>> {
    // Common tensor name patterns for LLaMA/Mistral models
    let patterns = vec![
        // LLaMA pattern
        (
            format!("model.layers.{}.self_attn.q_proj.weight", layer_idx),
            format!("model.layers.{}.self_attn.k_proj.weight", layer_idx),
            format!("model.layers.{}.self_attn.v_proj.weight", layer_idx),
            format!("model.layers.{}.self_attn.o_proj.weight", layer_idx),
            format!("model.layers.{}.mlp.gate_proj.weight", layer_idx),
            format!("model.layers.{}.mlp.up_proj.weight", layer_idx),
            format!("model.layers.{}.mlp.down_proj.weight", layer_idx),
            format!("model.layers.{}.input_layernorm.weight", layer_idx),
            format!("model.layers.{}.post_attention_layernorm.weight", layer_idx),
        ),
        // GGUF pattern (dots replaced with underscores)
        (
            format!("blk.{}.attn_q.weight", layer_idx),
            format!("blk.{}.attn_k.weight", layer_idx),
            format!("blk.{}.attn_v.weight", layer_idx),
            format!("blk.{}.attn_output.weight", layer_idx),
            format!("blk.{}.ffn_gate.weight", layer_idx),
            format!("blk.{}.ffn_up.weight", layer_idx),
            format!("blk.{}.ffn_down.weight", layer_idx),
            format!("blk.{}.attn_norm.weight", layer_idx),
            format!("blk.{}.ffn_norm.weight", layer_idx),
        ),
    ];

    // Try each pattern until we find matching tensors
    for (q, k, v, o, gate, up, down, attn_norm, ffn_norm) in patterns {
        if let (Some(q_buf), Some(k_buf), Some(v_buf), Some(o_buf),
                Some(gate_buf), Some(up_buf), Some(down_buf),
                Some(attn_norm_buf), Some(ffn_norm_buf)) = (
            tensors.get(&q),
            tensors.get(&k),
            tensors.get(&v),
            tensors.get(&o),
            tensors.get(&gate),
            tensors.get(&up),
            tensors.get(&down),
            tensors.get(&attn_norm),
            tensors.get(&ffn_norm),
        ) {
            return Ok(TransformerBlock::new(
                config.clone(),
                q_buf.clone(),
                k_buf.clone(),
                v_buf.clone(),
                o_buf.clone(),
                gate_buf.clone(),
                up_buf.clone(),
                down_buf.clone(),
                attn_norm_buf.clone(),
                ffn_norm_buf.clone(),
            ));
        }
    }

    Err(format!("Could not find tensors for layer {}", layer_idx).into())
}

/// Stream tokens one-by-one with callback
async fn stream_generate<F>(
    model: &mut Model,
    input_tokens: &[u32],
    max_tokens: usize,
    sampler: &Sampler,
    tokenizer: &Tokenizer,
    mut on_token: F,
) -> Result<Vec<u32>, Box<dyn std::error::Error>>
where
    F: FnMut(&str) -> Result<bool, Box<dyn std::error::Error>>, // Returns true to continue, false to stop
{
    let mut generated = Vec::new();
    let mut context = input_tokens.to_vec();
    
    tracing::info!("Starting streaming generation for {} tokens", max_tokens);
    
    for step in 0..max_tokens {
        // Generate one token
        let logits = model.forward(&context).await?;
        let next_token = sampler.sample(&logits)?;
        
        generated.push(next_token);
        context.push(next_token);
        
        // Decode and stream the token
        let token_text = tokenizer.decode(&[next_token])?;
        
        // Call the callback with the token text
        let should_continue = on_token(&token_text)?;
        
        if !should_continue {
            tracing::info!("Generation stopped early by callback at step {}", step);
            break;
        }
        
        // Check for EOS token (tokenizer-specific, typically token 2 for LLaMA)
        if next_token == 2 || next_token == tokenizer.eos_token_id().unwrap_or(2) {
            tracing::info!("EOS token encountered at step {}", step);
            break;
        }
    }
    
    Ok(generated)
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
    tracing::info!("Initialized compute engine");

    // Load HuggingFace config
    let hf_config = HuggingFaceConfig::from_file(&config_path)?;
    tracing::info!("Loaded config: {:?}", hf_config);

    // Convert to ModelConfig
    let model_config = ModelConfig::from_hf_config(&hf_config)?;

    // Load tokenizer
    let tokenizer = Tokenizer::from_file(&tokenizer_path)?;
    tracing::info!("Loaded tokenizer with vocab size: {}", tokenizer.vocab_size());

    // Load model weights based on file extension
    let (tensors, _quantization) = if model_path.extension().and_then(|s| s.to_str()) == Some("gguf") {
        let loader = GGUFLoader::new(&model_path)?;
        loader.load_tensors(&engine)?
    } else {
        let loader = SafetensorsLoader::new(&model_path)?;
        (loader.load_tensors(&engine)?, None)
    };

    tracing::info!("Loaded {} tensors", tensors.len());

    // Build transformer blocks
    let block_config = TransformerBlockConfig {
        hidden_dim: model_config.hidden_size,
        num_heads: model_config.num_attention_heads,
        num_kv_heads: model_config.num_key_value_heads,
        intermediate_size: model_config.intermediate_size,
        head_dim: model_config.hidden_size / model_config.num_attention_heads,
        rope_theta: hf_config.rope_theta.unwrap_or(10000.0),
        rms_norm_eps: model_config.rms_norm_eps,
    };

    let mut blocks = Vec::new();
    for layer_idx in 0..model_config.num_hidden_layers {
        let block = build_transformer_block(&block_config, layer_idx, &tensors)?;
        blocks.push(block);
    }

    tracing::info!("Built {} transformer blocks", blocks.len());

    // Get embedding and output tensors
    let embed_tokens = tensors.get("model.embed_tokens.weight")
        .or_else(|| tensors.get("token_embd.weight"))
        .ok_or("Could not find embedding tensor")?;

    let output_norm = tensors.get("model.norm.weight")
        .or_else(|| tensors.get("output_norm.weight"))
        .ok_or("Could not find output norm tensor")?;

    let lm_head = tensors.get("lm_head.weight")
        .or_else(|| tensors.get("output.weight"))
        .ok_or("Could not find lm_head tensor")?;

    // Create model
    let mut model = Model::new(
        model_config,
        &engine,
        embed_tokens.clone(),
        blocks,
        output_norm.clone(),
        lm_head.clone(),
    )?;

    tracing::info!("Model created successfully");

    // Create sampler with reasonable defaults for streaming
    let sampler_config = SamplerConfig {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        repetition_penalty: 1.1,
        max_tokens: 200, // Will be overridden by stream_generate
    };
    let sampler = Sampler::new(sampler_config);

    // Encode prompt
    let input_tokens = tokenizer.encode(&prompt, false)?;
    tracing::info!("Encoded prompt into {} tokens", input_tokens.len());

    println!("\n=== Streaming Generation ===");
    println!("Prompt: {}", prompt);
    print!("Output: ");
    io::stdout().flush()?;

    let start = std::time::Instant::now();

    // Stream generation with callback
    let generated = stream_generate(
        &mut model,
        &input_tokens,
        200,
        &sampler,
        &tokenizer,
        |token_text| {
            // Print each token as it arrives
            print!("{}", token_text);
            io::stdout().flush()?;
            Ok(true) // Continue generation
        },
    ).await?;

    let elapsed = start.elapsed();
    
    println!("\n\n=== Statistics ===");
    println!("Tokens generated: {}", generated.len());
    println!("Time: {:.2}s", elapsed.as_secs_f64());
    println!("Speed: {:.2} tok/s", generated.len() as f64 / elapsed.as_secs_f64());
    
    // Optionally decode full output for verification
    let full_output = tokenizer.decode(&generated)?;
    println!("\n=== Full Output (Verification) ===");
    println!("{}", full_output);

    Ok(())
}
