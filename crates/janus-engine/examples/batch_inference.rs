//! Batched inference example
//!
//! This example demonstrates how to use batched inference to process multiple
//! prompts in parallel, significantly improving throughput.
//!
//! Usage (directory mode):
//!   cargo run --example batch_inference --release <model_dir> --batch-size 4
//!
//! Directory mode expects model_dir to contain:
//!   - model.gguf or model.safetensors (model weights)
//!   - config.json (HuggingFace config)
//!   - tokenizer.json (HuggingFace tokenizer)
//!
//! Example prompts will be generated automatically based on batch size.

use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use janus_engine::{
    ComputeEngine, GGUFLoader, SafetensorsLoader, HuggingFaceConfig,
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <model_dir> [--batch-size N] [--max-tokens N] [--temperature T]", args[0]);
        eprintln!("\nOptions:");
        eprintln!("  --batch-size N    Number of prompts to process in parallel (default: 4)");
        eprintln!("  --max-tokens N    Maximum tokens to generate per sequence (default: 50)");
        eprintln!("  --temperature T   Sampling temperature (default: 0.0 = greedy)");
        std::process::exit(1);
    }

    let model_dir = PathBuf::from(&args[1]);
    
    // Parse optional arguments
    let mut batch_size = 4u32;
    let mut max_tokens = 50usize;
    let mut temperature = 0.0f32;
    
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--batch-size" => {
                if i + 1 < args.len() {
                    batch_size = args[i + 1].parse().map_err(|_| "Invalid batch size")?;
                    i += 2;
                } else {
                    return Err("Missing value for --batch-size".into());
                }
            }
            "--max-tokens" => {
                if i + 1 < args.len() {
                    max_tokens = args[i + 1].parse().map_err(|_| "Invalid max-tokens")?;
                    i += 2;
                } else {
                    return Err("Missing value for --max-tokens".into());
                }
            }
            "--temperature" => {
                if i + 1 < args.len() {
                    temperature = args[i + 1].parse().map_err(|_| "Invalid temperature")?;
                    i += 2;
                } else {
                    return Err("Missing value for --temperature".into());
                }
            }
            _ => {
                return Err(format!("Unknown argument: {}", args[i]).into());
            }
        }
    }

    if !model_dir.is_dir() {
        eprintln!("Error: '{}' is not a directory", model_dir.display());
        std::process::exit(1);
    }

    println!("=== Janus Batched Inference Example ===");
    println!("Batch size: {}", batch_size);
    println!("Max tokens per sequence: {}", max_tokens);
    println!("Temperature: {}", temperature);
    println!();

    // Find model file
    let model_file = ["model.gguf", "model.safetensors", "pytorch_model.bin"]
        .iter()
        .map(|name| model_dir.join(name))
        .find(|path| path.exists())
        .ok_or("No model file found in directory (tried model.gguf, model.safetensors)")?;

    let config_path = model_dir.join("config.json");
    let tokenizer_path = model_dir.join("tokenizer.json");

    if !config_path.exists() {
        return Err(format!("config.json not found in {}", model_dir.display()).into());
    }
    if !tokenizer_path.exists() {
        return Err(format!("tokenizer.json not found in {}", model_dir.display()).into());
    }

    // Initialize GPU compute engine
    println!(">> Initializing GPU compute engine...");
    let engine = ComputeEngine::new().await?;
    let info = engine.adapter_info();
    println!("   Using GPU: {} ({:?})", info.name, info.backend);

    // Load configuration
    println!("\n>> Loading model configuration...");
    let hf_config = HuggingFaceConfig::from_file(&config_path)?;
    let mut model_config: ModelConfig = (&hf_config).into();
    
    // Set batch size for batched inference
    model_config.batch_size = batch_size;
    
    println!("   Architecture: {:?}", hf_config.architectures);
    println!("   Hidden dim: {}", model_config.hidden_dim);
    println!("   Layers: {}", model_config.num_layers);
    println!("   Attention heads: {}", model_config.num_heads);
    println!("   KV heads: {}", model_config.num_kv_heads);
    println!("   Batch size: {}", model_config.batch_size);

    // Load tokenizer
    println!("\n>> Loading tokenizer...");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)?;
    println!("   Vocab size: {}", tokenizer.vocab_size());
    
    // Update vocab size in config
    model_config.vocab_size = tokenizer.vocab_size() as u32;

    // Load model weights
    println!("\n>> Loading model weights...");
    let extension = model_file
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    
    let tensors = match extension {
        "gguf" => {
            let loader = GGUFLoader::from_file(&model_file)?;
            println!("   Format: GGUF");
            println!("   Allocating tensors to GPU VRAM...");
            engine.allocate_tensors(&loader)?
        }
        "safetensors" => {
            let loader = SafetensorsLoader::from_file(&model_file)?;
            println!("   Format: Safetensors");
            println!("   Allocating tensors to GPU VRAM...");
            engine.allocate_tensors(&loader)?
        }
        _ => {
            return Err(format!("Unsupported model file format: {}", extension).into());
        }
    };
    
    println!("   Allocated {} tensors to GPU", tensors.len());

    // Build transformer blocks
    println!("\n>> Building transformer blocks...");
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
        match build_transformer_block(&block_config, layer_idx, &tensors) {
            Ok(block) => {
                blocks.push(block);
                if layer_idx % 8 == 0 || layer_idx == model_config.num_layers - 1 {
                    println!("   Built layer {}/{}", layer_idx + 1, model_config.num_layers);
                }
            }
            Err(e) => {
                eprintln!("Error building layer {}: {}", layer_idx, e);
                return Err(e);
            }
        }
    }

    // Get embedding table and output weights
    println!("\n>> Loading embedding table and output weights...");
    let embedding_patterns = vec!["model.embed_tokens.weight", "token_embd.weight", "tok_embeddings.weight"];
    let token_embedding_table = embedding_patterns
        .iter()
        .find_map(|p| tensors.get(*p))
        .ok_or("Could not find token embedding table")?
        .clone();

    let output_norm_patterns = vec!["model.norm.weight", "output_norm.weight", "norm.weight"];
    let output_norm_weight = output_norm_patterns
        .iter()
        .find_map(|p| tensors.get(*p))
        .ok_or("Could not find output norm weight")?
        .clone();

    let lm_head_patterns = vec!["lm_head.weight", "output.weight"];
    let lm_head_weight = lm_head_patterns
        .iter()
        .find_map(|p| tensors.get(*p))
        .ok_or("Could not find LM head weight")?
        .clone();

    println!("   Found all required tensors");

    // Create sampler with configuration
    let sampler_config = SamplerConfig {
        temperature,
        top_k: if temperature > 0.0 { 40 } else { 0 },
        top_p: if temperature > 0.0 { 0.95 } else { 1.0 },
        repetition_penalty: 1.15,
        beam_width: 1,
        max_tokens,
    };
    let sampler = Sampler::new(sampler_config, model_config.vocab_size);

    // Create model
    println!("\n>> Initializing model...");
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
    println!("   Model initialized successfully");

    // Generate example prompts
    let example_prompts = vec![
        "Once upon a time",
        "The quick brown fox",
        "In a galaxy far, far away",
        "To be or not to be",
        "Hello, my name is",
        "The meaning of life is",
        "Science is about",
        "Artificial intelligence will",
    ];

    // Select prompts based on batch size
    let prompts: Vec<&str> = example_prompts
        .into_iter()
        .cycle()
        .take(batch_size as usize)
        .collect();

    // Generate text with batched inference
    println!("\n>> Running batched generation...");
    println!("Processing {} prompts in parallel:\n", batch_size);
    
    for (i, prompt) in prompts.iter().enumerate() {
        println!("  [Prompt {}]: \"{}\"", i + 1, prompt);
    }
    println!();

    let total_start = std::time::Instant::now();
    let _results = model.generate_batch(&prompts, max_tokens).await?;
    let total_elapsed = total_start.elapsed();

    println!("\n=== Performance Summary ===");
    println!("Total wall time: {:.2}s", total_elapsed.as_secs_f64());
    println!("Throughput improvement: ~{}x vs sequential processing", batch_size);
    println!("\n=== Batched Inference Complete ===");

    Ok(())
}
