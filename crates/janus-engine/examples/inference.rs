//! Simple inference example showing how to load a model and generate text
//!
//! Usage (directory mode):
//!   cargo run --example inference <model_dir> "<prompt>"
//!
//! Usage (file mode):
//!   cargo run --example inference <model.gguf|model.safetensors> <config.json> <tokenizer.json> "<prompt>"
//!
//! Directory mode expects model_dir to contain:
//!   - model.gguf or model.safetensors (model weights)
//!   - config.json (HuggingFace config)
//!   - tokenizer.json (HuggingFace tokenizer)

use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use janus_engine::{
    ComputeEngine, GGUFLoader, SafetensorsLoader, ModelLoader, HuggingFaceConfig,
    Model, Tokenizer, Sampler, TransformerBlock, TransformerBlockConfig
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

    // Determine if we're in directory mode or file mode
    let (model_path, config_path, tokenizer_path, prompt) = if args.len() == 3 {
        // Directory mode: <model_dir> "<prompt>"
        let model_dir = PathBuf::from(&args[1]);
        let prompt = &args[2];
        
        if !model_dir.is_dir() {
            eprintln!("Error: '{}' is not a directory", model_dir.display());
            eprintln!("\nUsage (directory mode):");
            eprintln!("  cargo run --example inference <model_dir> \"<prompt>\"");
            eprintln!("\nUsage (file mode):");
            eprintln!("  cargo run --example inference <model_file> <config.json> <tokenizer.json> \"<prompt>\"");
            std::process::exit(1);
        }

        // Find model file
        let model_file = ["model.gguf", "model.safetensors", "pytorch_model.bin"]
            .iter()
            .map(|name| model_dir.join(name))
            .find(|path| path.exists())
            .ok_or("No model file found in directory (tried model.gguf, model.safetensors)")?;

        let config = model_dir.join("config.json");
        let tokenizer = model_dir.join("tokenizer.json");

        (model_file, config, tokenizer, prompt.clone())
    } else if args.len() == 5 {
        // File mode: <model_file> <config.json> <tokenizer.json> "<prompt>"
        (
            PathBuf::from(&args[1]),
            PathBuf::from(&args[2]),
            PathBuf::from(&args[3]),
            args[4].clone(),
        )
    } else {
        eprintln!("Usage (directory mode):");
        eprintln!("  cargo run --example inference <model_dir> \"<prompt>\"");
        eprintln!("\nUsage (file mode):");
        eprintln!("  cargo run --example inference <model_file> <config.json> <tokenizer.json> \"<prompt>\"");
        eprintln!("\nExamples:");
        eprintln!("  cargo run --example inference models/llama-7b \"Hello, world!\"");
        eprintln!("  cargo run --example inference model.gguf config.json tokenizer.json \"Hello, world!\"");
        std::process::exit(1);
    };

    println!("=== Janus Engine Inference Example ===\n");
    println!("Model file: {:?}", model_path);
    println!("Config: {:?}", config_path);
    println!("Tokenizer: {:?}", tokenizer_path);
    println!("Prompt: \"{}\"\n", prompt);

    // Load configuration
    println!(">> Loading model configuration...");
    let hf_config = HuggingFaceConfig::from_file(&config_path)?;
    let model_config = hf_config.to_model_config();
    println!("   Hidden dim: {}", model_config.hidden_dim);
    println!("   Layers: {}", model_config.num_layers);
    println!("   Attention heads: {}", model_config.num_heads);
    println!("   Vocab size: {}", model_config.vocab_size);

    // Load tokenizer
    println!("\n>> Loading tokenizer...");
    let tokenizer = Tokenizer::from_file(tokenizer_path.to_str().unwrap())?;
    println!("   Tokenizer loaded successfully");

    // Initialize GPU compute engine
    println!("\n>> Initializing GPU compute engine...");
    let engine = ComputeEngine::new().await?;
    let info = engine.adapter_info();
    println!("   Using GPU: {} ({:?})", info.name, info.backend);

    // Load model weights based on file extension
    println!("\n>> Loading model weights...");
    let extension = model_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    
    let tensors = match extension {
        "gguf" => {
            let loader = GGUFLoader::from_file(&model_path)?;
            println!("   Format: GGUF");
            
            // Show model metadata
            if let Some(arch) = loader.get_metadata("general.architecture") {
                println!("   Architecture: {}", arch);
            }
            
            // Allocate tensors to GPU
            println!("   Allocating tensors to GPU VRAM...");
            engine.allocate_tensors(&loader)?
        }
        "safetensors" => {
            let loader = SafetensorsLoader::from_file(&model_path)?;
            println!("   Format: Safetensors");
            
            // Allocate tensors to GPU
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
                eprintln!("\nAvailable tensors:");
                let mut sorted: Vec<_> = tensors.keys().collect();
                sorted.sort();
                for name in sorted.iter().take(20) {
                    eprintln!("  - {}", name);
                }
                if sorted.len() > 20 {
                    eprintln!("  ... and {} more", sorted.len() - 20);
                }
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

    // Create sampler
    let sampler = Sampler::greedy(model_config.vocab_size);

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

    // Generate text
    println!("\n>> Generating text...\n");
    println!("---");
    let _output = model.generate(&prompt, 128).await?;
    println!("---");

    println!("\n=== Generation Complete ===");
    
    Ok(())
}
