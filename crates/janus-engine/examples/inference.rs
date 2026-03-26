//! Simple inference example showing how to load a model and generate text
//!
//! Usage (directory mode):
//!   cargo run --example inference <model_dir> "<prompt>"
//!
//! Usage (file mode, GGUF):
//!   cargo run --example inference <model.gguf> <tokenizer.json> "<prompt>"
//!
//! Usage (file mode, Safetensors):
//!   cargo run --example inference <model.safetensors> <config.json> <tokenizer.json> "<prompt>"
//!
//! Directory mode expects model_dir to contain:
//!   - model.gguf or model.safetensors (model weights)
//!   - config.json (HuggingFace config, only for Safetensors)
//!   - tokenizer.json (HuggingFace tokenizer)

use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use janus_engine::{
    ComputeEngine, GGUFFile, SafetensorsFile, ModelLoader, HuggingFaceConfig,
    Model, ModelConfig, Tokenizer, Sampler, TransformerBlock, TransformerBlockConfig
};
use janus_engine::model::block::get_tensor;
use janus_engine::model::config::model_config_from_gguf_metadata;

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
            eprintln!("  cargo run --example inference <model.gguf> <tokenizer.json> \"<prompt>\"");
            eprintln!("  cargo run --example inference <model.safetensors> <config.json> <tokenizer.json> \"<prompt>\"");
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

        let config_path = if model_file.extension().and_then(|e| e.to_str()) == Some("gguf") {
            None
        } else {
            Some(config)
        };

        (model_file, config_path, tokenizer, prompt.clone())
    } else if args.len() == 4 {
        // File mode (GGUF): <model.gguf> <tokenizer.json> "<prompt>"
        (
            PathBuf::from(&args[1]),
            None,
            PathBuf::from(&args[2]),
            args[3].clone(),
        )
    } else if args.len() == 5 {
        // File mode (Safetensors): <model.safetensors> <config.json> <tokenizer.json> "<prompt>"
        (
            PathBuf::from(&args[1]),
            Some(PathBuf::from(&args[2])),
            PathBuf::from(&args[3]),
            args[4].clone(),
        )
    } else {
        eprintln!("Usage (directory mode):");
        eprintln!("  cargo run --example inference <model_dir> \"<prompt>\"");
        eprintln!("\nUsage (file mode):");
        eprintln!("  cargo run --example inference <model.gguf> <tokenizer.json> \"<prompt>\"");
        eprintln!("  cargo run --example inference <model.safetensors> <config.json> <tokenizer.json> \"<prompt>\"");
        eprintln!("\nExamples:");
        eprintln!("  cargo run --example inference models/llama-7b \"Hello, world!\"");
        eprintln!("  cargo run --example inference model.gguf tokenizer.json \"Hello, world!\"");
        eprintln!("  cargo run --example inference model.safetensors config.json tokenizer.json \"Hello, world!\"");
        std::process::exit(1);
    };

    println!("=== Janus Engine Inference Example ===\n");
    println!("Model file: {:?}", model_path);
    println!("Config: {:?}", config_path);
    println!("Tokenizer: {:?}", tokenizer_path);
    println!("Prompt: \"{}\"\n", prompt);

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
    
    let (tensors, model_config) = match extension {
        "gguf" => {
            let loader = GGUFFile::from_file(&model_path)?;
            println!("   Format: GGUF");
            
            // Show model metadata
            if let Some(arch) = loader.get_metadata("general.architecture") {
                println!("   Architecture: {}", arch);
            }

            let model_config = model_config_from_gguf_metadata(
                loader.gguf_metadata(),
                tokenizer.vocab_size() as u32,
            )
            .map_err(|e| format!("Failed to build model config from GGUF metadata: {}", e))?;
            
            // Allocate tensors to GPU
            println!("   Allocating tensors to GPU VRAM...");
            (engine.allocate_tensors(&loader)?, model_config)
        }
        "safetensors" => {
            println!(">> Loading model configuration...");
            let config_path = config_path
                .as_ref()
                .ok_or("config.json is required when loading Safetensors")?;
            let hf_config = HuggingFaceConfig::from_file(config_path)?;
            let model_config: ModelConfig = (&hf_config).into();

            let loader = SafetensorsFile::from_file(&model_path)?;
            println!("   Format: Safetensors");
            
            // Allocate tensors to GPU
            println!("   Allocating tensors to GPU VRAM...");
            (engine.allocate_tensors(&loader)?, model_config)
        }
        _ => {
            return Err(format!("Unsupported model file format: {}", extension).into());
        }
    };
    
    println!("   Allocated {} tensors to GPU", tensors.len());
    println!("\n>> Model configuration:");
    println!("   Hidden dim: {}", model_config.hidden_dim);
    println!("   Layers: {}", model_config.num_layers);
    println!("   Attention heads: {}", model_config.num_heads);
    println!("   Vocab size: {}", model_config.vocab_size);

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
    let token_embedding_table =
        get_tensor(&tensors, "model.embed_tokens.weight", "token_embd.weight")?.clone();

    let output_norm_weight =
        get_tensor(&tensors, "model.norm.weight", "output_norm.weight")?.clone();

    let lm_head_weight = get_tensor(&tensors, "lm_head.weight", "output.weight")?.clone();

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
