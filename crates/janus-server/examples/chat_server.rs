//! Chat server example
//!
//! Usage:
//!   cargo run --example chat_server <model_path_or_dir> [--port 8080]
//!
//! If <model_path_or_dir> is a directory, it should contain:
//!   - model.gguf or model.safetensors
//!   - config.json
//!   - tokenizer.json
//!
//! If <model_path_or_dir> is a file, the config.json and tokenizer.json
//! should be in the same directory.
//!
//! Examples:
//!   cargo run --example chat_server ./models/llama-7b
//!   cargo run --example chat_server ./models/llama-7b/model.gguf
//!   cargo run --example chat_server ./models/llama-7b/model-00001-of-00002.safetensors
//!
//! Then test with:
//!   curl http://localhost:8080/v1/chat/completions \
//!     -H "Content-Type: application/json" \
//!     -d '{
//!       "model": "model",
//!       "messages": [
//!         {"role": "system", "content": "You are a helpful assistant."},
//!         {"role": "user", "content": "Hello!"}
//!       ],
//!       "stream": true
//!     }'

use janus_engine::{
    ChatFormatter, ComputeEngine, GGUFLoader, SafetensorsLoader, HuggingFaceConfig, Model, 
    ModelConfig, Sampler, SamplerConfig, Tokenizer, TransformerBlock, TransformerBlockConfig,
};
use janus_server::{create_router, handlers::AppState};
use std::collections::HashMap;
use std::env;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Build TransformerBlock from tensor map
fn build_transformer_block(
    config: &TransformerBlockConfig,
    layer_idx: u32,
    tensors: &HashMap<String, wgpu::Buffer>,
) -> Result<TransformerBlock, Box<dyn std::error::Error>> {
    let patterns = vec![
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

    for (q, k, v, o, gate, up, down, attn_norm, ffn_norm) in patterns {
        if let (
            Some(q_buf),
            Some(k_buf),
            Some(v_buf),
            Some(o_buf),
            Some(gate_buf),
            Some(up_buf),
            Some(down_buf),
            Some(attn_norm_buf),
            Some(ffn_norm_buf),
        ) = (
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
        eprintln!("Usage: {} <model_path_or_dir> [--port PORT]", args[0]);
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  {} ./models/llama-7b", args[0]);
        eprintln!("  {} ./models/llama-7b/model.gguf", args[0]);
        eprintln!("  {} ./models/llama-7b/model-00001-of-00002.safetensors", args[0]);
        std::process::exit(1);
    }

    let model_path_or_dir = PathBuf::from(&args[1]);
    let mut port = 8080u16;

    // Parse optional --port argument
    let mut i = 2;
    while i < args.len() {
        if args[i] == "--port" && i + 1 < args.len() {
            port = args[i + 1].parse()?;
            i += 2;
        } else {
            i += 1;
        }
    }

    // Determine model file and directory paths
    let (model_path, model_dir) = if model_path_or_dir.is_file() {
        // User specified a direct file path
        let model_dir = model_path_or_dir
            .parent()
            .ok_or("Model file has no parent directory")?
            .to_path_buf();
        (model_path_or_dir, model_dir)
    } else if model_path_or_dir.is_dir() {
        // User specified a directory - find model.gguf or model.safetensors
        let model_file = model_path_or_dir
            .join("model.gguf")
            .exists()
            .then(|| model_path_or_dir.join("model.gguf"))
            .or_else(|| {
                model_path_or_dir
                    .join("model.safetensors")
                    .exists()
                    .then(|| model_path_or_dir.join("model.safetensors"))
            })
            .ok_or_else(|| {
                format!(
                    "No model file found in directory {:?}. Looking for model.gguf or model.safetensors",
                    model_path_or_dir
                )
            })?;
        (model_file, model_path_or_dir)
    } else {
        return Err(format!(
            "Model path {:?} does not exist or is not a file/directory",
            model_path_or_dir
        )
        .into());
    };

    let config_path = model_dir.join("config.json");
    let tokenizer_path = model_dir.join("tokenizer.json");

    tracing::info!("Loading model from {:?}", model_path);
    tracing::info!("Loading config from {:?}", config_path);
    tracing::info!("Loading tokenizer from {:?}", tokenizer_path);

    // Initialize compute engine
    tracing::info!("Initializing GPU compute engine...");
    let engine = ComputeEngine::new().await?;
    let device_info = engine.adapter_info();
    tracing::info!("Using GPU: {} ({:?})", device_info.name, device_info.backend);

    // Load model file
    let tensors = if model_path.extension().and_then(|s| s.to_str()) == Some("gguf") {
        tracing::info!("Loading GGUF model from {:?}", model_path);
        let model_loader = GGUFLoader::from_file(&model_path)?;
        engine.allocate_tensors(&model_loader)?
    } else if model_path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s == "safetensors")
        .unwrap_or(false)
    {
        tracing::info!("Loading Safetensors model from {:?}", model_path);
        let model_loader = SafetensorsLoader::from_file(&model_path)?;
        engine.allocate_tensors(&model_loader)?
    } else {
        return Err(format!(
            "Unsupported model file extension. Expected .gguf or .safetensors, got {:?}",
            model_path.extension()
        )
        .into());
    };

    // Load config
    let hf_config = HuggingFaceConfig::from_file(&config_path)?;
    let model_config: ModelConfig = (&hf_config).into();

    // Load tokenizer
    let tokenizer = Tokenizer::from_file(&tokenizer_path)?;

    // Create sampler with reasonable defaults for chat
    let sampler_config = SamplerConfig {
        temperature: 0.7,
        top_k: 40,
        top_p: 0.9,
        repetition_penalty: 1.1,
        beam_width: 1,
        max_tokens: 512,
    };
    let sampler = Sampler::new(sampler_config, tokenizer.vocab_size() as u32);

    // Build transformer blocks
    let mut blocks = Vec::new();
    let block_config = TransformerBlockConfig {
        batch_size: model_config.batch_size,
        hidden_dim: model_config.hidden_dim,
        num_heads: model_config.num_heads,
        num_kv_heads: model_config.num_kv_heads,
        head_dim: model_config.head_dim,
        ffn_dim: model_config.ffn_dim,
        rms_norm_eps: model_config.rms_norm_eps,
    };

    for layer_idx in 0..model_config.num_layers {
        tracing::info!("Building transformer block {}/{}", layer_idx + 1, model_config.num_layers);
        let block = build_transformer_block(&block_config, layer_idx, &tensors)?;
        blocks.push(block);
    }

    // Extract embedding and output tensors
    let token_embedding_table = tensors
        .get("token_embd.weight")
        .or_else(|| tensors.get("model.embed_tokens.weight"))
        .ok_or("Could not find token embedding table")?
        .clone();

    let output_norm_weight = tensors
        .get("output_norm.weight")
        .or_else(|| tensors.get("model.norm.weight"))
        .ok_or("Could not find output normalization weight")?
        .clone();

    let lm_head_weight = tensors
        .get("output.weight")
        .or_else(|| tensors.get("lm_head.weight"))
        .ok_or("Could not find LM head weight")?
        .clone();

    // Create model
    let model = Model::new(
        model_config,
        engine,
        tokenizer,
        sampler,
        token_embedding_table,
        blocks,
        output_norm_weight,
        lm_head_weight,
    )?;

    // Create chat formatter
    let model_name = model_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");
    let chat_formatter = ChatFormatter::from_model_name(model_name);

    // Create shared application state
    let state = Arc::new(AppState {
        model: Arc::new(Mutex::new(model)),
        chat_formatter,
        model_name: model_name.to_string(),
    });

    // Create router
    let app = create_router(state);

    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!("Starting server on http://{}", addr);
    tracing::info!("Chat endpoint: http://{}/v1/chat/completions", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
