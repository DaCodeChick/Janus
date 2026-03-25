//! Chat server example
//!
//! Usage:
//!   cargo run --example chat_server <model_path_or_dir> [--port 8080] [--template zephyr]
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
//! Once running, open http://localhost:8080/chat in your browser for an
//! interactive chat interface, or visit http://localhost:8080 for API docs.

use janus_engine::{
    ChatFormatter, ChatTemplateFormat, ComputeEngine, GGUFLoader, SafetensorsLoader,
    HuggingFaceConfig, Model,
    ModelConfig, Sampler, SamplerConfig, Tokenizer, TransformerBlock, TransformerBlockConfig,
};
use janus_server::{create_router, handlers::AppState};
use std::collections::HashMap;
use std::env;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

fn parse_template(name: &str) -> Option<ChatTemplateFormat> {
    match name.to_lowercase().as_str() {
        "chatml" => Some(ChatTemplateFormat::ChatML),
        "llama3" | "llama-3" => Some(ChatTemplateFormat::Llama3),
        "llama2" | "llama-2" => Some(ChatTemplateFormat::Llama2),
        "alpaca" => Some(ChatTemplateFormat::Alpaca),
        "vicuna" => Some(ChatTemplateFormat::Vicuna),
        "zephyr" | "tinyllama" => Some(ChatTemplateFormat::Zephyr),
        _ => None,
    }
}

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
        eprintln!(
            "Usage: {} <model_path_or_dir> [--port PORT] [--template TEMPLATE]",
            args[0]
        );
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  {} ./models/llama-7b", args[0]);
        eprintln!("  {} ./models/llama-7b/model.gguf", args[0]);
        eprintln!("  {} ./models/llama-7b/model-00001-of-00002.safetensors", args[0]);
        eprintln!("  {} ./models/tinyllama/model.safetensors --template llama2", args[0]);
        eprintln!();
        eprintln!("Templates: chatml, llama3, llama2, alpaca, vicuna, zephyr");
        std::process::exit(1);
    }

    let model_path_or_dir = PathBuf::from(&args[1]);
    let input_is_file = model_path_or_dir.is_file();
    let mut port = 8080u16;
    let mut template_override: Option<ChatTemplateFormat> = None;

    // Parse optional --port argument
    let mut i = 2;
    while i < args.len() {
        if args[i] == "--port" && i + 1 < args.len() {
            port = args[i + 1].parse()?;
            i += 2;
        } else if (args[i] == "--template" || args[i] == "-t") && i + 1 < args.len() {
            template_override = parse_template(&args[i + 1]);
            if template_override.is_none() {
                return Err(format!(
                    "Unknown template '{}'. Supported: chatml, llama3, llama2, alpaca, vicuna, zephyr",
                    args[i + 1]
                )
                .into());
            }
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

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🔧 Janus Chat Server Initialization");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    
    println!("📂 Loading model from: {:?}", model_path);
    println!("⚙️  Loading config from: {:?}", config_path);
    println!("🔤 Loading tokenizer from: {:?}", tokenizer_path);
    println!();

    // Initialize compute engine
    println!("🎮 Initializing GPU compute engine...");
    let engine = match ComputeEngine::new().await {
        Ok(e) => {
            println!("✅ GPU compute engine initialized successfully");
            e
        }
        Err(e) => {
            eprintln!("❌ Failed to initialize GPU compute engine: {}", e);
            return Err(e.into());
        }
    };
    
    let device_info = engine.adapter_info();
    println!("🖥️  Using GPU: {} ({:?})", device_info.name, device_info.backend);
    println!();

    // Load model file
    println!("📥 Loading model weights...");
    let tensors = if model_path.extension().and_then(|s| s.to_str()) == Some("gguf") {
        println!("   Format: GGUF");
        let model_loader = match GGUFLoader::from_file(&model_path) {
            Ok(loader) => {
                println!("✅ GGUF file parsed successfully");
                loader
            }
            Err(e) => {
                eprintln!("❌ Failed to load GGUF file: {}", e);
                return Err(e.into());
            }
        };
        
        println!("🔄 Allocating tensors to GPU memory...");
        match engine.allocate_tensors(&model_loader) {
            Ok(t) => {
                println!("✅ Tensors allocated to GPU ({} tensors)", t.len());
                t
            }
            Err(e) => {
                eprintln!("❌ Failed to allocate tensors: {}", e);
                return Err(e.into());
            }
        }
    } else if model_path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s == "safetensors")
        .unwrap_or(false)
    {
        println!("   Format: Safetensors");
        let model_loader = match SafetensorsLoader::from_file(&model_path) {
            Ok(loader) => {
                println!("✅ Safetensors file parsed successfully");
                loader
            }
            Err(e) => {
                eprintln!("❌ Failed to load Safetensors file: {}", e);
                return Err(e.into());
            }
        };
        
        println!("🔄 Allocating tensors to GPU memory...");
        match engine.allocate_tensors(&model_loader) {
            Ok(t) => {
                println!("✅ Tensors allocated to GPU ({} tensors)", t.len());
                t
            }
            Err(e) => {
                eprintln!("❌ Failed to allocate tensors: {}", e);
                return Err(e.into());
            }
        }
    } else {
        eprintln!("❌ Unsupported model file extension. Expected .gguf or .safetensors, got {:?}", model_path.extension());
        return Err(format!(
            "Unsupported model file extension. Expected .gguf or .safetensors, got {:?}",
            model_path.extension()
        )
        .into());
    };
    println!();

    // Load config
    println!("⚙️  Loading model configuration...");
    let hf_config = match HuggingFaceConfig::from_file(&config_path) {
        Ok(cfg) => {
            println!("✅ Config loaded successfully");
            cfg
        }
        Err(e) => {
            eprintln!("❌ Failed to load config: {}", e);
            return Err(e.into());
        }
    };
    let model_config: ModelConfig = (&hf_config).into();
    println!("   Layers: {}", model_config.num_layers);
    println!("   Hidden dim: {}", model_config.hidden_dim);
    println!("   Heads: {}", model_config.num_heads);
    println!();

    // Load tokenizer
    println!("🔤 Loading tokenizer...");
    let tokenizer = match Tokenizer::from_file(&tokenizer_path) {
        Ok(tok) => {
            println!("✅ Tokenizer loaded ({} vocab size)", tok.vocab_size());
            tok
        }
        Err(e) => {
            eprintln!("❌ Failed to load tokenizer: {}", e);
            return Err(e.into());
        }
    };
    println!();

    // Create sampler with reasonable defaults for chat
    println!("🎲 Initializing sampler...");
    let sampler_config = SamplerConfig {
        temperature: 0.7,
        top_k: 40,
        top_p: 0.9,
        repetition_penalty: 1.1,
        beam_width: 1,
        max_tokens: 512,
    };
    let sampler = Sampler::new(sampler_config, tokenizer.vocab_size() as u32);
    println!("✅ Sampler initialized");
    println!();

    // Build transformer blocks
    println!("🔨 Building transformer blocks...");
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
        if layer_idx % 5 == 0 || layer_idx == model_config.num_layers - 1 {
            println!("   Building block {}/{}", layer_idx + 1, model_config.num_layers);
        }
        let block = match build_transformer_block(&block_config, layer_idx, &tensors) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("❌ Failed to build transformer block {}: {}", layer_idx, e);
                return Err(e);
            }
        };
        blocks.push(block);
    }
    println!("✅ All transformer blocks built successfully");
    println!();

    // Extract embedding and output tensors
    println!("🔍 Extracting embedding and output tensors...");
    let token_embedding_table = tensors
        .get("token_embd.weight")
        .or_else(|| tensors.get("model.embed_tokens.weight"))
        .ok_or_else(|| {
            eprintln!("❌ Could not find token embedding table");
            "Could not find token embedding table"
        })?
        .clone();

    let output_norm_weight = tensors
        .get("output_norm.weight")
        .or_else(|| tensors.get("model.norm.weight"))
        .ok_or_else(|| {
            eprintln!("❌ Could not find output normalization weight");
            "Could not find output normalization weight"
        })?
        .clone();

    let lm_head_weight = tensors
        .get("output.weight")
        .or_else(|| tensors.get("lm_head.weight"))
        .ok_or_else(|| {
            eprintln!("❌ Could not find LM head weight");
            "Could not find LM head weight"
        })?
        .clone();
    println!("✅ Embedding and output tensors extracted");
    println!();

    // Create model
    println!("🧠 Assembling final model...");
    let model = match Model::new(
        model_config,
        engine,
        tokenizer,
        sampler,
        token_embedding_table,
        blocks,
        output_norm_weight,
        lm_head_weight,
    ) {
        Ok(m) => {
            println!("✅ Model assembled successfully");
            m
        }
        Err(e) => {
            eprintln!("❌ Failed to create model: {}", e);
            return Err(e.into());
        }
    };
    println!();

    // Create chat formatter
    println!("💬 Setting up chat formatter...");
    let model_name = model_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");

    let detection_name = if input_is_file {
        model_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(model_name)
    } else {
        model_name
    };

    let chat_formatter = if let Some(template) = template_override {
        println!("   Using explicit template override: {:?}", template);
        ChatFormatter::new(template)
    } else {
        println!("   Auto-detecting template from: {}", detection_name);
        ChatFormatter::from_model_name(detection_name)
    };
    println!("✅ Chat formatter ready ({:?})", chat_formatter.format());
    println!();

    // Create shared application state
    println!("🌐 Creating application state...");
    let state = Arc::new(AppState {
        model: Arc::new(Mutex::new(model)),
        chat_formatter,
        model_name: model_name.to_string(),
    });
    println!("✅ Application state created");
    println!();

    // Create router
    println!("🛣️  Setting up routes...");
    let app = create_router(state);
    println!("✅ Routes configured");
    println!();

    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🚀 Janus Chat Server Started!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("💬 Chat UI:      http://{}/chat", addr);
    println!("📖 API Docs:     http://{}", addr);
    println!("💚 Health Check: http://{}/health", addr);
    println!();
    println!("Server listening on {}", addr);
    println!("Press Ctrl+C to stop");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
