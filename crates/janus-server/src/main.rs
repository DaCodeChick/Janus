use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use janus_engine::model::block::get_tensor;
use janus_engine::model::config::model_config_from_gguf_metadata;
use janus_engine::{
    ChatFormatter, ChatTemplateFormat, ComputeEngine, GgufLoader, HuggingFaceConfig, JanusApp,
    Model, ModelConfig, SafetensorsLoader, Sampler, SamplerConfig, Tokenizer, TransformerBlock,
    TransformerBlockConfig,
};
use janus_mod_imggen::ImgGenPlugin;
use janus_mod_instruct::InstructPlugin;
use janus_mod_knowledge::KnowledgePlugin;
use janus_mod_lora::LoraPlugin;
use janus_mod_router::RouterPlugin;
use janus_mod_rp::RpPlugin;
use janus_mod_tts::TtsPlugin;
use janus_mod_vecmem::VecMemPlugin;
use janus_mod_vision::VisionPlugin;
use janus_mod_vismem::VisMemPlugin;
use janus_mod_voice::VoicePlugin;
use janus_server::{create_router, handlers::AppState};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum TemplateArg {
    Chatml,
    Llama3,
    Llama2,
    Alpaca,
    Vicuna,
    Zephyr,
}

impl From<TemplateArg> for ChatTemplateFormat {
    fn from(value: TemplateArg) -> Self {
        match value {
            TemplateArg::Chatml => ChatTemplateFormat::ChatML,
            TemplateArg::Llama3 => ChatTemplateFormat::Llama3,
            TemplateArg::Llama2 => ChatTemplateFormat::Llama2,
            TemplateArg::Alpaca => ChatTemplateFormat::Alpaca,
            TemplateArg::Vicuna => ChatTemplateFormat::Vicuna,
            TemplateArg::Zephyr => ChatTemplateFormat::Zephyr,
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "janus-server")]
#[command(about = "OpenAI-compatible Janus chat server")]
struct Args {
    /// Model file (.gguf or .safetensors) or directory containing model files
    model: PathBuf,

    /// Host interface to bind
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Port to listen on
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Override auto-detected chat template
    #[arg(long)]
    template: Option<TemplateArg>,
}

fn build_transformer_block(
    config: &TransformerBlockConfig,
    layer_idx: u32,
    tensors: &HashMap<String, wgpu::Buffer>,
) -> Result<TransformerBlock> {
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

fn resolve_model_paths(path: &Path) -> Result<(PathBuf, PathBuf)> {
    if path.is_file() {
        let model_dir = path
            .parent()
            .context("model file does not have a parent directory")?
            .to_path_buf();
        return Ok((path.to_path_buf(), model_dir));
    }

    if path.is_dir() {
        let gguf = path.join("model.gguf");
        if gguf.exists() {
            return Ok((gguf, path.to_path_buf()));
        }

        let safetensors = path.join("model.safetensors");
        if safetensors.exists() {
            return Ok((safetensors, path.to_path_buf()));
        }

        bail!(
            "no model file found in {:?} (expected model.gguf or model.safetensors)",
            path
        );
    }

    bail!("model path {:?} does not exist", path)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let mut app = JanusApp::new();
    app.add_plugin(InstructPlugin)
        .add_plugin(RouterPlugin)
        .add_plugin(KnowledgePlugin)
        .add_plugin(LoraPlugin)
        .add_plugin(RpPlugin)
        .add_plugin(TtsPlugin)
        .add_plugin(VecMemPlugin)
        .add_plugin(VisionPlugin)
        .add_plugin(VisMemPlugin)
        .add_plugin(VoicePlugin)
        .add_plugin(ImgGenPlugin);

    let (model_path, model_dir) = resolve_model_paths(&args.model)?;
    let config_path = model_dir.join("config.json");
    let tokenizer_path = model_dir.join("tokenizer.json");
    let extension = model_path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();

    println!("Loading model: {:?}", model_path);
    println!("Loading tokenizer: {:?}", tokenizer_path);
    if extension == "safetensors" {
        println!("Loading config: {:?}", config_path);
    } else {
        println!("Model config source: embedded GGUF metadata");
    }

    let engine = ComputeEngine::new().await.context("failed to initialize GPU")?;
    app.set_gpu_context(&engine);
    let tokenizer =
        Tokenizer::from_file(&tokenizer_path).context("failed to load tokenizer.json")?;

    let (tensors, model_config) = if extension == "gguf" {
        let loader = GgufLoader::from_file(&model_path).context("failed to parse GGUF file")?;
        let model_config = model_config_from_gguf_metadata(
            loader.gguf_metadata(),
            tokenizer.vocab_size() as u32,
        )
        .map_err(|e| anyhow::anyhow!("failed to build config from GGUF metadata: {}", e))?;
        (
            engine
                .allocate_tensors(&loader)
                .context("failed to allocate GGUF tensors")?,
            model_config,
        )
    } else if extension == "safetensors" {
        let hf_config =
            HuggingFaceConfig::from_file(&config_path).context("failed to load config.json")?;
        let model_config: ModelConfig = (&hf_config).into();
        let loader =
            SafetensorsLoader::from_file(&model_path).context("failed to parse Safetensors")?;
        (
            engine
                .allocate_tensors(&loader)
                .context("failed to allocate Safetensors tensors")?,
            model_config,
        )
    } else {
        bail!("unsupported model extension '{}': expected .gguf or .safetensors", extension);
    };

    let sampler = Sampler::new(
        SamplerConfig {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.1,
            beam_width: 1,
            max_tokens: 512,
        },
        tokenizer.vocab_size() as u32,
    );

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
        blocks.push(
            build_transformer_block(&block_config, layer_idx, &tensors)
                .with_context(|| format!("failed to build transformer block {}", layer_idx))?,
        );
    }

    let token_embedding_table =
        get_tensor(&tensors, "model.embed_tokens.weight", "token_embd.weight")?.clone();
    let output_norm_weight = get_tensor(&tensors, "model.norm.weight", "output_norm.weight")?.clone();
    let lm_head_weight = get_tensor(&tensors, "lm_head.weight", "output.weight")?.clone();

    let model = Model::new(
        model_config,
        engine,
        tokenizer,
        sampler,
        token_embedding_table,
        blocks,
        output_norm_weight,
        lm_head_weight,
    )
    .context("failed to assemble model")?;

    let model_name = model_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("unknown")
        .to_string();

    let chat_formatter = if let Some(template) = args.template {
        ChatFormatter::new(template.into())
    } else {
        let detection_name = model_dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(&model_name);
        ChatFormatter::from_model_name(detection_name)
    };

    let shared_model = Arc::new(Mutex::new(model));
    app.set_model(shared_model.clone());

    let state = Arc::new(AppState {
        model: shared_model,
        chat_formatter,
        model_name,
    });

    app.set_router(create_router(state));
    let addr = format!("{}:{}", args.host, args.port)
        .parse()
        .with_context(|| format!("invalid bind address {}:{}", args.host, args.port))?;
    app.set_bind_addr(addr);

    println!("Janus server listening on http://{}", addr);
    app.run().await?;

    Ok(())
}
