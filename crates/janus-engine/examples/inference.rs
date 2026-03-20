//! Simple inference example showing how to load a model and generate text
//!
//! Usage:
//!   cargo run --example inference <model.gguf|model.safetensors> <tokenizer.json> "<prompt>"

use std::env;
use std::path::PathBuf;
use janus_engine::{ComputeEngine, GGUFLoader, SafetensorsLoader, ModelLoader};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();

    if args.len() < 4 {
        eprintln!("Usage: cargo run --example inference <model.gguf|model.safetensors> <tokenizer.json> \"<prompt>\"");
        eprintln!("\nExample:");
        eprintln!("  cargo run --example inference model.gguf tokenizer.json \"Hello, world!\"");
        std::process::exit(1);
    }

    let model_path = PathBuf::from(&args[1]);
    let _tokenizer_path = PathBuf::from(&args[2]);
    let prompt = &args[3];

    println!("=== Janus Engine Inference Example ===\n");
    println!("Model: {:?}", model_path);
    println!("Prompt: \"{}\"\n", prompt);

    // Detect format from file extension
    let extension = model_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    // Initialize GPU compute engine
    println!(">> Initializing GPU compute engine...");
    let engine = ComputeEngine::new().await?;
    let info = engine.adapter_info();
    println!("   Using GPU: {} ({:?})", info.name, info.backend);

    // Load model based on format
    println!("\n>> Loading model file...");
    match extension {
        "gguf" => {
            let loader = GGUFLoader::from_file(&model_path)?;
            println!("   Format: GGUF");
            
            // Show model metadata
            if let Some(arch) = loader.get_metadata("general.architecture") {
                println!("   Architecture: {}", arch);
            }
            if let Some(name) = loader.get_metadata("general.name") {
                println!("   Name: {}", name);
            }
            
            // Allocate tensors to GPU
            println!("\n>> Allocating tensors to GPU VRAM...");
            let tensors = engine.allocate_tensors(&loader)?;
            println!("   Allocated {} tensors to GPU", tensors.len());
            
            // Note: Full Model::new() would be called here in a real implementation
            // For now, this example just demonstrates the format loading
            println!("\n>> Model loaded successfully!");
            println!("   (Full inference pipeline requires model configuration)");
        }
        "safetensors" => {
            let loader = SafetensorsLoader::from_file(&model_path)?;
            println!("   Format: Safetensors");
            
            // Get tensor count
            let tensor_map = loader.tensors()?;
            println!("   Tensors: {}", tensor_map.len());
            
            // Show some tensor info
            for (name, tensor) in tensor_map.iter().take(3) {
                println!("   - {}: shape={:?}, dtype={:?}, size={} bytes",
                    name, tensor.shape, tensor.dtype, tensor.data.len());
            }
            
            // Allocate tensors to GPU
            println!("\n>> Allocating tensors to GPU VRAM...");
            let tensors = engine.allocate_tensors(&loader)?;
            println!("   Allocated {} tensors to GPU", tensors.len());
            
            println!("\n>> Model loaded successfully!");
            println!("   (Full inference pipeline requires model configuration)");
        }
        other => {
            eprintln!("Error: Unsupported file format '.{}'", other);
            eprintln!("Supported formats: .gguf, .safetensors");
            std::process::exit(1);
        }
    }

    println!("\n=== Example Complete ===");
    println!("\nNote: This example demonstrates model loading and GPU allocation.");
    println!("Full text generation requires:");
    println!("  1. Model configuration (hidden_dim, num_layers, etc.)");
    println!("  2. Tokenizer initialization");
    println!("  3. Sampler configuration");
    println!("  4. Complete Model::new() with all components");
    
    Ok(())
}
