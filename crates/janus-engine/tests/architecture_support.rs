//! Test new architecture support (Phi, Gemma, Qwen)

use janus_engine::model::config::HuggingFaceConfig;

#[test]
fn test_phi_architecture() {
    let json = r#"
    {
        "architectures": ["PhiForCausalLM"],
        "hidden_size": 2560,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "vocab_size": 51200,
        "intermediate_size": 10240,
        "max_position_embeddings": 2048
    }
    "#;

    let config: HuggingFaceConfig = serde_json::from_str(json).unwrap();
    let result = config.validate();

    assert!(result.is_ok(), "Phi architecture should be supported");
    assert_eq!(config.hidden_size, 2560);
    assert_eq!(config.num_kv_heads(), 32);
}

#[test]
fn test_phi3_architecture() {
    let json = r#"
    {
        "architectures": ["Phi3ForCausalLM"],
        "hidden_size": 3072,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "vocab_size": 32064,
        "intermediate_size": 8192,
        "max_position_embeddings": 4096
    }
    "#;

    let config: HuggingFaceConfig = serde_json::from_str(json).unwrap();
    let result = config.validate();

    assert!(result.is_ok(), "Phi-3 architecture should be supported");
    assert_eq!(config.hidden_size, 3072);
}

#[test]
fn test_gemma_architecture() {
    let json = r#"
    {
        "architectures": ["GemmaForCausalLM"],
        "hidden_size": 2048,
        "num_hidden_layers": 18,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "vocab_size": 256000,
        "intermediate_size": 16384,
        "max_position_embeddings": 8192
    }
    "#;

    let config: HuggingFaceConfig = serde_json::from_str(json).unwrap();
    let result = config.validate();

    assert!(result.is_ok(), "Gemma architecture should be supported");
    assert_eq!(config.num_kv_heads(), 1);
    assert_eq!(config.head_dim(), 256);
}

#[test]
fn test_gemma2_architecture() {
    let json = r#"
    {
        "architectures": ["Gemma2ForCausalLM"],
        "hidden_size": 2304,
        "num_hidden_layers": 26,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "vocab_size": 256000,
        "intermediate_size": 9216,
        "max_position_embeddings": 8192
    }
    "#;

    let config: HuggingFaceConfig = serde_json::from_str(json).unwrap();
    let result = config.validate();

    assert!(result.is_ok(), "Gemma 2 architecture should be supported");
    assert_eq!(config.num_kv_heads(), 4);
}

#[test]
fn test_qwen_architecture() {
    let json = r#"
    {
        "architectures": ["QWenLMHeadModel"],
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "vocab_size": 151936,
        "intermediate_size": 11008,
        "max_position_embeddings": 8192
    }
    "#;

    let config: HuggingFaceConfig = serde_json::from_str(json).unwrap();
    let result = config.validate();

    assert!(result.is_ok(), "Qwen architecture should be supported");
    assert_eq!(config.vocab_size, 151936);
}

#[test]
fn test_qwen2_architecture() {
    let json = r#"
    {
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": 3584,
        "num_hidden_layers": 32,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "vocab_size": 151936,
        "intermediate_size": 18944,
        "max_position_embeddings": 32768
    }
    "#;

    let config: HuggingFaceConfig = serde_json::from_str(json).unwrap();
    let result = config.validate();

    assert!(result.is_ok(), "Qwen2 architecture should be supported");
    assert_eq!(config.num_kv_heads(), 4);
    assert_eq!(config.max_seq_len(), 32768);
}

#[test]
fn test_architecture_with_gqa() {
    // Test that GQA validation works for new architectures
    let json = r#"
    {
        "architectures": ["GemmaForCausalLM"],
        "hidden_size": 2048,
        "num_hidden_layers": 18,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "vocab_size": 256000
    }
    "#;

    let config: HuggingFaceConfig = serde_json::from_str(json).unwrap();
    let result = config.validate();

    assert!(
        result.is_ok(),
        "GQA with 8 heads / 2 KV heads should be valid"
    );
    assert_eq!(config.num_attention_heads / config.num_kv_heads(), 4);
}

#[test]
fn test_unsupported_architecture_error() {
    let json = r#"
    {
        "architectures": ["UnknownArchitecture"],
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "vocab_size": 32000
    }
    "#;

    let config: HuggingFaceConfig = serde_json::from_str(json).unwrap();
    let result = config.validate();

    assert!(
        result.is_err(),
        "Unknown architecture should fail validation"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("Unsupported architecture"));
    assert!(err_msg.contains("PhiForCausalLM"));
    assert!(err_msg.contains("GemmaForCausalLM"));
    assert!(err_msg.contains("Qwen2ForCausalLM"));
}
