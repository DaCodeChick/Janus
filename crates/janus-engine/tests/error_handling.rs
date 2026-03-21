//! Integration tests for error handling and validation
//!
//! These tests verify that the model loading and configuration validation
//! provide helpful error messages for common issues.

use janus_engine::model::config::HuggingFaceConfig;

#[test]
fn test_config_zero_values() {
    let json = r#"
    {
        "hidden_size": 0,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "vocab_size": 32000
    }
    "#;

    let config: HuggingFaceConfig = serde_json::from_str(json).unwrap();
    let result = config.validate();

    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("hidden_size"));
    assert!(err_msg.contains("greater than 0"));
}

#[test]
fn test_config_indivisible_dimensions() {
    let json = r#"
    {
        "hidden_size": 4097,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "vocab_size": 32000
    }
    "#;

    let config: HuggingFaceConfig = serde_json::from_str(json).unwrap();
    let result = config.validate();

    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("hidden_size"));
    assert!(err_msg.contains("divisible"));
    assert!(err_msg.contains("num_attention_heads"));
}

#[test]
fn test_config_gqa_validation() {
    let json = r#"
    {
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 5,
        "vocab_size": 32000
    }
    "#;

    let config: HuggingFaceConfig = serde_json::from_str(json).unwrap();
    let result = config.validate();

    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("num_attention_heads"));
    assert!(err_msg.contains("num_key_value_heads"));
    assert!(err_msg.contains("divisible"));
}

#[test]
fn test_config_unsupported_architecture() {
    let json = r#"
    {
        "architectures": ["GPT2LMHeadModel"],
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "vocab_size": 50257
    }
    "#;

    let config: HuggingFaceConfig = serde_json::from_str(json).unwrap();
    let result = config.validate();

    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("Unsupported architecture"));
    assert!(err_msg.contains("GPT2LMHeadModel"));
    assert!(err_msg.contains("Supported architectures"));
}

#[test]
fn test_config_unreasonably_large_values() {
    let json = r#"
    {
        "hidden_size": 100000,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "vocab_size": 32000
    }
    "#;

    let config: HuggingFaceConfig = serde_json::from_str(json).unwrap();
    let result = config.validate();

    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("unreasonably large"));
}

#[test]
fn test_config_valid_llama() {
    let json = r#"
    {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "vocab_size": 32000,
        "intermediate_size": 11008,
        "max_position_embeddings": 2048,
        "rms_norm_eps": 0.00001
    }
    "#;

    let config: HuggingFaceConfig = serde_json::from_str(json).unwrap();
    let result = config.validate();

    assert!(result.is_ok(), "Valid LLaMA config should pass validation");
}

#[test]
fn test_config_valid_tinyllama_gqa() {
    let json = r#"
    {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 2048,
        "num_hidden_layers": 22,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "vocab_size": 32000,
        "intermediate_size": 5632,
        "max_position_embeddings": 2048
    }
    "#;

    let config: HuggingFaceConfig = serde_json::from_str(json).unwrap();
    let result = config.validate();

    assert!(
        result.is_ok(),
        "Valid TinyLlama GQA config should pass validation"
    );

    // Verify GQA configuration
    assert_eq!(config.num_attention_heads, 32);
    assert_eq!(config.num_kv_heads(), 4);
    assert_eq!(config.num_attention_heads % config.num_kv_heads(), 0);
}

#[test]
fn test_config_missing_field_error_message() {
    let json = r#"
    {
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "vocab_size": 32000
    }
    "#;

    let result: Result<HuggingFaceConfig, _> = serde_json::from_str(json);

    assert!(result.is_err());
    // The serde error should mention the missing field
}

#[test]
fn test_config_supported_architectures() {
    let supported = vec![
        "LlamaForCausalLM",
        "MistralForCausalLM",
        "GPTNeoXForCausalLM",
        "PhiForCausalLM",
        "Phi3ForCausalLM",
        "GemmaForCausalLM",
        "Gemma2ForCausalLM",
        "QWenLMHeadModel",
        "Qwen2ForCausalLM",
    ];

    for arch in supported {
        let json = format!(
            r#"{{
                "architectures": ["{}"],
                "hidden_size": 2048,
                "num_hidden_layers": 16,
                "num_attention_heads": 16,
                "vocab_size": 32000
            }}"#,
            arch
        );

        let config: HuggingFaceConfig = serde_json::from_str(&json).unwrap();
        let result = config.validate();

        assert!(
            result.is_ok(),
            "Architecture {} should be supported, but validation failed: {:?}",
            arch,
            result.unwrap_err()
        );
    }
}

#[test]
fn test_helpful_error_message_content() {
    let json = r#"
    {
        "hidden_size": 4097,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "vocab_size": 32000
    }
    "#;

    let config: HuggingFaceConfig = serde_json::from_str(json).unwrap();
    let result = config.validate();

    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();

    // Check that error message contains:
    // 1. The actual problem
    assert!(err_msg.contains("divisible") || err_msg.contains("Invalid"));

    // 2. The specific values
    assert!(err_msg.contains("4097") || err_msg.contains("32"));

    // 3. Helpful context
    // The error should be informative enough that a user knows what to fix
    assert!(err_msg.len() > 50, "Error message should be detailed");
}
