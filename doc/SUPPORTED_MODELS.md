# Supported Model Architectures

Janus Engine supports a wide range of modern decoder-only transformer architectures. All supported models use similar core components: RoPE positional embeddings, RMSNorm layer normalization, and grouped-query attention (GQA).

## Fully Supported Architectures

### LLaMA Family

**Architecture**: `LlamaForCausalLM`

- ✅ **LLaMA** (Meta AI) - 7B, 13B, 30B, 65B
- ✅ **LLaMA 2** (Meta AI) - 7B, 13B, 70B
- ✅ **LLaMA 3** (Meta AI) - 8B, 70B
- ✅ **Vicuna** - All sizes
- ✅ **Alpaca** - All sizes

**Key Features**:
- RoPE positional embeddings
- RMSNorm layer normalization
- Grouped-query attention (GQA) support
- SwiGLU activation in FFN

### Mistral Family

**Architecture**: `MistralForCausalLM`

- ✅ **Mistral** (Mistral AI) - 7B
- ✅ **Mixtral** - 8x7B (note: MoE not yet optimized)

**Key Features**:
- Sliding window attention (treated as standard attention)
- GQA with 8 KV heads
- Same architecture as LLaMA with minor variations

### TinyLlama / Pythia

**Architecture**: `GPTNeoXForCausalLM`

- ✅ **TinyLlama** - 1.1B
- ✅ **Pythia** (EleutherAI) - 70M to 12B
- ✅ **GPT-NeoX** - 20B

**Key Features**:
- GPT-NeoX architecture
- Rotary positional embeddings
- Parallel attention and FFN (optimized in Janus)

### Microsoft Phi Family

**Architectures**: `PhiForCausalLM`, `Phi3ForCausalLM`

- ✅ **Phi-1** - 1.3B
- ✅ **Phi-1.5** - 1.3B
- ✅ **Phi-2** - 2.7B
- ✅ **Phi-3** - 3.8B, 7B, 14B

**Key Features**:
- Compact, efficient models trained on high-quality data
- Similar architecture to LLaMA
- Supports both standard attention and GQA
- RoPE and RMSNorm

**Notes**:
- Some Phi models use different activation functions (handled transparently)
- Phi-3 uses longer context lengths (up to 128k with special handling)

### Google Gemma Family

**Architectures**: `GemmaForCausalLM`, `Gemma2ForCausalLM`

- ✅ **Gemma** - 2B, 7B
- ✅ **Gemma 2** - 9B, 27B

**Key Features**:
- Highly optimized architecture similar to LLaMA
- Aggressive GQA (e.g., 8 heads with 1 KV head)
- RoPE and RMSNorm
- Efficient inference characteristics

**Notes**:
- Gemma uses very aggressive GQA configurations
- Gemma 2 includes architectural improvements for efficiency

### Alibaba Qwen Family

**Architectures**: `QWenLMHeadModel`, `Qwen2ForCausalLM`

- ✅ **Qwen** - 1.8B, 7B, 14B, 72B
- ✅ **Qwen 1.5** - Multiple sizes
- ✅ **Qwen 2** - 0.5B, 1.5B, 7B, 72B

**Key Features**:
- Excellent multilingual support (especially Chinese)
- GQA support in Qwen2
- Long context support (up to 32k tokens)
- RoPE and RMSNorm

**Notes**:
- Qwen 2 significantly improved over original Qwen
- Large vocabulary size (151,936 tokens)

## Architecture Compatibility

All supported architectures share these common components:

### ✅ Shared Features
1. **RoPE** (Rotary Positional Embeddings)
2. **RMSNorm** (Root Mean Square Normalization)
3. **GQA** (Grouped-Query Attention)
4. **Causal Attention** (Decoder-only)
5. **Standard Transformer Blocks**

### Quantization Support

All architectures support the following quantization formats:
- **Q4_K**: 4-bit (144 bytes per 256 elements)
- **Q5_K**: 5-bit (176 bytes per 256 elements)  
- **Q8_0**: 8-bit (34 bytes per 32 elements)
- **F16**: Half precision
- **BF16**: Bra
in floating point
- **F32**: Full precision

## Model Configuration Requirements

Your `config.json` must include:

```json
{
  "architectures": ["LlamaForCausalLM"],  // One of the supported architectures
  "hidden_size": 4096,                     // Model dimension
  "num_hidden_layers": 32,                 // Number of transformer layers
  "num_attention_heads": 32,               // Number of attention heads
  "num_key_value_heads": 32,               // KV heads for GQA (optional, defaults to num_attention_heads)
  "vocab_size": 32000,                     // Vocabulary size
  "intermediate_size": 11008,              // FFN intermediate dimension (optional)
  "max_position_embeddings": 2048,         // Max sequence length (optional)
  "rms_norm_eps": 1e-5                     // RMSNorm epsilon (optional)
}
```

## Loading Models

### From HuggingFace

```bash
# Download a model
huggingface-cli download microsoft/phi-2 --local-dir models/phi-2

# Run inference
janus-engine run models/phi-2 "Your prompt here"
```

### Supported Model Formats

- ✅ **GGUF** (recommended for quantized models)
- ✅ **Safetensors** (recommended for unquantized models)
- ❌ **PyTorch** (.bin, .pt) - not yet supported

## Architecture-Specific Notes

### Phi Models
- Phi-3 supports very long contexts (up to 128k) but may require rope scaling
- Phi models are optimized for code and reasoning tasks

### Gemma Models
- Use aggressive GQA (e.g., 8:1 ratio) for memory efficiency
- Vocabulary size is 256,000 tokens
- Excellent for instruction following

### Qwen Models
- Large vocabulary (151,936 tokens) increases memory usage
- Excellent for multilingual tasks
- Qwen2 significantly outperforms original Qwen

### Mistral Models
- Sliding window attention is treated as standard attention
- Very efficient 7B model competitive with larger models
- GQA with 8 KV heads for efficiency

## Unsupported Features

The following features are not yet supported:

- ❌ **Mixture of Experts (MoE)** - Mixtral will work but won't be optimized
- ❌ **Sliding Window Attention** - Treated as standard attention
- ❌ **Flash Attention** - Uses standard attention implementation
- ❌ **Encoder-Decoder Models** - Only decoder-only supported

## Validation

Janus validates your model configuration on load:

1. ✅ Architecture is in the supported list
2. ✅ `hidden_size % num_attention_heads == 0` (valid head dimension)
3. ✅ `num_attention_heads % num_key_value_heads == 0` (valid GQA configuration)
4. ✅ All dimensions are positive and reasonable
5. ✅ Tensor shapes match configuration

## Performance Notes

### Quantized Inference
- **Q4_K**: ~2-3x faster, ~75% memory reduction, minimal quality loss
- **Q5_K**: ~1.5-2x faster, ~60% memory reduction, better quality than Q4_K
- **Q8_0**: ~1.2-1.5x faster, ~50% memory reduction, excellent quality

### Model Size Recommendations
- **1-3B models**: Q8_0 or F16 for best quality
- **7-13B models**: Q5_K for balanced quality/speed
- **30B+ models**: Q4_K for memory constraints

## Contributing

To request support for a new architecture:

1. Check if it uses RoPE, RMSNorm, and standard transformer blocks
2. Open an issue with the architecture details
3. Provide a link to the model's `config.json`
4. Note any unique architectural features

Most modern decoder-only transformers are compatible!
