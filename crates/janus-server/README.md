# Janus Chat Server - Quick Start Guide

This guide shows how to run the Janus chat server and interact with it using the OpenAI-compatible API.

## Prerequisites

- A model directory containing:
  - `model.gguf` or `model.safetensors` (model weights)
  - `config.json` (HuggingFace config)
  - `tokenizer.json` (HuggingFace tokenizer)

## Starting the Server

```bash
cargo run --release --example chat_server <model_dir> [--port 8080]
```

Example:
```bash
cargo run --release --example chat_server ~/models/TinyLlama-1.1B-Chat
```

The server will start and display:
```
Starting server on http://0.0.0.0:8080
Chat endpoint: http://0.0.0.0:8080/v1/chat/completions
```

## Testing the API

### Non-Streaming Request

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Streaming Request (Server-Sent Events)

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Tell me a short story."}
    ],
    "max_tokens": 200,
    "temperature": 0.8,
    "stream": true
  }'
```

### Using with OpenAI Python Client

```python
from openai import OpenAI

# Point to local Janus server
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="dummy"  # Not used but required by client
)

# Non-streaming
response = client.chat.completions.create(
    model="model",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    max_tokens=100,
    temperature=0.7
)

print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="model",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a story."}
    ],
    max_tokens=200,
    temperature=0.8,
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='')
```

## Supported Parameters

- `model`: Model identifier (currently ignored, uses loaded model)
- `messages`: Array of chat messages with `role` and `content`
- `max_tokens`: Maximum tokens to generate (default: 128)
- `temperature`: Sampling temperature 0.0-2.0 (default: 0.7)
- `top_p`: Nucleus sampling parameter (default: 0.9)
- `top_k`: Top-k sampling parameter (default: 40)
- `stream`: Enable streaming responses (default: false)
- `stop`: Array of stop strings (template stops added automatically)

## Chat Template Formats

The server automatically detects the chat template format from the model name:

- **ChatML**: Mistral, Hermes, OpenChat → `<|im_start|>role\ncontent<|im_end|>`
- **Llama 3**: Llama-3 models → `<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>`
- **Llama 2**: Llama-2 models → `[INST] <<SYS>>system<</SYS>>user [/INST]`
- **Zephyr**: TinyLlama, Zephyr → `<|role|>\ncontent</s>`
- **Alpaca**: Alpaca models → `### Instruction:\n...\n### Response:`
- **Vicuna**: Vicuna models → `USER: ...\nASSISTANT: ...`

## Stop Tokens

The server automatically uses template-specific stop tokens:

- ChatML: `<|im_end|>`
- Llama 3: `<|eot_id|>`, `<|end_of_text|>`
- Llama 2: `</s>`
- Zephyr: `</s>`
- Alpaca: `###`
- Vicuna: `USER:`

You can also provide custom stop strings via the `stop` parameter.

## Architecture

The implementation consists of:

1. **ChatFormatter** (`janus-engine/src/model/chat_template.rs`): Converts OpenAI-style messages to model-specific prompt formats
2. **Generation Loop** (`janus-engine/src/model/transformer/generation.rs`): Enhanced with stop string detection and streaming callbacks
3. **API Server** (`janus-server/`): Axum-based HTTP server with OpenAI-compatible endpoints
4. **SSE Streaming**: Real-time token streaming using Server-Sent Events

## Next Steps

- Add support for multiple concurrent requests with request queuing
- Implement model info endpoint `/v1/models`
- Add authentication/API key support
- Add rate limiting
- Support for function calling / tools
