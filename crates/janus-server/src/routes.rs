//! API routes and router configuration

use crate::handlers::{AppState, ChatCompletionHandler};
use axum::{routing::{get, post}, Router, Json, response::Html};
use serde_json::json;
use std::sync::Arc;

/// Root endpoint - displays server info
async fn root() -> Html<&'static str> {
    println!("📄 GET / - Serving API docs page");
    Html(r#"<!DOCTYPE html>
<html>
<head>
    <title>Janus Chat Server</title>
    <style>
        body { font-family: system-ui; max-width: 800px; margin: 40px auto; padding: 0 20px; }
        code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }
        pre { background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }
        h1 { color: #333; }
        .endpoint { margin: 20px 0; padding: 15px; border-left: 4px solid #007bff; background: #f8f9fa; }
        .cta { display: inline-block; margin: 20px 0; padding: 12px 24px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; font-weight: 600; }
        .cta:hover { background: #0056b3; }
    </style>
</head>
<body>
    <h1>🚀 Janus Chat Server</h1>
    <p>OpenAI-compatible chat completion API powered by GPU-accelerated LLM inference.</p>
    
    <a href="/chat" class="cta">💬 Open Chat UI</a>
    
    <h2>Available Endpoints</h2>
    
    <div class="endpoint">
        <h3>GET /chat</h3>
        <p>Interactive web chat interface</p>
        <p><strong>Example:</strong> <a href="/chat">Open Chat UI</a></p>
    </div>

    <div class="endpoint">
        <h3>POST /v1/chat/completions</h3>
        <p>Chat completion endpoint (OpenAI-compatible)</p>
        <p><strong>Example:</strong></p>
        <pre>curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "stream": false
  }'</pre>
    </div>

    <div class="endpoint">
        <h3>GET /health</h3>
        <p>Health check endpoint</p>
        <p><strong>Example:</strong> <a href="/health">GET /health</a></p>
    </div>

    <h2>Streaming</h2>
    <p>Add <code>"stream": true</code> to your request for server-sent events (SSE) streaming:</p>
    <pre>curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'</pre>
</body>
</html>"#)
}

/// Chat UI endpoint - interactive web interface
async fn chat_ui() -> Html<&'static str> {
    println!("💬 GET /chat - Serving chat UI");
    Html(r#"<!DOCTYPE html>
<html>
<head>
    <title>Janus Chat</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: #1a1a1a;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .header {
            background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%);
            color: #e2e8f0;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        .header h1 { font-size: 24px; font-weight: 600; }
        .header p { opacity: 0.8; margin-top: 5px; font-size: 14px; }
        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            max-width: 900px;
            width: 100%;
            margin: 0 auto;
            background: #2d3748;
            box-shadow: 0 0 20px rgba(0,0,0,0.5);
        }
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .message {
            display: flex;
            gap: 12px;
            max-width: 80%;
            animation: slideIn 0.3s ease;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .message.user { align-self: flex-end; flex-direction: row-reverse; }
        .message.assistant { align-self: flex-start; }
        .avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            flex-shrink: 0;
        }
        .message.user .avatar {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .message.assistant .avatar {
            background: linear-gradient(135deg, #38b2ac 0%, #319795 100%);
        }
        .message-content {
            padding: 12px 16px;
            border-radius: 18px;
            line-height: 1.5;
            word-wrap: break-word;
        }
        .message.user .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #ffffff;
        }
        .message.assistant .message-content {
            background: #1a202c;
            color: #e2e8f0;
            border: 1px solid #4a5568;
        }
        .input-area {
            padding: 20px;
            background: #2d3748;
            border-top: 1px solid #4a5568;
        }
        .input-container {
            display: flex;
            gap: 12px;
            align-items: flex-end;
        }
        #messageInput {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #4a5568;
            background: #1a202c;
            color: #e2e8f0;
            border-radius: 24px;
            font-size: 15px;
            font-family: inherit;
            resize: none;
            max-height: 120px;
            transition: border-color 0.2s;
        }
        #messageInput::placeholder {
            color: #718096;
        }
        #messageInput:focus {
            outline: none;
            border-color: #667eea;
        }
        #sendButton {
            padding: 12px 24px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 24px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, opacity 0.2s;
        }
        #sendButton:hover:not(:disabled) {
            transform: translateY(-2px);
        }
        #sendButton:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .typing-indicator {
            display: flex;
            gap: 4px;
            padding: 12px 16px;
        }
        .typing-indicator span {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #718096;
            animation: bounce 1.4s infinite ease-in-out both;
        }
        .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
        .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }
        .error {
            background: #742a2a;
            color: #feb2b2;
            padding: 12px;
            border-radius: 8px;
            margin: 12px 20px;
            border: 1px solid #9b2c2c;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>💬 Janus Chat</h1>
        <p>Powered by GPU-accelerated local LLM inference</p>
    </div>
    
    <div class="chat-container">
        <div class="messages" id="messages"></div>
        
        <div class="input-area">
            <div class="input-container">
                <textarea
                    id="messageInput"
                    placeholder="Type your message..."
                    rows="1"
                ></textarea>
                <button id="sendButton">Send</button>
            </div>
        </div>
    </div>

    <script>
        const messagesContainer = document.getElementById('messages');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const conversationHistory = [];

        // Auto-resize textarea
        messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        });

        // Send on Enter (Shift+Enter for new line)
        messageInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        sendButton.addEventListener('click', sendMessage);

        function addMessage(role, content) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            
            const avatar = document.createElement('div');
            avatar.className = 'avatar';
            avatar.textContent = role === 'user' ? '👤' : '🤖';
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.textContent = content;
            
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(contentDiv);
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
            
            return contentDiv;
        }

        function addTypingIndicator() {
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message assistant';
            messageDiv.id = 'typing';
            
            const avatar = document.createElement('div');
            avatar.className = 'avatar';
            avatar.textContent = '🤖';
            
            const typingDiv = document.createElement('div');
            typingDiv.className = 'message-content';
            typingDiv.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
            
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(typingDiv);
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function removeTypingIndicator() {
            const typing = document.getElementById('typing');
            if (typing) typing.remove();
        }

        function showError(message) {
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error';
            errorDiv.textContent = '❌ ' + message;
            messagesContainer.appendChild(errorDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
            setTimeout(() => errorDiv.remove(), 5000);
        }

        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message || sendButton.disabled) return;

            // Add user message
            addMessage('user', message);
            conversationHistory.push({ role: 'user', content: message });
            
            // Clear input
            messageInput.value = '';
            messageInput.style.height = 'auto';
            
            // Disable input
            sendButton.disabled = true;
            messageInput.disabled = true;
            
            // Show typing indicator
            addTypingIndicator();

            try {
                const response = await fetch('/v1/chat/completions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: 'model',
                        messages: conversationHistory,
                        stream: true,
                        temperature: 0.7,
                        max_tokens: 512
                    })
                });

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                removeTypingIndicator();
                const assistantContent = addMessage('assistant', '');
                let fullResponse = '';

                const reader = response.body.getReader();
                const decoder = new TextDecoder();

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\n');

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const data = line.slice(6);
                            if (data === '[DONE]') continue;
                            
                            try {
                                const json = JSON.parse(data);
                                const content = json.choices?.[0]?.delta?.content;
                                if (content) {
                                    fullResponse += content;
                                    assistantContent.textContent = fullResponse;
                                    messagesContainer.scrollTop = messagesContainer.scrollHeight;
                                }
                            } catch (e) {
                                // Skip invalid JSON
                            }
                        }
                    }
                }

                conversationHistory.push({ role: 'assistant', content: fullResponse });

            } catch (error) {
                removeTypingIndicator();
                showError('Failed to send message: ' + error.message);
                console.error('Error:', error);
            } finally {
                sendButton.disabled = false;
                messageInput.disabled = false;
                messageInput.focus();
            }
        }

        // Focus input on load
        messageInput.focus();
    </script>
</body>
</html>"#)
}

/// Health check endpoint
async fn health(axum::extract::State(state): axum::extract::State<Arc<AppState>>) -> Json<serde_json::Value> {
    println!("💚 GET /health - Health check");
    Json(json!({
        "status": "ok",
        "model": state.model_name,
    }))
}

/// Create the application router with all endpoints
pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(root))
        .route("/chat", get(chat_ui))
        .route("/health", get(health))
        .route("/v1/chat/completions", post(ChatCompletionHandler::handle))
        .with_state(state)
}
