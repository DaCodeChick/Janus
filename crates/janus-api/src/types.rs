use abi_stable::{std_types::RStr, StableAbi};

/// FFI-safe prompt context passed to plugins
#[repr(C)]
#[derive(StableAbi, Clone, Debug)]
pub struct PromptContext {
    /// The user prompt
    pub prompt: RStr<'static>,
    /// System prompt/instructions
    pub system_prompt: RStr<'static>,
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Sampling temperature (0.0-2.0)
    pub temperature: f32,
    /// Top-p nucleus sampling
    pub top_p: f32,
    /// Context window size in tokens
    pub context_size: u32,
    /// JSON-encoded extensible metadata
    pub metadata_json: RStr<'static>,
}

/// FFI-safe response chunk for streaming inference
#[repr(C)]
#[derive(StableAbi, Clone, Debug)]
pub struct ResponseChunk {
    /// Generated text data
    pub data: RStr<'static>,
    /// True if this is the final chunk
    pub is_final: bool,
    /// Number of tokens in this chunk
    pub token_count: u32,
}

/// Plugin capability flags wrapper for FFI safety
#[repr(transparent)]
#[derive(StableAbi, Clone, Copy, Debug, PartialEq, Eq)]
pub struct PluginCapabilities(u32);

impl PluginCapabilities {
    /// Supports streaming token generation
    pub const STREAMING: Self = Self(0b00000001);
    /// Supports JSON mode/structured output
    pub const JSON_MODE: Self = Self(0b00000010);
    /// Supports function/tool calling
    pub const FUNCTION_CALLING: Self = Self(0b00000100);
    /// Supports vision inputs
    pub const VISION: Self = Self(0b00001000);
    /// Supports code execution
    pub const CODE_EXECUTION: Self = Self(0b00010000);
    /// Supports external tool use
    pub const TOOL_USE: Self = Self(0b00100000);

    /// Create empty capability set
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Create capability set with all flags
    pub const fn all() -> Self {
        Self(0b00111111)
    }

    /// Check if capability is set
    pub const fn contains(&self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Combine capabilities
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl std::ops::BitOr for PluginCapabilities {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitAnd for PluginCapabilities {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

/// Routing preference hint from plugin analysis
#[repr(u8)]
#[derive(StableAbi, Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoutingPreference {
    /// Prefer local GGUF inference
    Local = 0,
    /// Prefer cloud API
    Cloud = 1,
    /// No preference, let router decide
    NoPreference = 2,
}

/// Plugin error codes for FFI boundary
#[repr(u8)]
#[derive(StableAbi, Clone, Copy, Debug, PartialEq, Eq)]
pub enum PluginErrorCode {
    Success = 0,
    InvalidInput = 1,
    ModelLoadFailed = 2,
    InferenceFailed = 3,
    OutOfMemory = 4,
    Timeout = 5,
    NetworkError = 6,
    NotImplemented = 7,
}

/// FFI-safe plugin metadata
#[repr(C)]
#[derive(StableAbi, Clone, Debug)]
pub struct PluginInfo {
    /// Plugin name
    pub name: RStr<'static>,
    /// Semantic version
    pub version: RStr<'static>,
    /// Author name
    pub author: RStr<'static>,
    /// Description of plugin purpose
    pub description: RStr<'static>,
    /// Capability flags
    pub capabilities: PluginCapabilities,
    /// Maximum context size supported
    pub max_context_tokens: u32,
    /// Preferred batch size for inference
    pub preferred_batch_size: u32,
}
