//! KV cache compression configuration

/// Configuration for KV cache compression
#[derive(Debug, Clone, Copy)]
pub struct CacheCompressionConfig {
    /// Enable compression (default: false)
    pub enabled: bool,

    /// Size of the uncompressed window (recent tokens to keep at full precision)
    /// Default: 512 tokens
    pub uncompressed_window: u32,

    /// Compression ratio for middle range tokens (2 = compress 2:1, 4 = compress 4:1)
    /// Default: 2
    pub compression_ratio: u32,

    /// Trigger compression when cache is this percentage full (0.0-1.0)
    /// Default: 0.8 (80%)
    pub compression_trigger: f32,
}

impl Default for CacheCompressionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            uncompressed_window: 512,
            compression_ratio: 2,
            compression_trigger: 0.8,
        }
    }
}
