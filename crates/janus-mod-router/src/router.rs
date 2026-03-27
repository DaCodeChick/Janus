//! Deterministic routing logic based on heuristics

use crate::types::{RouteDestination, RoutingRequest};

/// Configuration for the deterministic router
#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// Maximum tokens before forcing cloud API
    pub max_local_tokens: usize,
    /// Keywords that indicate complex tasks requiring cloud API
    pub complexity_keywords: Vec<String>,
    /// Whether to enable complexity checking
    pub enable_complexity_check: bool,
    /// Whether to enable token threshold checking
    pub enable_token_check: bool,
    /// Whether to enable VRAM fallback checking
    pub enable_vram_check: bool,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            max_local_tokens: 8192,
            complexity_keywords: vec![
                "analyze".to_string(),
                "refactor".to_string(),
                "code".to_string(),
                "debug".to_string(),
                "optimize".to_string(),
                "review".to_string(),
                "architect".to_string(),
                "design pattern".to_string(),
                "algorithm".to_string(),
            ],
            enable_complexity_check: true,
            enable_token_check: true,
            enable_vram_check: true,
        }
    }
}

impl RouterConfig {
    /// Create a new router configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum local tokens threshold
    pub fn with_max_local_tokens(mut self, max_tokens: usize) -> Self {
        self.max_local_tokens = max_tokens;
        self
    }

    /// Add a complexity keyword
    pub fn add_complexity_keyword(mut self, keyword: impl Into<String>) -> Self {
        self.complexity_keywords.push(keyword.into());
        self
    }

    /// Set complexity keywords
    pub fn with_complexity_keywords(mut self, keywords: Vec<String>) -> Self {
        self.complexity_keywords = keywords;
        self
    }

    /// Enable or disable complexity checking
    pub fn set_complexity_check(mut self, enabled: bool) -> Self {
        self.enable_complexity_check = enabled;
        self
    }

    /// Enable or disable token checking
    pub fn set_token_check(mut self, enabled: bool) -> Self {
        self.enable_token_check = enabled;
        self
    }

    /// Enable or disable VRAM checking
    pub fn set_vram_check(mut self, enabled: bool) -> Self {
        self.enable_vram_check = enabled;
        self
    }
}

/// Deterministic router that uses heuristics to route requests
pub struct DeterministicRouter {
    config: RouterConfig,
}

impl DeterministicRouter {
    /// Create a new router with default configuration
    pub fn new() -> Self {
        Self {
            config: RouterConfig::default(),
        }
    }

    /// Create a router with custom configuration
    pub fn with_config(config: RouterConfig) -> Self {
        Self { config }
    }

    /// Route a request to either local engine or cloud API
    pub fn route(&self, request: &RoutingRequest) -> RouteDestination {
        if !request.system_state.local_engine_available {
            return RouteDestination::CloudAPI;
        }

        if self.config.enable_vram_check && request.system_state.is_vram_exhausted() {
            return RouteDestination::CloudAPI;
        }

        if self.config.enable_token_check && request.required_tokens > self.config.max_local_tokens
        {
            return RouteDestination::CloudAPI;
        }

        if self.config.enable_complexity_check && self.is_complex_query(&request.prompt) {
            return RouteDestination::CloudAPI;
        }

        RouteDestination::LocalEngine
    }

    /// Check if a prompt contains complexity indicators
    fn is_complex_query(&self, prompt: &str) -> bool {
        let prompt_lower = prompt.to_lowercase();

        self.config
            .complexity_keywords
            .iter()
            .any(|keyword| prompt_lower.contains(&keyword.to_lowercase()))
    }

    /// Get the current configuration
    pub fn config(&self) -> &RouterConfig {
        &self.config
    }
}

impl Default for DeterministicRouter {
    fn default() -> Self {
        Self::new()
    }
}
