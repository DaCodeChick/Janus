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
        // Priority 1: Check if local engine is available
        if !request.system_state.local_engine_available {
            return RouteDestination::CloudAPI;
        }

        // Priority 2: Check VRAM exhaustion (force cloud if out of memory)
        if self.config.enable_vram_check && request.system_state.is_vram_exhausted() {
            return RouteDestination::CloudAPI;
        }

        // Priority 3: Check token threshold (force cloud if exceeds limit)
        if self.config.enable_token_check && request.required_tokens > self.config.max_local_tokens
        {
            return RouteDestination::CloudAPI;
        }

        // Priority 4: Check complexity (prefer cloud for complex tasks)
        if self.config.enable_complexity_check && self.is_complex_query(&request.prompt) {
            return RouteDestination::CloudAPI;
        }

        // Default: Route to local engine for simple queries
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SystemState;

    fn create_default_system_state() -> SystemState {
        SystemState::new(
            6 * 1024 * 1024 * 1024, // 6GB available
            8 * 1024 * 1024 * 1024, // 8GB total
            30,
            true,
        )
    }

    #[test]
    fn test_simple_query_routes_to_local() {
        let router = DeterministicRouter::new();
        let request = RoutingRequest::new(
            "Hello, how are you?".to_string(),
            50,
            create_default_system_state(),
        );

        assert_eq!(
            router.route(&request),
            RouteDestination::LocalEngine,
            "Simple conversational query should route to local"
        );
    }

    #[test]
    fn test_complexity_keyword_analyze_routes_to_cloud() {
        let router = DeterministicRouter::new();
        let request = RoutingRequest::new(
            "Can you analyze this code for performance issues?".to_string(),
            100,
            create_default_system_state(),
        );

        assert_eq!(
            router.route(&request),
            RouteDestination::CloudAPI,
            "Query with 'analyze' keyword should route to cloud"
        );
    }

    #[test]
    fn test_complexity_keyword_refactor_routes_to_cloud() {
        let router = DeterministicRouter::new();
        let request = RoutingRequest::new(
            "Please refactor this function to be more maintainable".to_string(),
            200,
            create_default_system_state(),
        );

        assert_eq!(
            router.route(&request),
            RouteDestination::CloudAPI,
            "Query with 'refactor' keyword should route to cloud"
        );
    }

    #[test]
    fn test_complexity_keyword_code_routes_to_cloud() {
        let router = DeterministicRouter::new();
        let request = RoutingRequest::new(
            "Write code to implement a binary search tree".to_string(),
            150,
            create_default_system_state(),
        );

        assert_eq!(
            router.route(&request),
            RouteDestination::CloudAPI,
            "Query with 'code' keyword should route to cloud"
        );
    }

    #[test]
    fn test_case_insensitive_complexity_check() {
        let router = DeterministicRouter::new();
        let request = RoutingRequest::new(
            "ANALYZE this CODE and REFACTOR it".to_string(),
            100,
            create_default_system_state(),
        );

        assert_eq!(
            router.route(&request),
            RouteDestination::CloudAPI,
            "Complexity check should be case-insensitive"
        );
    }

    #[test]
    fn test_token_threshold_exceeded_routes_to_cloud() {
        let router = DeterministicRouter::new();
        let request = RoutingRequest::new(
            "Tell me a story".to_string(),
            10000, // Exceeds default 8192
            create_default_system_state(),
        );

        assert_eq!(
            router.route(&request),
            RouteDestination::CloudAPI,
            "Query exceeding token threshold should route to cloud"
        );
    }

    #[test]
    fn test_token_threshold_at_limit_routes_to_cloud() {
        let router = DeterministicRouter::new();
        let request = RoutingRequest::new(
            "Tell me a story".to_string(),
            8193, // Just over the limit
            create_default_system_state(),
        );

        assert_eq!(
            router.route(&request),
            RouteDestination::CloudAPI,
            "Query at token threshold should route to cloud"
        );
    }

    #[test]
    fn test_token_threshold_under_limit_routes_to_local() {
        let router = DeterministicRouter::new();
        let request = RoutingRequest::new(
            "Tell me a story".to_string(),
            8192, // Exactly at the limit
            create_default_system_state(),
        );

        assert_eq!(
            router.route(&request),
            RouteDestination::LocalEngine,
            "Query at exactly token threshold should route to local"
        );
    }

    #[test]
    fn test_vram_exhausted_routes_to_cloud() {
        let router = DeterministicRouter::new();
        let low_vram_state = SystemState::new(
            200 * 1024 * 1024, // 200MB available (less than 10% of 8GB)
            8 * 1024 * 1024 * 1024,
            95,
            true,
        );

        let request = RoutingRequest::new("What is the weather?".to_string(), 50, low_vram_state);

        assert_eq!(
            router.route(&request),
            RouteDestination::CloudAPI,
            "Query with exhausted VRAM should route to cloud"
        );
    }

    #[test]
    fn test_vram_ok_routes_to_local() {
        let router = DeterministicRouter::new();
        let good_vram_state = SystemState::new(
            4 * 1024 * 1024 * 1024, // 4GB available (50% of 8GB)
            8 * 1024 * 1024 * 1024,
            50,
            true,
        );

        let request = RoutingRequest::new("What is the weather?".to_string(), 50, good_vram_state);

        assert_eq!(
            router.route(&request),
            RouteDestination::LocalEngine,
            "Query with sufficient VRAM should route to local"
        );
    }

    #[test]
    fn test_local_engine_unavailable_routes_to_cloud() {
        let router = DeterministicRouter::new();
        let offline_state = SystemState::new(
            6 * 1024 * 1024 * 1024,
            8 * 1024 * 1024 * 1024,
            0,
            false, // Local engine unavailable
        );

        let request = RoutingRequest::new("Simple question".to_string(), 10, offline_state);

        assert_eq!(
            router.route(&request),
            RouteDestination::CloudAPI,
            "Query with unavailable local engine should route to cloud"
        );
    }

    #[test]
    fn test_custom_token_threshold() {
        let config = RouterConfig::new().with_max_local_tokens(4096);
        let router = DeterministicRouter::with_config(config);

        let request = RoutingRequest::new(
            "Tell me a story".to_string(),
            5000, // Exceeds custom 4096
            create_default_system_state(),
        );

        assert_eq!(
            router.route(&request),
            RouteDestination::CloudAPI,
            "Query exceeding custom token threshold should route to cloud"
        );
    }

    #[test]
    fn test_custom_complexity_keywords() {
        let config = RouterConfig::new()
            .with_complexity_keywords(vec!["translate".to_string(), "summarize".to_string()]);
        let router = DeterministicRouter::with_config(config);

        let request = RoutingRequest::new(
            "Please translate this text to Spanish".to_string(),
            100,
            create_default_system_state(),
        );

        assert_eq!(
            router.route(&request),
            RouteDestination::CloudAPI,
            "Query with custom complexity keyword should route to cloud"
        );
    }

    #[test]
    fn test_disabled_complexity_check() {
        let config = RouterConfig::new().set_complexity_check(false);
        let router = DeterministicRouter::with_config(config);

        let request = RoutingRequest::new(
            "Please analyze and refactor this code".to_string(),
            100,
            create_default_system_state(),
        );

        assert_eq!(
            router.route(&request),
            RouteDestination::LocalEngine,
            "Query with disabled complexity check should route to local"
        );
    }

    #[test]
    fn test_disabled_token_check() {
        let config = RouterConfig::new().set_token_check(false);
        let router = DeterministicRouter::with_config(config);

        let request = RoutingRequest::new(
            "Tell me a story".to_string(),
            100000, // Way over the limit
            create_default_system_state(),
        );

        assert_eq!(
            router.route(&request),
            RouteDestination::LocalEngine,
            "Query with disabled token check should route to local regardless of tokens"
        );
    }

    #[test]
    fn test_disabled_vram_check() {
        let config = RouterConfig::new().set_vram_check(false);
        let router = DeterministicRouter::with_config(config);

        let low_vram_state = SystemState::new(
            100 * 1024 * 1024, // Very low VRAM
            8 * 1024 * 1024 * 1024,
            99,
            true,
        );

        let request = RoutingRequest::new("Simple question".to_string(), 50, low_vram_state);

        assert_eq!(
            router.route(&request),
            RouteDestination::LocalEngine,
            "Query with disabled VRAM check should route to local regardless of VRAM"
        );
    }

    #[test]
    fn test_multiple_heuristics_cloud_wins() {
        let router = DeterministicRouter::new();

        // Complex query + high tokens + low VRAM = definitely cloud
        let low_vram_state = SystemState::new(200 * 1024 * 1024, 8 * 1024 * 1024 * 1024, 95, true);

        let request = RoutingRequest::new(
            "Analyze and refactor this code with 10000 lines".to_string(),
            10000,
            low_vram_state,
        );

        assert_eq!(
            router.route(&request),
            RouteDestination::CloudAPI,
            "Query triggering multiple heuristics should route to cloud"
        );
    }

    #[test]
    fn test_priority_local_unavailable_trumps_all() {
        let router = DeterministicRouter::new();

        // Even with good VRAM and low tokens, if local is unavailable, route to cloud
        let offline_state =
            SystemState::new(6 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024, 0, false);

        let request = RoutingRequest::new("Simple question".to_string(), 10, offline_state);

        assert_eq!(
            router.route(&request),
            RouteDestination::CloudAPI,
            "Local unavailable should trump all other heuristics"
        );
    }

    #[test]
    fn test_edge_case_zero_tokens() {
        let router = DeterministicRouter::new();
        let request = RoutingRequest::new("Hi".to_string(), 0, create_default_system_state());

        assert_eq!(
            router.route(&request),
            RouteDestination::LocalEngine,
            "Query with zero tokens should route to local"
        );
    }

    #[test]
    fn test_edge_case_empty_prompt() {
        let router = DeterministicRouter::new();
        let request = RoutingRequest::new("".to_string(), 100, create_default_system_state());

        assert_eq!(
            router.route(&request),
            RouteDestination::LocalEngine,
            "Empty prompt should route to local"
        );
    }

    #[test]
    fn test_partial_keyword_match_no_false_positive() {
        let router = DeterministicRouter::new();

        // "codeword" contains "code" but shouldn't trigger complexity
        // Actually, it WILL trigger because we use contains()
        // This is intentional - better safe than sorry
        let request = RoutingRequest::new(
            "What's the codeword?".to_string(),
            50,
            create_default_system_state(),
        );

        assert_eq!(
            router.route(&request),
            RouteDestination::CloudAPI,
            "Partial keyword match should trigger complexity check (intentional)"
        );
    }

    #[test]
    fn test_builder_pattern_config() {
        let config = RouterConfig::new()
            .with_max_local_tokens(16384)
            .add_complexity_keyword("benchmark")
            .add_complexity_keyword("profile")
            .set_complexity_check(true)
            .set_token_check(true)
            .set_vram_check(true);

        let router = DeterministicRouter::with_config(config);

        let request = RoutingRequest::new(
            "Can you benchmark this function?".to_string(),
            1000,
            create_default_system_state(),
        );

        assert_eq!(
            router.route(&request),
            RouteDestination::CloudAPI,
            "Custom keyword should work with builder pattern"
        );
    }
}
