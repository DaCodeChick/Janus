//! Core types for routing decisions

/// System state information used for routing decisions
#[derive(Debug, Clone, PartialEq)]
pub struct SystemState {
    /// Available GPU VRAM in bytes
    pub available_vram_bytes: u64,
    /// Total GPU VRAM in bytes
    pub total_vram_bytes: u64,
    /// GPU utilization percentage (0-100)
    pub gpu_utilization: u8,
    /// Whether the local engine is available
    pub local_engine_available: bool,
}

impl SystemState {
    /// Create a new system state
    pub fn new(
        available_vram_bytes: u64,
        total_vram_bytes: u64,
        gpu_utilization: u8,
        local_engine_available: bool,
    ) -> Self {
        Self {
            available_vram_bytes,
            total_vram_bytes,
            gpu_utilization: gpu_utilization.min(100),
            local_engine_available,
        }
    }

    /// Check if GPU is out of VRAM (less than 10% available)
    pub fn is_vram_exhausted(&self) -> bool {
        if self.total_vram_bytes == 0 {
            return true;
        }
        let threshold = self.total_vram_bytes / 10; // 10%
        self.available_vram_bytes < threshold
    }

    /// Get VRAM usage percentage
    pub fn vram_usage_percent(&self) -> u8 {
        if self.total_vram_bytes == 0 {
            return 100;
        }
        let used = self
            .total_vram_bytes
            .saturating_sub(self.available_vram_bytes);
        ((used * 100) / self.total_vram_bytes).min(100) as u8
    }
}

impl Default for SystemState {
    fn default() -> Self {
        Self {
            available_vram_bytes: 8 * 1024 * 1024 * 1024, // 8GB
            total_vram_bytes: 8 * 1024 * 1024 * 1024,
            gpu_utilization: 0,
            local_engine_available: true,
        }
    }
}

/// A routing request containing prompt and context information
#[derive(Debug, Clone)]
pub struct RoutingRequest {
    /// The user's prompt text
    pub prompt: String,
    /// Estimated number of tokens required for context
    pub required_tokens: usize,
    /// Current system state
    pub system_state: SystemState,
}

impl RoutingRequest {
    /// Create a new routing request
    pub fn new(prompt: String, required_tokens: usize, system_state: SystemState) -> Self {
        Self {
            prompt,
            required_tokens,
            system_state,
        }
    }
}

/// Destination for routing a request
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteDestination {
    /// Route to local GGUF inference engine
    LocalEngine,
    /// Route to cloud API (e.g., OpenAI, Anthropic)
    CloudAPI,
}

impl RouteDestination {
    /// Check if this is a local route
    pub fn is_local(&self) -> bool {
        matches!(self, RouteDestination::LocalEngine)
    }

    /// Check if this is a cloud route
    pub fn is_cloud(&self) -> bool {
        matches!(self, RouteDestination::CloudAPI)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_state_vram_exhausted() {
        let state = SystemState::new(
            500 * 1024 * 1024,      // 500MB available
            8 * 1024 * 1024 * 1024, // 8GB total
            50,
            true,
        );
        assert!(
            state.is_vram_exhausted(),
            "500MB of 8GB should be exhausted"
        );
    }

    #[test]
    fn test_system_state_vram_ok() {
        let state = SystemState::new(
            2 * 1024 * 1024 * 1024, // 2GB available
            8 * 1024 * 1024 * 1024, // 8GB total
            50,
            true,
        );
        assert!(!state.is_vram_exhausted(), "2GB of 8GB should be fine");
    }

    #[test]
    fn test_system_state_vram_usage_percent() {
        let state = SystemState::new(
            2 * 1024 * 1024 * 1024, // 2GB available
            8 * 1024 * 1024 * 1024, // 8GB total
            50,
            true,
        );
        assert_eq!(state.vram_usage_percent(), 75, "6GB used of 8GB = 75%");
    }

    #[test]
    fn test_route_destination_checks() {
        assert!(RouteDestination::LocalEngine.is_local());
        assert!(!RouteDestination::LocalEngine.is_cloud());
        assert!(RouteDestination::CloudAPI.is_cloud());
        assert!(!RouteDestination::CloudAPI.is_local());
    }
}
