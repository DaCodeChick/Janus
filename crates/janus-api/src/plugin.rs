// Allow non_local_definitions for abi_stable macro-generated code
#![allow(non_local_definitions)]

use crate::callback::*;
use crate::types::*;
use abi_stable::{
    sabi_trait,
    std_types::{RResult, RStr},
};

/// The universal plugin interface for Janus
///
/// Plugins implement this trait to provide inference capabilities.
/// Uses `abi_stable`'s trait object system for maximum FFI safety.
#[sabi_trait]
pub trait JanusPlugin: Send + Sync {
    /// Initialize the plugin with configuration
    ///
    /// # Arguments
    /// - `config_json`: JSON-encoded configuration string
    ///
    /// # Returns
    /// - `Ok(())` on success
    /// - `Err(PluginErrorCode)` on failure
    fn init(&mut self, config_json: RStr<'_>) -> RResult<(), PluginErrorCode>;

    /// Get plugin metadata
    fn info(&self) -> PluginInfo;

    /// Analyze prompt and return routing preference
    ///
    /// This allows the plugin to hint whether the prompt should be
    /// handled locally or routed to a cloud API based on complexity.
    fn analyze(&self, context: &PromptContext) -> RoutingPreference;

    /// Execute streaming inference with callback
    ///
    /// # Arguments
    /// - `context`: The prompt and generation parameters
    /// - `callback`: Function called for each generated chunk
    ///
    /// # Returns
    /// - `Ok(())` if generation completed successfully
    /// - `Err(PluginErrorCode)` on error
    fn infer_stream(
        &mut self,
        context: &PromptContext,
        callback: StreamCallback,
    ) -> RResult<(), PluginErrorCode>;

    /// Execute blocking inference (non-streaming)
    ///
    /// # Arguments
    /// - `context`: The prompt and generation parameters
    ///
    /// # Returns
    /// - `Ok(text)` with the complete generated response
    /// - `Err(PluginErrorCode)` on error
    fn infer_blocking(
        &mut self,
        context: &PromptContext,
    ) -> RResult<RStr<'static>, PluginErrorCode>;

    /// Shutdown and cleanup resources
    fn shutdown(&mut self);
}
