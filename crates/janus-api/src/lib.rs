//! Janus Plugin API
//!
//! FFI-safe interface for runtime-loadable inference plugins using `abi_stable`.
//! This crate defines the contract between the Janus engine core and its plugins.

pub mod types;
pub mod callback;
pub mod plugin;

pub use types::*;
pub use callback::*;
pub use plugin::*;

// Re-export abi_stable for plugin implementors
pub use abi_stable;
