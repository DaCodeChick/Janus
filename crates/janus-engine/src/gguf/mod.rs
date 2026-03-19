//! GGUF (GPT-Generated Unified Format) file parser

mod error;
mod parser;
mod types;

pub use error::{GGUFError, Result};
pub use parser::GGUFFile;
pub use types::{GGMLType, MetadataValue, TensorInfo};
