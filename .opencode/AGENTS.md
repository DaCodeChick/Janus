# Agent Optimization Policy

This document defines the code quality standards and optimization policies for AI agents working on the Janus Engine codebase.

## Core Principles

### 1. Refactor
- Continuously improve code structure and readability
- Eliminate technical debt as it's discovered
- Prefer clarity over cleverness
- Keep functions focused on a single responsibility

### 2. Despaghettify
- Remove tangled dependencies and circular references
- Flatten deep nesting where possible
- Extract complex logic into well-named helper functions
- Maintain clear separation of concerns

### 3. Simplify
- Choose the simplest solution that solves the problem
- Avoid over-engineering and premature optimization
- Remove dead code and unused dependencies
- Prefer explicit over implicit behavior

### 4. Split Large Modules
- Break large files (>1000 lines) into focused submodules
- Create clear module boundaries with well-defined interfaces
- Each module should have a single, clear purpose
- Use Rust's module system to organize related functionality

### 5. Consistency
- **Uniform code style**: Follow established patterns in the codebase
- **Dependency minimalism**: Don't use multiple dependencies that do the same thing
- **Exception rule**: Only introduce new dependencies when they're significantly better at handling a specific task
- **Naming conventions**: Follow Rust naming conventions and existing patterns
- **Error handling**: Use consistent error types and patterns across the codebase

### 6. Optimize
- **Const functions**: Mark functions as `const` whenever possible
- **Trait-based conversions**: Always use `From`/`TryFrom`/`Into`/`TryInto` traits instead of custom conversion functions
- **Standard library first**: Use stdlib traits and types before creating custom ones
- **Don't reinvent the wheel**: Check if Rust's standard library or existing dependencies already solve the problem
- **Zero-cost abstractions**: Leverage Rust's type system for compile-time guarantees

### 7. Error Handling
- **Never use `unwrap()` in production code**: Always handle errors explicitly with `?` or `match`
- **Never use `expect()` in production code**: Same as unwrap - use proper error handling
- **Never use `assert!` or `assert_eq!` in production code**: Use proper validation with `Result` returns
- **Exceptions**: `unwrap()`, `expect()`, and `assert!` are acceptable ONLY in:
  - Test code (`#[cfg(test)]` modules)
  - Example code that's demonstrative
  - Internal helper functions where invariants are guaranteed by the type system
- **Prefer `?` operator**: Propagate errors up the call stack for caller handling
- **Provide context**: Use `map_err()` or custom error types to add context to errors
- **Document error conditions**: Clearly specify what errors a function can return

## Examples

### ✅ Good: Proper Error Handling
```rust
pub fn load_model(path: &Path) -> Result<Model> {
    let file = File::open(path)
        .map_err(|e| ModelError::FileOpen(path.to_path_buf(), e))?;
    
    let data = parse_model_data(&file)?;
    
    if data.layers.len() != config.num_layers {
        return Err(ModelError::InvalidLayerCount {
            expected: config.num_layers,
            got: data.layers.len(),
        });
    }
    
    Ok(Model::new(data))
}
```

### ❌ Bad: Using unwrap/assert in Production
```rust
pub fn load_model(path: &Path) -> Model {
    let file = File::open(path).unwrap(); // NEVER DO THIS
    let data = parse_model_data(&file).expect("failed to parse"); // NEVER DO THIS
    assert_eq!(data.layers.len(), config.num_layers); // NEVER DO THIS
    Model::new(data)
}
```

### ✅ Good: Using Standard Traits
```rust
impl TryFrom<u32> for GGMLType {
    type Error = GGUFError;
    
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(GGMLType::F32),
            1 => Ok(GGMLType::F16),
            _ => Err(GGUFError::UnsupportedType(value)),
        }
    }
}
```

### ❌ Bad: Custom Conversion Functions
```rust
impl GGMLType {
    fn from_u32(value: u32) -> Result<Self, GGUFError> {
        // Don't create custom conversion functions
    }
}
```

### ✅ Good: Const Functions
```rust
const fn calculate_buffer_size(rows: u32, cols: u32) -> u64 {
    (rows as u64) * (cols as u64) * std::mem::size_of::<f32>() as u64
}
```

### ✅ Good: Module Organization
```
compute/
├── mod.rs          // Public API and re-exports
├── engine.rs       // GPU initialization
├── ops.rs          // High-level operations
├── cache.rs        // KV cache management
└── shaders/        // WGSL compute shaders
    ├── matmul.wgsl
    ├── attention.wgsl
    └── softmax.wgsl
```

### ❌ Bad: Monolithic Files
```
compute.rs          // 5000 lines, everything mixed together
```

## Enforcement

When reviewing or modifying code:

1. **Check for optimization opportunities**: Can functions be const? Are standard traits being used?
2. **Evaluate complexity**: Is this the simplest solution? Can it be broken down?
3. **Assess consistency**: Does this match existing patterns? Is it introducing redundancy?
4. **Consider modularity**: Should this be in a separate module? Is the file too large?
5. **Audit error handling**: Are there any `unwrap()`, `expect()`, or `assert!` calls in production code?
6. **Verify Result usage**: Do functions that can fail return `Result` instead of panicking?

## Error Handling Checklist

Before committing code, verify:

- [ ] No `unwrap()` in production code (outside `#[cfg(test)]`)
- [ ] No `expect()` in production code (outside `#[cfg(test)]`)
- [ ] No `assert!()` or `assert_eq!()` in production code (outside `#[cfg(test)]`)
- [ ] All fallible operations use `?` or proper `match` handling
- [ ] Functions that can fail return `Result` or `Option`
- [ ] Error messages provide sufficient context for debugging
- [ ] Custom error types are used where appropriate (via `thiserror`)

## Philosophy

> "Perfection is achieved, not when there is nothing more to add, but when there is nothing left to take away."
> — Antoine de Saint-Exupéry

Write code that future maintainers (including AI agents) will thank you for.
