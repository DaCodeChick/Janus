use crate::{ComputeEngine, Model};
use anyhow::{Context, Result, bail};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Plugin contract for extending JanusApp.
pub trait JanusPlugin {
    fn build(&self, app: &mut JanusApp);
}

/// Bevy-style application builder for composing Janus systems.
pub struct JanusApp {
    /// GPU/WebGPU runtime metadata captured during initialization.
    pub gpu_info: Option<wgpu::AdapterInfo>,
    /// Shared model reference used by serving layers.
    pub model: Option<Arc<Mutex<Model>>>,
    /// Router under construction by server/plugin layers.
    pub router: Option<axum::Router>,
    /// Network address used by the built-in server runner.
    pub bind_addr: Option<SocketAddr>,
}

impl JanusApp {
    pub fn new() -> Self {
        Self {
            gpu_info: None,
            model: None,
            router: None,
            bind_addr: None,
        }
    }

    pub fn add_plugin<T: JanusPlugin>(&mut self, plugin: T) -> &mut Self {
        plugin.build(self);
        self
    }

    pub fn set_gpu_context(&mut self, engine: &ComputeEngine) -> &mut Self {
        self.gpu_info = Some(engine.adapter_info());
        self
    }

    pub fn set_model(&mut self, model: Arc<Mutex<Model>>) -> &mut Self {
        self.model = Some(model);
        self
    }

    pub fn set_router(&mut self, router: axum::Router) -> &mut Self {
        self.router = Some(router);
        self
    }

    pub fn set_bind_addr(&mut self, addr: SocketAddr) -> &mut Self {
        self.bind_addr = Some(addr);
        self
    }

    pub fn take_router(&mut self) -> Option<axum::Router> {
        self.router.take()
    }

    pub async fn run(&mut self) -> Result<()> {
        let addr = self
            .bind_addr
            .ok_or_else(|| anyhow::anyhow!("bind address was not configured"))?;
        let router = self
            .take_router()
            .ok_or_else(|| anyhow::anyhow!("router was not initialized"))?;

        if self.model.is_none() {
            bail!("model was not initialized");
        }

        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .with_context(|| format!("failed to bind TCP listener on {}", addr))?;
        axum::serve(listener, router)
            .await
            .context("server exited with error")?;

        Ok(())
    }
}

impl Default for JanusApp {
    fn default() -> Self {
        Self::new()
    }
}
