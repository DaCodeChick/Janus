use janus_engine::{JanusApp, JanusPlugin};

pub struct LoraPlugin;

impl JanusPlugin for LoraPlugin {
    fn build(&self, _app: &mut JanusApp) {}
}
