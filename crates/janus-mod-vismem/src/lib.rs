use janus_engine::{JanusApp, JanusPlugin};

pub struct VisMemPlugin;

impl JanusPlugin for VisMemPlugin {
    fn build(&self, _app: &mut JanusApp) {}
}
