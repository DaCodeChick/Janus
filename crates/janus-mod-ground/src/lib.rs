use janus_engine::{JanusApp, JanusPlugin};

pub struct GroundPlugin;

impl JanusPlugin for GroundPlugin {
    fn build(&self, _app: &mut JanusApp) {}
}
