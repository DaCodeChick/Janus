use janus_engine::{JanusApp, JanusPlugin};

pub struct VisionPlugin;

impl JanusPlugin for VisionPlugin {
    fn build(&self, _app: &mut JanusApp) {}
}
