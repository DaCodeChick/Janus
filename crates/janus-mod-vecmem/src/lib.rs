use janus_engine::{JanusApp, JanusPlugin};

pub struct VecMemPlugin;

impl JanusPlugin for VecMemPlugin {
    fn build(&self, _app: &mut JanusApp) {}
}
