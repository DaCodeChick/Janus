use janus_engine::{JanusApp, JanusPlugin};

pub struct InstructPlugin;

impl JanusPlugin for InstructPlugin {
    fn build(&self, _app: &mut JanusApp) {}
}
