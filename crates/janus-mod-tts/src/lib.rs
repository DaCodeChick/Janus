use janus_engine::{JanusApp, JanusPlugin};

pub struct TtsPlugin;

impl JanusPlugin for TtsPlugin {
    fn build(&self, _app: &mut JanusApp) {}
}
