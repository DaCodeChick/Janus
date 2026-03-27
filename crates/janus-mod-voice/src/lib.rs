use janus_engine::{JanusApp, JanusPlugin};

pub struct VoicePlugin;

impl JanusPlugin for VoicePlugin {
    fn build(&self, _app: &mut JanusApp) {}
}
