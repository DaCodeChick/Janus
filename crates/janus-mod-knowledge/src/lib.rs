use janus_engine::{JanusApp, JanusPlugin};

pub struct KnowledgePlugin;

impl JanusPlugin for KnowledgePlugin {
    fn build(&self, _app: &mut JanusApp) {}
}
