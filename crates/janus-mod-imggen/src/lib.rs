use janus_engine::{JanusApp, JanusPlugin};

pub struct ImgGenPlugin;

impl JanusPlugin for ImgGenPlugin {
    fn build(&self, _app: &mut JanusApp) {}
}
