use janus_engine::{JanusApp, JanusPlugin};

pub struct RpPlugin;

pub type RoleplayPlugin = RpPlugin;

impl JanusPlugin for RpPlugin {
    fn build(&self, _app: &mut JanusApp) {}
}
