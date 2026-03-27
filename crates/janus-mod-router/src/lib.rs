pub mod router;
pub mod types;

pub use router::{DeterministicRouter, RouterConfig};
pub use types::{RouteDestination, RoutingRequest, SystemState};

use janus_engine::{JanusApp, JanusPlugin};

pub struct RouterPlugin;

impl JanusPlugin for RouterPlugin {
    fn build(&self, _app: &mut JanusApp) {}
}
