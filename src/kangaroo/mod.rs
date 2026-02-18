//! Kangaroo module for Pollard's rho/kangaroo method
//!
//! Contains central orchestrator, tame/wild generation, stepping logic, and collision detection.

pub mod collision;
pub mod generator;
pub mod manager;
pub mod search_config;
pub mod stepper; // temporarily disabled
// pub mod controller;
#[cfg(test)]
pub mod tests;

// Re-export main types
pub use collision::{fermat_ecdlp_diff, vow_parallel_rho, CollisionDetector, CollisionResult};
pub use generator::KangarooGenerator;
pub use manager::KangarooManager;
pub use search_config::SearchConfig;
pub use stepper::KangarooStepper;
// pub use controller::{KangarooController, ManagerStats};
