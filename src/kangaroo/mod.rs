//! Kangaroo module for Pollard's rho/kangaroo method
//!
//! Contains central orchestrator, tame/wild generation, stepping logic, and collision detection.

pub mod manager;
pub mod generator;
pub mod stepper;
pub mod collision;
pub mod search_config;
pub mod controller;
#[cfg(test)]
pub mod tests;

// Re-export main types
pub use manager::KangarooManager;
pub use generator::KangarooGenerator;
pub use stepper::KangarooStepper;
pub use collision::{CollisionDetector, CollisionResult, vow_parallel_rho, fermat_ecdlp_diff};
pub use search_config::SearchConfig;
pub use controller::{KangarooController, ManagerStats};