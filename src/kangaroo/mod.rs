//! Kangaroo module for Pollard's rho/kangaroo method
//!
//! Contains central orchestrator, tame/wild generation, stepping logic, and collision detection.

pub mod manager;
pub mod generator;
pub mod stepper;
pub mod collision;
pub mod search_config;
pub mod controller;

// Re-export main types
pub use manager::KangarooManager;
pub use generator::{KangarooGenerator, WildKangarooGenerator, TameKangarooGenerator};
pub use stepper::KangarooStepper;
pub use collision::{CollisionDetector, CollisionResult};
pub use search_config::SearchConfig;
pub use controller::{KangarooController, ManagerStats};