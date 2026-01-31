//! Kangaroo module for Pollard's rho/kangaroo method
//!
//! Contains central orchestrator, tame/wild generation, stepping logic, and collision detection.

pub mod manager;
pub mod generator;
pub mod stepper;
pub mod collision;

// Re-export main types
pub use manager::KangarooManager;
pub use generator::{KangarooGenerator, WildKangarooGenerator, TameKangarooGenerator};
pub use stepper::KangarooStepper;
pub use collision::{CollisionDetector, CollisionResult};