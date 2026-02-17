#[cfg(test)]
mod gpu_hybrid;
pub use gpu_hybrid::*;
mod puzzle; // Existing puzzle validation tests
mod bias_validation; // Existing bias detection tests
mod math; // Math operation tests
mod config; // Configuration tests
mod kangaroo; // Kangaroo algorithm tests
mod fuzz_targets; // Fuzzing targets
mod magic9; // Magic9 specific tests