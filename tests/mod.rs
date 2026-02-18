#[cfg(test)]
mod gpu_hybrid;
pub use gpu_hybrid::*;
mod bias_validation; // Existing bias detection tests
mod config; // Configuration tests
mod fuzz_targets; // Fuzzing targets
mod kangaroo; // Kangaroo algorithm tests
mod magic9;
mod math; // Math operation tests
mod puzzle; // Existing puzzle validation tests // Magic9 specific tests
