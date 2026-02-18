//! Mathematics module for SpeedBitCrack V3
//!
//! Contains elliptic curve operations, modular arithmetic, and cryptographic primitives.

pub mod bigint;
pub mod constants;
pub mod secp;
#[cfg(test)]
pub mod tests;

// Re-export commonly used types
pub use bigint::{BarrettReducer, BigInt256, MontgomeryReducer};
pub use constants::*;
pub use secp::Secp256k1;
