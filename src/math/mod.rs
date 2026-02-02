//! Mathematics module for SpeedBitCrack V3
//!
//! Contains elliptic curve operations, modular arithmetic, and cryptographic primitives.

pub mod secp;
pub mod bigint;
pub mod constants;

// Re-export commonly used types
pub use secp::{Secp256k1, Scalar};
pub use bigint::{BigInt256, BarrettReducer, MontgomeryReducer};
pub use constants::*;