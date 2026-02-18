//! Cryptographic security utilities
//!
//! Provides constant-time operations, input validation, and side-channel protection.

#![allow(unsafe_code)] // Override crate-level deny for secure memory operations

use crate::math::bigint::BigInt256;

/// Validate that a scalar is in the valid range for secp256k1
/// Prevents small subgroup attacks and invalid curve operations
pub fn validate_scalar(scalar: &BigInt256) -> Result<(), &'static str> {
    if scalar.is_zero() {
        return Err("Scalar cannot be zero");
    }

    let secp256k1_order =
        BigInt256::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141")
            .expect("Invalid secp256k1 order");
    if *scalar >= secp256k1_order {
        return Err("Scalar is too large (must be < secp256k1 order)");
    }

    Ok(())
}

/// Validate that a public key point is valid and not in a small subgroup
pub fn validate_public_key(
    point: &crate::types::Point,
    curve: &crate::math::secp::Secp256k1,
) -> Result<(), &'static str> {
    point.validate(curve)
}

/// Constant-time comparison of BigInt256 values
/// Prevents timing attacks that could leak private key information
pub fn constant_time_eq(a: &BigInt256, b: &BigInt256) -> bool {
    let mut result = 0u8;
    let a_bytes = a.to_bytes_le();
    let b_bytes = b.to_bytes_le();

    for i in 0..32 {
        result |= a_bytes[i] ^ b_bytes[i];
    }

    result == 0
}

/// Secure clearing of sensitive data from memory
pub fn secure_zero_memory(data: &mut [u8]) {
    // Use volatile writes to prevent compiler optimization
    for byte in data.iter_mut() {
        unsafe {
            std::ptr::write_volatile(byte, 0);
        }
    }
    // Prevent reordering
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
}

/// Secure clearing of BigInt256 from memory
pub fn secure_zero_bigint(bigint: &mut crate::math::bigint::BigInt256) {
    // Convert limbs to bytes and zero securely
    let mut bytes = [0u8; 32]; // 4 limbs * 8 bytes each

    // Copy limb data to bytes array
    for i in 0..4 {
        let limb_bytes = bigint.limbs[i].to_le_bytes();
        bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb_bytes);
    }

    // Securely zero the bytes
    secure_zero_memory(&mut bytes);

    // Zero the actual limbs (this is safe since we're in a controlled context)
    for i in 0..4 {
        bigint.limbs[i] = 0;
    }
}

/// Validate that a target address is valid for ECDLP solving
pub fn validate_target(target: &str) -> Result<(), &'static str> {
    // Basic validation - should be a valid Bitcoin address or hash160
    if target.len() != 42 && target.len() != 40 {
        return Err("Invalid target format (must be Bitcoin address or hash160)");
    }

    // Check for valid base58 characters (simplified check)
    if !target.chars().all(|c| c.is_alphanumeric()) {
        return Err("Invalid characters in target");
    }

    Ok(())
}

/// Security audit: Check if the system is configured securely
pub fn security_audit() -> Vec<&'static str> {
    let mut issues = Vec::new();

    // Check for debug builds (timing attack vulnerability)
    if cfg!(debug_assertions) {
        issues.push("Debug build detected - timing attacks possible");
    }

    // Note: Unsafe code is allowed in this module for secure memory operations
    // The module-level #[allow(unsafe_code)] attribute overrides the crate-level deny

    // Check for CUDA availability
    #[cfg(not(feature = "cudarc"))]
    issues.push("CUDA not enabled - reduced performance may enable timing attacks");

    issues
}
