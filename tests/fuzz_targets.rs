//! Fuzz tests for cryptographic security
//!
//! These tests use libfuzzer to find edge cases and potential vulnerabilities
//! in the mathematical operations and elliptic curve implementations.

use speedbitcrack::math::{bigint::BigInt256, secp::Secp256k1};
use speedbitcrack::types::Point;

#[cfg(feature = "libfuzzer")]
use libfuzzer_sys::fuzz_target;

#[cfg(feature = "libfuzzer")]
fuzz_target!(|data: &[u8]| {
    if data.len() < 32 {
        return;
    }

    let curve = Secp256k1::new();

    // Test modular inverse with fuzzed input
    let input_bytes = &data[0..32];
    let modulus_bytes = &data[32..64];

    if modulus_bytes.iter().all(|&b| b == 0) {
        return; // Skip zero modulus
    }

    let a = BigInt256::from_bytes_be(input_bytes);
    let modulus = BigInt256::from_bytes_be(modulus_bytes);

    // Test that inverse works correctly when it exists
    if let Some(inv) = curve.mod_inverse(&a, &modulus) {
        let product = curve.montgomery_p.mul(&a, &inv);
        let reduced = curve.barrett_p.reduce(&product);
        assert_eq!(reduced, BigInt256::from_u64(1));
    }
});

#[cfg(feature = "libfuzzer")]
fuzz_target!(|data: &[u8]| {
    if data.len() < 96 {
        return;
    }

    let curve = Secp256k1::new();

    // Test point validation with fuzzed coordinates
    let x_bytes = &data[0..32];
    let y_bytes = &data[32..64];
    let z_bytes = &data[64..96];

    let x = BigInt256::from_bytes_be(x_bytes);
    let y = BigInt256::from_bytes_be(y_bytes);
    let z = BigInt256::from_bytes_be(z_bytes);

    let point = Point {
        x: x.to_u64_array(),
        y: y.to_u64_array(),
        z: z.to_u64_array(),
    };

    // Point validation should not panic
    let _ = point.validate_curve(&curve);
    let _ = point.validate_subgroup(&curve);
});

#[cfg(feature = "libfuzzer")]
fuzz_target!(|data: &[u8]| {
    // Additional fuzz test for comprehensive point validation
    if data.len() < 96 {
        return;
    }

    let curve = Secp256k1::new();

    // Create point from fuzzed data and test all validation aspects
    let x_bytes: [u8; 32] = data[0..32].try_into().unwrap();
    let y_bytes: [u8; 32] = data[32..64].try_into().unwrap();
    let z_bytes: [u8; 32] = data[64..96].try_into().unwrap();

    let x = BigInt256::from_bytes_be(&x_bytes);
    let y = BigInt256::from_bytes_be(&y_bytes);
    let z = BigInt256::from_bytes_be(&z_bytes);

    let point = Point {
        x: x.to_u64_array(),
        y: y.to_u64_array(),
        z: z.to_u64_array(),
    };

    // Test comprehensive validation - should not panic on any input
    let curve_valid = point.validate_curve(&curve);
    let subgroup_valid = point.validate_subgroup(&curve);
    let overall_valid = point.validate(&curve);

    // These should be consistent
    if curve_valid.is_ok() && subgroup_valid.is_ok() {
        assert!(overall_valid.is_ok());
    }
});

#[cfg(feature = "libfuzzer")]
fuzz_target!(|data: &[u8]| {
    if data.len() < 32 {
        return;
    }

    let curve = Secp256k1::new();

    // Test Barrett reduction with fuzzed input
    let input = BigInt256::from_bytes_be(&data[0..32]);

    let reduced = curve.barrett_p.reduce(&input);

    // Reduced value should always be in valid range
    assert!(reduced < curve.p);
    assert!(reduced >= BigInt256::zero());
});

#[cfg(not(feature = "libfuzzer"))]
mod dummy_fuzz {
    // Dummy module when libfuzzer is not available
    #[test]
    fn dummy_fuzz_test() {
        // This test does nothing but ensures the module compiles
        assert!(true);
    }
}