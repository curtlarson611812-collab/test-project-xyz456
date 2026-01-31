//! Comprehensive mathematical correctness tests
//!
//! Tests for BigInt256 operations, secp256k1 curve operations,
//! modular arithmetic, and cryptographic primitives.

use speedbitcrack::math::{bigint::BigInt256, secp::Secp256k1};
use speedbitcrack::types::Point;

#[test]
fn test_bigint256_from_hex() {
    let hex = "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F";
    let bigint = BigInt256::from_hex(hex);
    assert_eq!(bigint.to_hex().to_uppercase(), hex);
}

#[test]
fn test_bigint256_arithmetic() {
    let a = BigInt256::from_u64(12345);
    let b = BigInt256::from_u64(67890);

    let sum = a.add(&b);
    let expected_sum = BigInt256::from_u64(80235);
    assert_eq!(sum, expected_sum);

    let product = a.mul(&b);
    let expected_product = BigInt256::from_u64(12345 * 67890);
    assert_eq!(product, expected_product);
}

#[test]
fn test_modular_inverse() {
    let curve = Secp256k1::new();
    let a = BigInt256::from_u64(42);
    let modulus = curve.n; // secp256k1 order

    let inv = curve.mod_inverse(&a, &modulus);
    assert!(inv.is_some());

    let inv = inv.unwrap();
    let product = curve.montgomery_p.mul(&a, &inv);
    let reduced = curve.barrett_n.reduce(&product);
    assert_eq!(reduced, BigInt256::from_u64(1));
}

#[test]
fn test_point_validation() {
    let curve = Secp256k1::new();

    // Test generator point validation
    assert!(curve.g.validate_curve(&curve));
    assert!(curve.g.validate_subgroup(&curve));
    assert!(curve.g.validate(&curve).is_ok());

    // Test point at infinity
    let infinity = Point::infinity();
    assert!(infinity.validate(&curve).is_ok());
}

#[test]
fn test_point_operations() {
    let curve = Secp256k1::new();

    // Test point doubling: 2G should be on curve
    let double_g = curve.double(&curve.g);
    assert!(double_g.validate(&curve).is_ok());

    // Test point addition: G + G = 2G
    let add_gg = curve.add(&curve.g, &curve.g);
    assert_eq!(add_gg.x, double_g.x);
    assert_eq!(add_gg.y, double_g.y);
}

#[test]
fn test_scalar_multiplication() {
    let curve = Secp256k1::new();

    // Test multiplication by small scalars
    let two_g = curve.mul(&BigInt256::from_u64(2), &curve.g);
    let double_g = curve.double(&curve.g);
    assert_eq!(two_g.x, double_g.x);
    assert_eq!(two_g.y, double_g.y);

    // Test multiplication by zero: 0*G = infinity
    let zero_g = curve.mul(&BigInt256::zero(), &curve.g);
    assert!(zero_g.is_infinity());
}

#[test]
fn test_barrett_reduction() {
    let curve = Secp256k1::new();
    let modulus = curve.p;

    // Test reduction of large number
    let large = BigInt256::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF");
    let reduced = curve.barrett_p.reduce(&large);

    // Reduced value should be less than modulus
    assert!(reduced < modulus);
    assert!(reduced >= BigInt256::zero());
}

#[test]
fn test_montgomery_reduction() {
    let curve = Secp256k1::new();

    // Test Montgomery multiplication: a * b * R^-1 mod p
    let a = BigInt256::from_u64(12345);
    let b = BigInt256::from_u64(67890);

    let mont_product = curve.montgomery_p.mul(&a, &b);

    // Convert back to normal representation
    let normal_product = curve.montgomery_p.to_normal(&mont_product);

    // Should equal regular multiplication modulo p
    let expected = a.mul(&b);
    let expected_reduced = curve.barrett_p.reduce(&expected);

    assert_eq!(normal_product, expected_reduced);
}

#[test]
fn test_curve_parameters() {
    let curve = Secp256k1::new();

    // Verify curve equation: y² = x³ + ax + b mod p for generator point
    let g_affine = curve.g.to_affine(&curve);
    let x = BigInt256::from_u64_array(g_affine.x);
    let y = BigInt256::from_u64_array(g_affine.y);

    let y_squared = curve.montgomery_p.mul(&y, &y);
    let x_squared = curve.montgomery_p.mul(&x, &x);
    let x_cubed = curve.montgomery_p.mul(&x_squared, &x);
    let ax = curve.montgomery_p.mul(&curve.a, &x); // a = 0, so ax = 0
    let rhs = curve.montgomery_p.add(&x_cubed, &ax);
    let rhs = curve.montgomery_p.add(&rhs, &curve.b);

    assert_eq!(y_squared, rhs);
}

#[test]
fn test_invalid_inputs() {
    let curve = Secp256k1::new();

    // Test modular inverse with invalid inputs
    assert!(curve.mod_inverse(&BigInt256::zero(), &curve.n).is_none());
    assert!(curve.mod_inverse(&BigInt256::from_u64(1), &BigInt256::zero()).is_none());

    // Test point validation with invalid points
    let invalid_point = Point {
        x: [0; 4],
        y: [0; 4],
        z: [1, 0, 0, 0],
    };
    assert!(invalid_point.validate(&curve).is_err());
}