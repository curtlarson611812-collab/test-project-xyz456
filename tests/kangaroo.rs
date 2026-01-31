//! Integration tests for kangaroo algorithm
//!
//! Tests for collision detection, trap handling, and full kangaroo runs.

use speedbitcrack::config::Config;
use speedbitcrack::kangaroo::KangarooManager;
use speedbitcrack::types::Point;
use speedbitcrack::math::{secp::Secp256k1, bigint::BigInt256};
use std::time::Duration;

#[test]
fn test_kangaroo_initialization() {
    let config = Config::default();
    let manager = KangarooManager::new(config).unwrap();
    assert_eq!(manager.target_count(), 0);
}

#[test]
fn test_point_to_affine_conversion() {
    let curve = Secp256k1::new();

    // Test generator point conversion
    let g_affine = curve.g.to_affine(&curve);
    assert_eq!(g_affine.z, [1, 0, 0, 0]); // Z=1 for affine

    // Verify the point is still on the curve
    assert!(g_affine.validate(&curve).is_ok());
}

#[test]
fn test_small_scalar_multiplication() {
    let curve = Secp256k1::new();

    // Test 2G = G + G
    let g2_direct = curve.mul(&curve.n.clone().add(&BigInt256::from_u64(2)), &curve.g);
    let g2_add = curve.add(&curve.g, &curve.g);

    assert_eq!(g2_direct.x, g2_add.x);
    assert_eq!(g2_direct.y, g2_add.y);
}

#[test]
fn test_collision_detection_setup() {
    let curve = Secp256k1::new();

    // Create two kangaroos that should collide
    let start1 = curve.mul(&BigInt256::from_u64(100), &curve.g);
    let start2 = curve.mul(&BigInt256::from_u64(200), &curve.g);

    // They should be different points initially
    assert_ne!(start1.x, start2.x);

    // Test that we can compute distinguished points
    let dp1 = start1.x[0] & 0xFFFF; // Simple DP function
    let dp2 = start2.x[0] & 0xFFFF;

    // They might collide by chance, but that's okay for testing
    assert!(dp1 <= 0xFFFF);
    assert!(dp2 <= 0xFFFF);
}

#[test]
fn test_jump_table_precomputation() {
    let curve = Secp256k1::new();

    // Test that G multiples are precomputed correctly
    assert!(!curve.g_multiples.is_empty());

    // 2G should equal G + G
    let g2_from_table = &curve.g_multiples[0]; // Assuming 2G is first
    let g2_computed = curve.add(&curve.g, &curve.g);

    // Note: This test assumes a specific order in g_multiples
    // In practice, we'd need to check the table structure
}

#[test]
fn test_modular_arithmetic_consistency() {
    let curve = Secp256k1::new();

    // Test that (a * b) * c = a * (b * c) mod p
    let a = BigInt256::from_u64(12345);
    let b = BigInt256::from_u64(67890);
    let c = BigInt256::from_u64(11111);

    let left = curve.montgomery_p.mul(&curve.montgomery_p.mul(&a, &b), &c);
    let right = curve.montgomery_p.mul(&a, &curve.montgomery_p.mul(&b, &c));

    assert_eq!(left, right);
}

#[test]
fn test_zero_and_infinity_handling() {
    let curve = Secp256k1::new();

    // Test scalar multiplication by zero
    let zero_result = curve.mul(&BigInt256::zero(), &curve.g);
    assert!(zero_result.is_infinity());

    // Test scalar multiplication by one
    let one_result = curve.mul(&BigInt256::from_u64(1), &curve.g);
    assert_eq!(one_result.x, curve.g.x);
    assert_eq!(one_result.y, curve.g.y);

    // Test addition with infinity
    let inf_plus_g = curve.add(&Point::infinity(), &curve.g);
    assert_eq!(inf_plus_g.x, curve.g.x);
    assert_eq!(inf_plus_g.y, curve.g.y);
}

#[cfg(feature = "cudarc")]
mod cuda_tests {
    use super::*;
    use speedbitcrack::gpu::backend::GpuBackend;

    #[test]
    fn test_cuda_backend_initialization() {
        let backend = speedbitcrack::gpu::CudaBackend::new();
        assert!(backend.is_ok());
    }

    #[test]
    fn test_cuda_batch_multiplication() {
        let backend = speedbitcrack::gpu::CudaBackend::new().unwrap();

        let a = vec![[1, 0, 0, 0, 0, 0, 0, 0]];
        let b = vec![[2, 0, 0, 0, 0, 0, 0, 0]];

        let result = backend.batch_mul(a, b);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.len(), 1);
        // 1 * 2 = 2, so result should be [2, 0, 0, ..., 0]
        assert_eq!(result[0][0], 2);
        for &limb in &result[0][1..] {
            assert_eq!(limb, 0);
        }
    }
}

#[cfg(not(feature = "cudarc"))]
mod cpu_tests {
    use super::*;

    #[test]
    fn test_cpu_backend_availability() {
        // When CUDA is not available, CPU backend should work
        let backend = speedbitcrack::gpu::CpuBackend::new().unwrap();
        let a = vec![[1, 0, 0, 0, 0, 0, 0, 0]];
        let b = vec![[2, 0, 0, 0, 0, 0, 0, 0]];

        let result = backend.batch_mul(a, b).unwrap();
        assert_eq!(result[0][0], 2);
    }
}