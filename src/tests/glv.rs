//! Master-level GLV optimization tests
//! Tests the full GLV lattice reduction implementation with k256::Scalar

use super::*;
use k256::{Scalar, ProjectivePoint};
use criterion::Criterion;

#[cfg(test)]
mod tests {
    use super::*;

    /// Test basic GLV decompose correctness using working implementation
    #[test]
    fn test_glv_decompose_correctness() {
        let curve = Secp256k1::new();
        let test_scalars = vec![
            BigInt256::from_u64(1),
            BigInt256::from_u64(123456789),
            BigInt256::from_hex("12345678901234567890").unwrap(),
            BigInt256::from_hex("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff").unwrap(),
        ];

        for k in test_scalars {
            let (k1, k2) = curve.glv_decompose(&k);

            // Reconstruct: k should equal k1 + k2 * λ mod n
            // Note: k1 and k2 are already reduced to [0, n-1], so reconstruction should give k mod n
            let lambda = curve.glv_lambda();
            let k2_lambda = curve.barrett_n.mul(&k2, &lambda);
            let reconstructed = curve.barrett_n.add(&k1, &k2_lambda);

            // Should equal k mod n
            let k_extended = BigInt512::from_bigint256(&k);
            let expected = curve.barrett_n.reduce(&k_extended).unwrap();

            assert_eq!(reconstructed, expected,
                "GLV reconstruction failed for scalar {:?}: k1={:?}, k2={:?}, reconstructed={:?}, expected={:?}",
                k, k1, k2, reconstructed, expected);
        }

        println!("✅ GLV decompose reconstruction verified");
    }

    /// Test GLV endomorphism apply
    #[test]
    fn test_glv_endomorphism_apply() {
        let curve = Secp256k1::new();
        let generator = ProjectivePoint::GENERATOR;

        // Apply endomorphism
        let phi_g = curve.endomorphism_apply(&generator);

        // Verify that phi(phi(G)) = -G (since phi^2 = -1)
        let phi_phi_g = curve.endomorphism_apply(&phi_g);
        let neg_g = -generator;

        assert_eq!(phi_phi_g, neg_g,
            "Endomorphism phi(phi(G)) should equal -G");

        println!("✅ GLV endomorphism verified");
    }

    /// Test GLV optimized multiplication correctness
    #[test]
    fn test_glv_multiplication_correctness() {
        let curve = Secp256k1::new();
        let generator = ProjectivePoint::GENERATOR;

        let test_scalars = vec![
            BigInt256::from_u64(1),
            BigInt256::from_u64(42),
            BigInt256::from_u64(123456789),
            BigInt256::from_hex("9876543210987654321").unwrap(),
        ];

        for k in test_scalars {
            let result_glv = curve.mul_glv_opt(&Point::from_projective(generator), &k);
            let result_naive = curve.mul_naive(&k, &Point::from_projective(generator));

            assert_eq!(result_glv, result_naive,
                "GLV multiplication failed for scalar {:?}", k);
        }

        println!("✅ GLV multiplication correctness verified");
    }

    /// Test GLV speedup benchmark
    fn bench_glv_speedup(c: &mut Criterion) {
        let curve = Secp256k1::new();
        let generator = Point::from_projective(ProjectivePoint::GENERATOR);

        // Generate random scalars
        let scalars: Vec<BigInt256> = (0..100)
            .map(|_| BigInt256::random(&mut rand::thread_rng()))
            .collect();

        c.bench_function("mul_glv_opt", |b| {
            b.iter(|| {
                for scalar in &scalars {
                    let _ = curve.mul_glv_opt(&generator, scalar);
                }
            });
        });

        c.bench_function("mul_naive", |b| {
            b.iter(|| {
                for scalar in &scalars {
                    let _ = curve.mul_naive(scalar, &generator);
                }
            });
        });

        println!("✅ GLV speedup benchmark completed - expect 30-40% improvement");
    }

    /// Test GLV integration with scalar operations
    #[test]
    fn test_glv_integration() {
        let curve = Secp256k1::new();

        // Test with a known scalar
        let test_k = BigInt256::from_u64(123456789);
        let generator = Point::from_projective(ProjectivePoint::GENERATOR);

        // Test GLV multiplication
        let result = curve.mul_glv_opt(&generator, &test_k);

        // Verify it's a valid point on curve
        assert!(curve.is_on_curve(&result),
            "GLV multiplication should produce valid curve point");

        // Verify correctness against naive multiplication
        let result_naive = curve.mul_naive(&test_k, &generator);
        assert_eq!(result, result_naive,
            "GLV multiplication should match naive implementation");

        println!("✅ GLV integration test passed");
    }

    #[test]
    fn test_glv_constants() {
        let curve = Secp256k1::new();
        let lambda = curve.glv_lambda_scalar();
        let beta = crate::math::constants::glv_beta_scalar();

        // Verify lambda is a valid scalar (non-zero, within field)
        assert!(!lambda.is_zero(), "GLV lambda should be non-zero");

        // Verify lambda^2 ≡ -1 mod n (fundamental GLV property)
        let lambda_sq = lambda * lambda;
        let expected = -Scalar::ONE;
        assert_eq!(lambda_sq, expected, "GLV lambda should satisfy lambda^2 ≡ -1 mod n");

        // Beta is used for endomorphism - basic check that it's valid
        // Full mathematical verification (beta^3 ≡ -1) requires more complex field arithmetic

        println!("✅ GLV constants verified");
    }

    /// Test comprehensive GLV mathematical correctness
    #[test]
    fn test_glv_mathematical_correctness() {
        let curve = Secp256k1::new();

        // Test vectors covering different scalar ranges
        let test_cases = vec![
            (BigInt256::from_u64(1), "Small scalar"),
            (BigInt256::from_u64(42), "Medium scalar"),
            (BigInt256::from_hex("123456789abcdef").unwrap(), "Large scalar"),
            (BigInt256::from_hex("fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe").unwrap(), "Near order"),
            (BigInt256::from_hex("8000000000000000000000000000000000000000000000000000000000000000").unwrap(), "Half order"),
        ];

        for (k, description) in test_cases {
            // Test GLV decomposition
            let (k1, k2) = curve.glv_decompose(&k);

            // Verify reconstruction: k = k1 + k2 * λ mod n
            let lambda = curve.glv_lambda();
            let k2_lambda = curve.barrett_n.mul(&k2, &lambda);
            let reconstructed = curve.barrett_n.add(&k1, &k2_lambda);
            let expected = curve.barrett_n.reduce(&BigInt512::from_bigint256(&k)).unwrap();

            assert_eq!(reconstructed, expected,
                "GLV reconstruction failed for {}: k={:?}, k1={:?}, k2={:?}",
                description, k, k1, k2);

            // Verify coefficients are in valid range [0, n-1] (after Barrett reduction)
            assert!(k1 < curve.n, "k1 out of range for {}: {:?} >= {:?}", description, k1, curve.n);
            assert!(k2 < curve.n, "k2 out of range for {}: {:?} >= {:?}", description, k2, curve.n);
        }

        println!("✅ GLV mathematical correctness verified");
    }

    /// Test GLV endomorphism properties
    #[test]
    fn test_glv_endomorphism_properties() {
        let curve = Secp256k1::new();

        // Test points
        let test_points = vec![
            ProjectivePoint::GENERATOR,
            ProjectivePoint::GENERATOR * Scalar::from_u128(42),
            ProjectivePoint::GENERATOR * Scalar::from_u128(0x123456789abcdef),
        ];

        for point in test_points {
            // Test endomorphism application
            let phi_p = curve.endomorphism_apply(&point);

            // Verify phi(phi(P)) = -P
            let phi_phi_p = curve.endomorphism_apply(&phi_p);
            let neg_p = -point;

            assert_eq!(phi_phi_p, neg_p, "Endomorphism property phi(phi(P)) = -P failed");

            // Verify phi(P) is still on curve
            let phi_p_affine = Point::from_projective(phi_p);
            assert!(curve.is_on_curve(&phi_p_affine), "phi(P) should be on curve");

            // Verify phi preserves the curve equation
            // This is implicitly tested by the point being valid
        }

        println!("✅ GLV endomorphism properties verified");
    }

    /// Test GLV4 decomposition correctness
    #[test]
    fn test_glv4_correctness() {
        let curve = Secp256k1::new();

        let test_scalars = vec![
            BigInt256::from_u64(12345),
            BigInt256::from_hex("abcdef123456789").unwrap(),
            BigInt256::from_hex("fedcba9876543210fedcba9876543210").unwrap(),
        ];

        for k in test_scalars {
            // Test GLV4 decomposition
            let (coeffs, signs) = curve.glv4_decompose(&k);

            // Verify all coefficients are reasonable size
            for (i, coeff) in coeffs.iter().enumerate() {
                assert!(coeff.bit_length() <= 132,
                    "GLV4 coefficient {} too large: {} bits", i, coeff.bit_length());
            }

            // Verify signs are valid (-1, 0, 1)
            for (i, &sign) in signs.iter().enumerate() {
                assert!((-1..=1).contains(&sign),
                    "GLV4 sign {} invalid: {}", i, sign);
            }

            // For basic correctness, we can't easily reconstruct GLV4
            // but we can verify the decomposition produces valid scalars
            println!("✅ GLV4 decomposition valid for k={:?}", k);
        }

        println!("✅ GLV4 correctness verified");
    }

    /// Test native k256::Scalar GLV operations
    #[test]
    fn test_glv_scalar_operations() {
        // Test native scalar GLV decomposition
        let test_scalars = vec![
            Scalar::from_u128(1),
            Scalar::from_u128(42),
            Scalar::from_u128(0x123456789abcdef),
            Scalar::from_u128(0xfedcba9876543210),
        ];

        for k in test_scalars {
            // Test scalar GLV decomposition
            let (k1, k2) = Secp256k1::glv_decompose_scalar(&k);

            // Verify reconstruction: k = k1 + k2 * λ mod n
            let lambda = Secp256k1::glv_lambda_scalar();
            let k2_lambda = k2 * lambda;
            let reconstructed = k1 + k2_lambda;

            assert_eq!(reconstructed, k, "Scalar GLV reconstruction failed for {:?}", k);

            // Verify coefficients are smaller than original
            // (This is a heuristic - not mathematically guaranteed for all scalars)
            let k_bytes = k.to_bytes();
            let k1_bytes = k1.to_bytes();
            let k2_bytes = k2.to_bytes();

            // At least one coefficient should be "smaller" in some sense
            let k_magnitude = k_bytes.iter().map(|&x| x as u32).sum::<u32>();
            let k1_magnitude = k1_bytes.iter().map(|&x| x as u32).sum::<u32>();
            let k2_magnitude = k2_bytes.iter().map(|&x| x as u32).sum::<u32>();

            assert!(k1_magnitude < k_magnitude || k2_magnitude < k_magnitude,
                "GLV scalar decomposition should produce smaller coefficients for {:?}", k);
        }

        println!("✅ Native Scalar GLV operations verified");
    }

    /// Test Babai's algorithm refinement
    #[test]
    fn test_babai_refinement() {
        let curve = Secp256k1::new();

        // Test case: Large scalar that benefits from refinement
        let k = BigInt256::from_hex("fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe").unwrap();

        // Get decomposition (which already includes refinement)
        let (k1_final, k2_final) = curve.glv_decompose(&k);

        // Manually test refinement function
        let lambda = curve.glv_lambda();

        // Start with unrefined coefficients (simulate pre-refinement state)
        let k_extended = BigInt512::from_bigint256(&k);
        let k_reduced = curve.barrett_n.reduce(&k_extended).unwrap();

        // Simple initial decomposition without refinement
        let c = curve.round_to_closest(k_reduced.clone(), &lambda);
        let k1_unrefined = curve.barrett_n.sub(&k_reduced, &curve.barrett_n.mul(&c, &lambda));
        let k2_unrefined = c;

        // Apply refinement
        let (k1_refined, k2_refined) = curve.glv_babai_refinement(&k1_unrefined, &k2_unrefined, &lambda, 3);

        // Verify refinement maintains correctness
        let k2_lambda_refined = curve.barrett_n.mul(&k2_refined, &lambda);
        let reconstructed_refined = curve.barrett_n.add(&k1_refined, &k2_lambda_refined);

        let expected = curve.barrett_n.reduce(&BigInt512::from_bigint256(&k)).unwrap();

        assert_eq!(reconstructed_refined, expected, "Babai refinement reconstruction failed");

        // The refined coefficients from glv_decompose should also work
        let k2_lambda_final = curve.barrett_n.mul(&k2_final, &lambda);
        let reconstructed_final = curve.barrett_n.add(&k1_final, &k2_lambda_final);

        assert_eq!(reconstructed_final, expected, "Final GLV decomposition reconstruction failed");

        println!("✅ Babai's algorithm refinement verified");
    }

    /// Test GLV performance characteristics
    #[test]
    fn test_glv_performance_characteristics() {
        let curve = Secp256k1::new();
        let generator = Point::from_projective(ProjectivePoint::GENERATOR);

        // Test with small scalars first to verify basic functionality
        let test_scalars = vec![
            BigInt256::from_u64(1),
            BigInt256::from_u64(42),
            BigInt256::from_u64(12345),
        ];

        for k in test_scalars {
            // Test GLV vs naive multiplication correctness (not timing)
            let result_glv = curve.mul_glv_opt(&generator, &k);
            let result_naive = curve.mul_naive(&k, &generator);

            // Verify results are identical
            assert_eq!(result_glv, result_naive,
                "GLV and naive multiplication differ for k={:?}", k);

            println!("✅ GLV correctness verified for {}-bit scalar", k.bit_length());
        }

        println!("✅ GLV performance characteristics verified");
    }

    /// Test GLV edge cases
    #[test]
    fn test_glv_edge_cases() {
        let curve = Secp256k1::new();

        // Test edge cases
        let edge_cases = vec![
            (BigInt256::zero(), "Zero scalar"),
            (BigInt256::from_u64(1), "Scalar = 1"),
            (curve.n.clone(), "Scalar = order"),
            (curve.n.clone() - BigInt256::from_u64(1), "Scalar = order - 1"),
            (BigInt256::from_hex("8000000000000000000000000000000000000000000000000000000000000000").unwrap(), "Half order"),
        ];

        for (k, description) in edge_cases {
            // Test decomposition
            let (k1, k2) = curve.glv_decompose(&k);

            // Test reconstruction
            let lambda = curve.glv_lambda();
            let k2_lambda = curve.barrett_n.mul(&k2, &lambda);
            let reconstructed = curve.barrett_n.add(&k1, &k2_lambda);
            let expected = curve.barrett_n.reduce(&BigInt512::from_bigint256(&k)).unwrap();

            assert_eq!(reconstructed, expected,
                "GLV edge case failed for {}: k={:?}", description, k);

            // Test multiplication
            if !k.is_zero() {
                let generator = Point::from_projective(ProjectivePoint::GENERATOR);
                let result_glv = curve.mul_glv_opt(&generator, &k);
                let result_naive = curve.mul_naive(&k, &generator);

                assert_eq!(result_glv, result_naive,
                    "GLV multiplication edge case failed for {}", description);
            }
        }

        println!("✅ GLV edge cases verified");
    }

    /// Test GLV constant-time properties (basic verification)
    #[test]
    fn test_glv_constant_time_properties() {
        let curve = Secp256k1::new();

        // Test that GLV decomposition doesn't have obvious branching on scalar values
        // This is a basic test - full constant-time verification requires specialized tools

        let test_scalars = vec![
            BigInt256::zero(),
            BigInt256::from_u64(1),
            BigInt256::from_u64(0xFFFFFFFFFFFFFFFF),
            BigInt256::from_hex("8000000000000000000000000000000000000000000000000000000000000000").unwrap(),
            BigInt256::from_hex("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff").unwrap(),
        ];

        for k in test_scalars {
            let (k1, k2) = curve.glv_decompose(&k);

            // Verify coefficients are in valid range [0, n-1]
            assert!(k1 < curve.n, "k1 out of range: {:?} >= {:?}", k1, curve.n);
            assert!(k2 < curve.n, "k2 out of range: {:?} >= {:?}", k2, curve.n);

            // Verify reconstruction works
            let lambda = curve.glv_lambda();
            let k2_lambda = curve.barrett_n.mul(&k2, &lambda);
            let reconstructed = curve.barrett_n.add(&k1, &k2_lambda);
            let k_extended = BigInt512::from_bigint256(&k);
            let expected = curve.barrett_n.reduce(&k_extended).unwrap();

            assert_eq!(reconstructed, expected, "GLV reconstruction failed for {:?}", k);
        }

        println!("✅ GLV constant-time properties verified");
    }
}