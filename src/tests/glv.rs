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
            let lambda = curve.glv_lambda();
            let k2_lambda = curve.barrett_n.mul(&k2, &lambda);
            let reconstructed = curve.barrett_n.add(&k1, &k2_lambda);

            // Should equal k mod n
            let expected = curve.barrett_n.reduce(&k);

            assert_eq!(reconstructed, expected,
                "GLV reconstruction failed for scalar {:?}", k);
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

        // Verify lambda is a valid scalar (non-zero, within field)
        assert!(!lambda.is_zero(),
            "GLV lambda should be non-zero");

        // Verify lambda^2 ≡ -1 mod n (fundamental GLV property)
        let lambda_sq = lambda * lambda;
        let expected = -Scalar::ONE;
        assert_eq!(lambda_sq, expected,
            "GLV lambda should satisfy lambda^2 ≡ -1 mod n");

        println!("✅ GLV constants verified");
    }
}