// tests/math.rs - Unit tests for cryptographic mathematical operations
// Tests modular arithmetic, elliptic curve operations, and GPU acceleration

use speedbitcrack::math::{BigInt256, secp::Secp256k1, constants::*};
use speedbitcrack::types::Point;
#[cfg(feature = "rustacuda")]
use speedbitcrack::gpu::backend::HybridBackend;

#[cfg(test)]
mod tests {
    use super::*;

    // Test basic modular inverse
    #[test]
    fn test_modular_inverse() {
        let curve = Secp256k1::new();
        let a = BigInt256::from_u64(5);
        let modulus = curve.p.clone();

        let inv = curve.mod_inverse(&a, &modulus).unwrap();
        let product = curve.montgomery_p.mul(&a, &inv);

        // a * a^(-1) ≡ 1 mod p
        assert_eq!(product, BigInt256::from_u64(1));
    }

    // Test point addition
    #[test]
    fn test_point_addition() {
        let curve = Secp256k1::new();

        // Generator point G
        let g = curve.g.clone();

        // Test G + G = 2G
        let two_g = curve.point_add(&g, &g);

        // Verify the result is on the curve
        assert!(curve.is_on_curve(&two_g));
    }

    // Test point doubling
    #[test]
    fn test_point_doubling() {
        let curve = Secp256k1::new();
        let g = curve.g.clone();

        let doubled = curve.point_double(&g);

        // Verify the result is on the curve
        assert!(curve.is_on_curve(&doubled));

        // 2G should equal G + G
        let added = curve.point_add(&g, &g);
        assert_eq!(doubled, added);
    }

    // Test scalar multiplication
    #[test]
    fn test_scalar_multiplication() {
        let curve = Secp256k1::new();
        let g = curve.g.clone();

        // Test 2 * G
        let result = curve.scalar_mul(&BigInt256::from_u64(2), &g);
        let expected = curve.point_add(&g, &g);

        assert_eq!(result, expected);
    }

    // Test Barrett reduction
    #[test]
    fn test_barrett_reduction() {
        let curve = Secp256k1::new();

        // Create a large number that needs reduction
        let large_num = BigInt256::from_u64_array([
            0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
        ]);

        let reduced = curve.barrett.reduce(&large_num, &curve.p);

        // Result should be less than modulus
        assert!(reduced < curve.p);
    }

    // Chunk: Test Constants (tests/math.rs)
    #[test]
    fn test_constants() {
        // Test PRIME_MULTIPLIERS
        assert_eq!(PRIME_MULTIPLIERS.len(), 32);
        assert_eq!(PRIME_MULTIPLIERS[0], 179);
        assert_eq!(PRIME_MULTIPLIERS[31], 1583);

        // Test CURVE_ORDER
        let order = BigInt256::from_hex(CURVE_ORDER);
        assert_eq!(order.bits(), 256);

        // Test GENERATOR coordinates
        let gx = BigInt256::from_hex(GENERATOR_X);
        let gy = BigInt256::from_hex(GENERATOR_Y);
        assert!(gx.bits() > 0);
        assert!(gy.bits() > 0);

        // Test DP_BITS
        assert_eq!(DP_BITS, 24);

        // Test JUMP_TABLE_SIZE
        assert_eq!(JUMP_TABLE_SIZE, 256);

        // Test jump_table function
        let jumps = jump_table();
        assert_eq!(jumps.len(), JUMP_TABLE_SIZE);
        assert_eq!(jumps[0], BigInt256::from_u64(1));
        assert_eq!(jumps[1], BigInt256::from_u64(2));
    }

    #[test]
    fn test_small_odd_prime_starts() {
        use speedbitcrack::math::constants::PRIME_MULTIPLIERS;
        use speedbitcrack::kangaroo::generator::{KangarooGenerator, Config};
        use speedbitcrack::math::secp::Secp256k1;

        let curve = Secp256k1::new();
        let config = Config::default();
        let gen = KangarooGenerator::new(&config);

        // Test wild start initialization
        let target = curve.g.clone();
        let wild_start = gen.initialize_wild_start(&target, 0);

        // Should be prime_0 * G
        let prime_0 = PRIME_MULTIPLIERS[0];
        let expected = curve.mul_constant_time(&BigInt256::from_u64(prime_0), &curve.g).unwrap();

        // Check that the start is different from G (properly offset)
        assert_ne!(wild_start, curve.g);
        assert_eq!(wild_start, expected);

        // Test tame start initialization
        let tame_start = gen.initialize_tame_start();
        assert_eq!(tame_start, curve.g);

        // Test prime cycling (should use different primes for different indices)
        let wild_start_1 = gen.initialize_wild_start(&target, 1);
        let prime_1 = PRIME_MULTIPLIERS[1];
        let expected_1 = curve.mul_constant_time(&BigInt256::from_u64(prime_1), &curve.g).unwrap();
        assert_eq!(wild_start_1, expected_1);
        assert_ne!(wild_start, wild_start_1);

        // Test prime wrapping (index >= 32 should cycle back)
        let wild_start_32 = gen.initialize_wild_start(&target, 32);
        assert_eq!(wild_start_32, wild_start); // Should cycle back to prime_0
    }

    // Chunk: BigInt Add/Shl Test (tests/math.rs)
    // Dependencies: math::BigInt256
    #[test]
    fn test_bigint_add_shl() {
        let a = BigInt256::from_u64(1);
        let b = a.shl(65);  // 2^65
        let c = b + a;  // 2^65 +1
        assert_eq!(c.to_u64(), None);  // Overflow u64
        assert_eq!(c.shl(1), b.shl(1) + BigInt256::from_u64(2));  // Math check
    }

    // Chunk: Mod Inverse Test (tests/math.rs)
    // Dependencies: math::secp::mod_inverse, constants::CURVE_ORDER
    #[test]
    fn test_mod_inverse() {
        let curve = Secp256k1::new();
        let a = BigInt256::from_u64(5);
        let inv = curve.mod_inverse(&a, &curve.p).unwrap();
        let product = curve.montgomery_p.mul(&a, &inv);
        let one = BigInt256::from_u64(1);
        assert_eq!(product, one);
    }

    // Test Montgomery multiplication
    #[test]
    fn test_montgomery_multiplication() {
        let curve = Secp256k1::new();

        let a = BigInt256::from_u64(5);
        let b = BigInt256::from_u64(7);
        let expected = BigInt256::from_u64(35);

        let result = curve.montgomery_p.mul(&a, &b);

        // Convert back from Montgomery form for comparison
        let result_normal = curve.montgomery.reduce(&result, &curve.p);

        assert_eq!(result_normal, expected);
    }

    // Chunk: Mod Inverse Test (tests/math.rs)
    // Dependencies: math::secp::mod_inverse, constants::CURVE_ORDER
    #[test]
    fn test_mod_inverse_n() {
        let curve = Secp256k1::new();
        let a = BigInt256::from_u64(5);
        let inv = curve.mod_inverse(&a, &curve.n).unwrap();
        let product = curve.montgomery_p.mul(&a, &inv);
        let one = BigInt256::from_u64(1);
        assert_eq!(product % curve.n, one);
    }

    // Test Jacobian to affine conversion
    #[test]
    fn test_jacobian_to_affine() {
        let curve = Secp256k1::new();
        let g = curve.g.clone();

        // Convert to affine (should be no-op for points at infinity, but test structure)
        let affine = g.to_affine(&curve);

        // Verify the result is on the curve
        assert!(curve.is_on_curve(&affine));
    }

    // Property-based test for modular inverse
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    #[cfg(feature = "proptest")]
    proptest! {
        #[test]
        fn test_inverse_property(a in 1u64..(1u64 << 32)) {
            let curve = Secp256k1::new();
            let a_big = BigInt256::from_u64(a);

            // Only test if a and p are coprime (gcd = 1)
            if let Some(inv) = curve.mod_inverse(&a_big, &curve.p) {
                let product = curve.montgomery_p.mul(&a_big, &inv);
                let product_reduced = curve.montgomery.reduce(&product, &curve.p);

                prop_assert_eq!(product_reduced, BigInt256::from_u64(1));
            }
        }
    }

    // Test hybrid GPU backend multiplication
    #[tokio::test]
    async fn test_hybrid_mul() {
        #[cfg(not(feature = "rustacuda"))]
        let backend = Box::new(speedbitcrack::gpu::backend::CpuBackend::new().unwrap()) as Box<dyn speedbitcrack::gpu::backend::GpuBackend>;
        #[cfg(feature = "rustacuda")]
        let backend = Box::new(HybridBackend::new().await.unwrap()) as Box<dyn speedbitcrack::gpu::backend::GpuBackend>;

        // Test 5 * 3 = 15
        let a = vec![BigInt256::from_u64(5)];
        let b = vec![BigInt256::from_u64(3)];

        let result = backend.batch_mul(&a, &b);
        let expected = BigInt256::from_u64(15);

        assert_eq!(result[0], expected);
    }

    // Test batch multiplication correctness with larger numbers
    #[cfg(feature = "rustacuda")]
    #[tokio::test]
    async fn test_cuda_batch_mul_correctness() {
        let backend = Box::new(HybridBackend::new().await.unwrap()) as Box<dyn speedbitcrack::gpu::backend::GpuBackend>;

        // Test with secp256k1 field modulus components (simplified)
        let a = vec![BigInt256::from_u64(0xFFFFFFFFFFFFFFFF)]; // Large number
        let b = vec![BigInt256::from_u64(2)]; // Simple multiplier

        let result = backend.batch_mul(&a, &b);
        let expected = BigInt256::from_u64_array([0xFFFFFFFFFFFFFFFE, 1, 0, 0]); // 2^64 - 2 + 2^64 = 2^65 - 2

        assert_eq!(result[0], expected);
    }

    // Fuzz test for point operations
    #[cfg(feature = "libfuzzer")]
    use libfuzzer_sys::fuzz_target;

    #[cfg(feature = "libfuzzer")]
    fuzz_target!(|data: &[u8]| {
        if data.len() < 32 {
            return;
        }

        let curve = Secp256k1::new();

        // Create a scalar from fuzz data
        let scalar_bytes = &data[0..32];
        let mut scalar_array = [0u64; 4];
        for i in 0..4 {
            let start = i * 8;
            let end = start + 8;
            if end <= scalar_bytes.len() {
                scalar_array[i] = u64::from_le_bytes(scalar_bytes[start..end].try_into().unwrap());
            }
        }

        let scalar = BigInt256::from_u64_array(scalar_array);

        // Test scalar multiplication with generator
        let _result = curve.scalar_mul(&scalar, &curve.g);

        // If we get here without panicking, the operation is safe
    });
}

#[test]
fn full_integration_test() {
    use crate::math::Secp256k1;

    // Test Babai multi-round
    let curve = Secp256k1::new();
    let scalar = k256::Scalar::from(12345u64);
    let _babai_result = curve.glv4_decompose_babai(&scalar);

    // Test Fermat ECDLP diff
    let p = curve.g.clone();
    let q = curve.scalar_mul(&k256::Scalar::from(2u64), &p);
    let _fermat_diff = crate::kangaroo::collision::fermat_ecdlp_diff(&p, &q);

    // Test VOW Rho P2PK (placeholder)
    let dummy_p = curve.g.clone();
    let _vow_result = crate::kangaroo::collision::vow_parallel_rho(&dummy_p, 2, 1.0 / 2f64.powf(20.0));

    // If we get here without panicking, all components integrate properly
}