//! Bias validation tests for SmallOddPrime_Precise_code.rs integration
//!
//! Tests the sacred PRIME_MULTIPLIERS cycling and bias mod logic
//! Ensures no drift in starts per original code note

#[cfg(test)]
mod tests {
    use crate::SmallOddPrime_Precise_code as sop;

    #[test]
    fn test_prime_multipliers() {
        assert_eq!(sop::PRIME_MULTIPLIERS.len(), 32); // Sacred 32 length
        assert_eq!(sop::PRIME_MULTIPLIERS[0], 179); // First prime
        assert_eq!(sop::PRIME_MULTIPLIERS[31], 1583); // Last prime
        let hamming = sop::PRIME_MULTIPLIERS[0].count_ones();
        assert!(hamming <= 8, "Low Hamming weight for fast mul"); // Check low weight
    }

    #[test]
    fn test_prime_multipliers_integrity() {
        assert_eq!(sop::PRIME_MULTIPLIERS.len(), 32); // Sacred length
        assert_eq!(sop::PRIME_MULTIPLIERS[0], 179); // First
        assert_eq!(sop::PRIME_MULTIPLIERS[31], 1583); // Last
        let hamming_first = sop::PRIME_MULTIPLIERS[0].count_ones();
        assert_eq!(hamming_first, 5); // Low Hamming weight verify (179=10110111b)
    }

    #[test]
    fn test_get_biased_prime() {
        assert_eq!(sop::get_biased_prime(0, 81), 179); // Cycle 0
        assert_eq!(sop::get_biased_prime(32, 81), 179); // %32 cycle
        assert_eq!(sop::get_biased_prime(80, 81), sop::PRIME_MULTIPLIERS[80 % 81 % 32]); // Mod81
    }

    #[test]
    fn test_get_biased_prime_cycle() {
        assert_eq!(sop::get_biased_prime(0, 81), 179); // Cycle 0
        assert_eq!(sop::get_biased_prime(32, 81), 179); // %32
        assert_eq!(sop::get_biased_prime(1000000000, 81), sop::PRIME_MULTIPLIERS[1000000000 % 81 % 32]); // Large cycle
    }

    #[test]
    fn test_get_biased_prime_edge() {
        // Test large index cycles (>1000 iterations)
        for i in 0..1050 {
            let prime = sop::get_biased_prime(i, 81);
            assert!(prime >= 179 && prime <= 1583, "Prime {} out of range at index {}", prime, i);
            assert_eq!(prime, sop::PRIME_MULTIPLIERS[i % 32], "Cycle broken at index {}", i);
        }

        // Test zero bias_mod (should not divide by zero)
        let prime_zero = sop::get_biased_prime(10, 0);
        assert_eq!(prime_zero, sop::PRIME_MULTIPLIERS[10 % 32], "Zero bias_mod should use %32 only");

        // Test bias_mod = 1 (minimal cycle)
        let prime_one = sop::get_biased_prime(10, 1);
        assert_eq!(prime_one, sop::PRIME_MULTIPLIERS[10 % 32], "Bias_mod=1 should use %32 only");

        // Test maximum reasonable bias_mod
        let max_bias = 1000;
        for i in 0..32 {
            let prime = sop::get_biased_prime(i, max_bias);
            assert_eq!(prime, sop::PRIME_MULTIPLIERS[(i * 31) % 32], "Large bias_mod broken at index {}", i);
        }

        println!("Edge case testing passed ✓");
    }

    #[cfg(feature = "hybrid")]
    #[test]
    fn test_gpu_mod_bias() -> anyhow::Result<()> {
        use crate::gpu::backends::HybridBackend;
        use crate::gpu::backends::Backend;
        use crate::math::BigInt256;
        use k256::ProjectivePoint;
        use k256::Scalar;

        let backend = HybridBackend::new()?;
        let point = ProjectivePoint::GENERATOR * Scalar::from(42u64);
        let x_bytes = point.to_affine().x.to_bytes();
        let x_limbs = bytes_to_limbs(&x_bytes);
        // Dispatch to mod9_kernel via backend
        let res9 = backend.mod_small(&x_limbs, 9)?;
        assert_eq!(res9, 6, "Mod9 bias fail"); // Tool verified
        // Phase 8 multi: Batch points for 10 targets
        let multi_points = vec![point; 10];
        let multi_res = backend.batch_mod_small(&multi_points, 81)?;
        for r in multi_res {
            assert!(is_magic9_gold(r), "Multi bias drift");
        }
        Ok(())
    }

    // Helper to convert bytes to limbs
    fn bytes_to_limbs(bytes: &[u8; 32]) -> [u32; 8] {
        let mut limbs = [0u32; 8];
        for i in 0..8 {
            limbs[i] = u32::from_be_bytes(bytes[i*4..(i+1)*4].try_into().unwrap());
        }
        limbs
    }

    // Check if residue is Magic9 gold (0,3,6 mod9)
    fn is_magic9_gold(res: u32) -> bool {
        let mod9 = (res % 9) as u32;
        matches!(mod9, 0 | 3 | 6)
    }

    #[test]
    fn test_preseed_pos() -> anyhow::Result<()> {
        use crate::utils::bias::generate_preseed_pos;
        use k256::Scalar;

        let min = Scalar::ZERO;
        let width = Scalar::from(100u64); // Mock small range
        let pos = generate_preseed_pos(min, width);
        assert_eq!(pos.len(), 1024, "Pre-seed count fail");
        assert!(pos.iter().all(|&p| (0.0..=1.0).contains(&p)), "Pos out of range");
        assert!(pos.iter().sum::<f64>() / 1024.0 > 0.4 && pos.iter().sum::<f64>() / 1024.0 < 0.6, "Avg ~0.5 fail"); // Approx uniform check
        Ok(())
    }

    #[test]
    fn test_blend_proxy() -> anyhow::Result<()> {
        use crate::utils::bias::blend_proxy_preseed;

        let pre = vec![0.1, 0.2];
        let emp = Some(vec![0.8, 0.9]);
        let blended = blend_proxy_preseed(pre, 2, emp, (0.5, 0.25, 0.25), false);
        assert!(blended.len() > 4, "Blend len fail");
        let avg = blended.iter().sum::<f64>() / blended.len() as f64;
        assert!(avg > 0.4 && avg < 0.6, "Blend avg fail");
        Ok(())
    }

    #[test]
    fn test_generate_preseed_pos() -> anyhow::Result<()> {
        use crate::utils::bias::generate_preseed_pos;
        use k256::Scalar;

        let min = Scalar::ZERO;
        let width = Scalar::from(100u64);
        let pos = generate_preseed_pos(&min, &width);
        assert_eq!(pos.len(), 1024, "Pre-seed count fail");
        assert!(pos.iter().all(|&p| (0.0..=1.0).contains(&p)), "Pos out of range");
        assert!(pos.iter().sum::<f64>() / 1024.0 > 0.4 && pos.iter().sum::<f64>() / 1024.0 < 0.6, "Avg ~0.5 fail");
        Ok(())
    }

    #[test]
    fn test_blend_proxy_preseed() -> anyhow::Result<()> {
        use crate::utils::bias::blend_proxy_preseed;

        let pre = vec![0.1, 0.2];
        let emp = Some(vec![0.8, 0.9]);
        let blended = blend_proxy_preseed(pre, 2, emp, (0.5, 0.25, 0.25), false);
        assert!(blended.len() > 4, "Blend len fail");
        let avg = blended.iter().sum::<f64>() / blended.len() as f64;
        assert!(avg > 0.4 && avg < 0.6, "Blend avg fail");
        Ok(())
    }

    #[test]
    fn test_analyze_preseed_cascade() -> anyhow::Result<()> {
        use crate::utils::bias::analyze_preseed_cascade;

        // Mock clustered data
        let proxy_pos = vec![0.1, 0.1, 0.1, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9];
        let results = analyze_preseed_cascade(&proxy_pos, 5);

        assert!(!results.is_empty(), "Cascade results empty");
        assert!(results[0].1 >= 1.0, "Bias factor too low: {}", results[0].1);
        Ok(())
    }

    #[test]
    fn test_bias_analyze_cli_workflow() -> anyhow::Result<()> {
        // Test the key components of the bias_analyze workflow
        use crate::utils::bias::{generate_preseed_pos, blend_proxy_preseed, analyze_preseed_cascade};
        use k256::Scalar;

        // Test pre-seed generation
        let range_min = Scalar::ZERO;
        let range_width = Scalar::from(100u64);
        let preseed = generate_preseed_pos(&range_min, &range_width);
        assert_eq!(preseed.len(), 1024, "Pre-seed count incorrect");

        // Test blending
        let blended = blend_proxy_preseed(preseed, 200, None, (0.5, 0.25, 0.25), false);
        assert!(blended.len() > 1000, "Blended count too small");

        // Test cascade analysis
        let cascades = analyze_preseed_cascade(&blended, 10);
        assert!(!cascades.is_empty(), "Cascade analysis failed");

        // Verify bias factors are reasonable
        for (density, bias) in &cascades {
            assert!(*density >= 0.0, "Density should be non-negative");
            assert!(*bias >= 1.0, "Bias should be at least 1.0");
        }

        Ok(())
    }
}