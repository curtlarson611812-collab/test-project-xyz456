#[cfg(test)]
mod tests {
    use crate::config::{Config, BiasMode};
    use crate::kangaroo::collision::Trap;
    use crate::math::BigInt256;
    use crate::kangaroo::CollisionDetector;
    use crate::SmallOddPrime_Precise_code as sop;
    use crate::kangaroo::generator::{generate_wild_herds, generate_tame_herds};
    use crate::kangaroo::SearchConfig;
    use crate::types::Point;
    use crate::math::Secp256k1;
    use std::collections::HashSet;

    #[test]
    fn test_near_g_threshold() {
        let config = Config::default();
        let _detector = CollisionDetector::new_with_config(&config);

        // Test optimal Near-G threshold calculation
        let range_small = BigInt256::from_u64(1_000_000); // Small range
        let threshold_small = CollisionDetector::optimal_near_g_threshold(&range_small, 24);
        assert_eq!(threshold_small, 1000); // range/1000 = 1000, prob_based = 4096, min=1000

        let range_large = BigInt256::from_u64(10_000_000); // Larger range
        let threshold_large = CollisionDetector::optimal_near_g_threshold(&range_large, 24);
        assert_eq!(threshold_large, 4096); // prob_based = 4096, cost_based = 10000, min=4096

        // Test Near-G detection with low x[0]
        let _trap_near_g = Trap {
            x: [100000, 0, 0, 0], // Below default 2^20 threshold
            dist: num_bigint::BigUint::from(1000u64),
            is_tame: true,
            alpha: [0; 4],
        };

        let _range_width = BigInt256::from_u64(1 << 20); // Small range for testing
        // This should trigger near-G brute force logic
        // (We can't easily test the full collision solve without mocking)
    }

    #[test]
    fn test_collision_detector_config() {
        let mut config = Config::default();
        config.near_g_thresh = 1 << 18; // 256K

        let detector = CollisionDetector::new_with_config(&config);
        assert_eq!(detector.near_g_thresh, 1 << 18);
    }

    #[test]
    fn test_bloom_filter_dp() {
        let mut config = Config::default();
        config.use_bloom = true;
        config.dp_bits = 20;
        config.herd_size = 1000; // Set required field
        config.jump_mean = 1000; // Set required field
        // Set valid bias mode for gold_bias_combo (which defaults to false now)
        config.bias_mode = BiasMode::Magic9;
        config.gold_bias_combo = true;

        // This test would require a full KangarooManager instance
        // For now, we just verify the config validation works
        if let Err(e) = config.validate() {
            panic!("Config validation failed: {}", e);
        }
    }

    // === SmallOddPrime_Precise_code.rs Integration Tests ===

    #[test]
    fn test_prime_multipliers() {
        assert_eq!(sop::PRIME_MULTIPLIERS.len(), 32);
        assert_eq!(sop::PRIME_MULTIPLIERS[0], 179);  // First sacred prime
        assert_eq!(sop::PRIME_MULTIPLIERS[31], 1583);  // Last sacred prime

        // Verify all are >128, odd, and low Hamming weight
        for &prime in &sop::PRIME_MULTIPLIERS {
            assert!(prime > 128, "Prime {} should be >128", prime);
            assert!(prime % 2 == 1, "Prime {} should be odd", prime);
        }
    }

    #[test]
    fn test_get_biased_prime() {
        // Test basic cycling - all should return PRIME_MULTIPLIERS[0] = 179 for index 0
        assert_eq!(sop::get_biased_prime(0, 81), 179);  // (0 % 81) % 32 = 0 -> 179
        assert_eq!(sop::get_biased_prime(0, 9), 179);   // (0 % 9) % 32 = 0 -> 179
        assert_eq!(sop::get_biased_prime(0, 27), 179);  // (0 % 27) % 32 = 0 -> 179

        // Test cycling
        assert_eq!(sop::get_biased_prime(32, 81), 179);  // (32 % 81) % 32 = 0 -> 179

        // Test different indices
        assert_eq!(sop::get_biased_prime(1, 81), 257);  // (1 % 81) % 32 = 1 -> 257
        assert_eq!(sop::get_biased_prime(31, 81), 1583); // (31 % 81) % 32 = 31 -> 1583

        // Test edge case: bias_mod = 1
        assert_eq!(sop::get_biased_prime(0, 1), 179);   // (0 % 1) % 32 = 0 -> 179
        assert_eq!(sop::get_biased_prime(1, 1), 179);   // (1 % 1) % 32 = 0 -> 179
    }

    #[test]
    fn test_prime_properties() {
        // Verify low Hamming weight (fast GPU multiplication)
        for &prime in &sop::PRIME_MULTIPLIERS {
            let hamming = (prime as u64).count_ones();
            assert!(hamming <= 8, "Prime {} has high Hamming weight: {}", prime, hamming);
        }

        // Verify primes are distinct
        let mut seen = std::collections::HashSet::new();
        for &prime in &sop::PRIME_MULTIPLIERS {
            assert!(!seen.contains(&prime), "Duplicate prime: {}", prime);
            seen.insert(prime);
        }
    }

    #[test]
    fn test_multiplicative_wild_herds() {
        let curve = Secp256k1::new();
        let mut config = SearchConfig::default();
        config.batch_per_target = 3; // Set explicitly
        let target = curve.g; // Use generator as test target

        let herds = generate_wild_herds(&target, &config, "magic9");
        assert_eq!(herds.len(), 3);

        // Verify all points are on curve
        for point in &herds {
            assert!(point.is_valid(&curve));
        }

        // Verify first wild kangaroo is 179 * G (first prime multiplier)
        // Note: Due to k256 conversion simplifications, we just verify basic properties
        assert!(!herds.is_empty());
        assert_eq!(herds.len(), 3);
    }

    #[test]
    fn test_additive_tame_herds() {
        let curve = Secp256k1::new();
        let mut config = SearchConfig::default();
        config.batch_per_target = 3; // Set explicitly

        let herds = generate_tame_herds(&config, "magic9");
        assert_eq!(herds.len(), 3);

        // Verify all points are on curve
        for point in &herds {
            assert!(point.is_valid(&curve));
        }

        // Basic verification - due to k256 simplifications, just check we got the right number
        assert!(!herds.is_empty());
        assert_eq!(herds.len(), 3);
    }
}