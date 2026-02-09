#[cfg(test)]
mod tests {
    use crate::config::{Config, BiasMode};
    use crate::kangaroo::collision::Trap;
    use crate::math::BigInt256;
    use crate::kangaroo::CollisionDetector;

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
}