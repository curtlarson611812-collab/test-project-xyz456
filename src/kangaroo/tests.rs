#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::types::Point;
    use crate::kangaroo::collision::Trap;
    use crate::math::BigInt256;
    use crate::kangaroo::CollisionDetector;

    #[test]
    fn test_near_g_threshold() {
        let config = Config::default();
        let detector = CollisionDetector::new_with_config(&config);

        // Test optimal Near-G threshold calculation
        let range_small = BigInt256::from_u64(1_000_000); // Small range
        let threshold_small = CollisionDetector::optimal_near_g_threshold(&range_small, 24);
        assert_eq!(threshold_small, 1000); // range/1000 = 1000, prob_based = 4096, min=1000

        let range_large = BigInt256::from_u64(10_000_000); // Larger range
        let threshold_large = CollisionDetector::optimal_near_g_threshold(&range_large, 24);
        assert_eq!(threshold_large, 4096); // prob_based = 4096, cost_based = 10000, min=4096

        // Test Near-G detection with low x[0]
        let trap_near_g = Trap {
            x: [100000, 0, 0, 0], // Below default 2^20 threshold
            dist: num_bigint::BigUint::from(1000u64),
            is_tame: true,
        };

        let range_width = BigInt256::from_u64(1 << 20); // Small range for testing
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
}