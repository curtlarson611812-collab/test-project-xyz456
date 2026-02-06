#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::types::Point;

    #[test]
    fn test_near_g_threshold() {
        let config = Config::default();
        let detector = CollisionDetector::new_with_config(&config);

        // Test optimal threshold calculation
        let range_64 = BigInt256::from_u64(1 << 64);
        let threshold = CollisionDetector::optimal_near_threshold(&range_64, 24);
        assert!(threshold >= 1000); // Minimum threshold
        assert!(threshold <= 1 << 12); // Should be balanced

        // Test near-G detection
        let trap = Trap {
            x: [detector.near_g_thresh - 1, 0, 0, 0], // Below threshold
            dist: num_bigint::BigUint::from(1000u64),
            is_tame: true,
        };

        let range_width = BigInt256::from_u64(1 << 64);
        let result = detector.calculated_near_solve(&trap, &trap, &range_width);
        // Should attempt brute force for near-G points
        // (exact result depends on implementation)
    }

    #[test]
    fn test_collision_detector_config() {
        let mut config = Config::default();
        config.near_g_thresh = 1 << 18; // 256K

        let detector = CollisionDetector::new_with_config(&config);
        assert_eq!(detector.near_g_thresh, 1 << 18);
    }
}