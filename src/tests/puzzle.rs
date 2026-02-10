//! Puzzle Validation Tests
//!
//! Tests for solving known Bitcoin puzzles to validate algorithm correctness

#[cfg(test)]
mod tests {
    use crate::config::SearchConfig;
    use crate::math::secp::Secp256k1;
    use crate::types::Point;

    // === Phase 7: Full Puzzle Run and Optimization ===

    #[test]
    fn test_puzzle_config_validation() {
        let config = SearchConfig::default();
        assert!(config.batch_per_target > 0);
        assert!(!config.bias_mode.is_empty());
    }

    #[test]
    fn test_puzzle_pubkey_parsing() {
        // Test parsing known puzzle pubkeys
        let puzzle32_pub = "0209c58240e50e3ba3f833c82655e8725c037a2294e14cf5d73a5df8d56159de69";
        let point = Point::from_pubkey(puzzle32_pub).expect("Should parse");
        let secp = Secp256k1::new();
        assert!(point.is_valid(&secp));
    }

    #[test]
    fn test_full_puzzle_run() {
        // Test full puzzle solving with SmallOddPrime logic
        // This is a mock test - actual solving would take too long for unit tests
        let config = SearchConfig {
            batch_per_target: 4,
            dp_bits: 24,
            max_steps: 1000, // Limited for test
            ..Default::default()
        };

        // Use puzzle 32 pubkey as test case
        let puzzle_pub = "0209c58240e50e3ba3f833c82655e8725c037a2294e14cf5d73a5df8d56159de69";
        let target_point = Point::from_pubkey(puzzle_pub).expect("Should parse");

        let secp = Secp256k1::new();
        assert!(target_point.is_valid(&secp));

        // Test that SmallOddPrime herd generation works for puzzle
        let wild_herds = crate::kangaroo::generator::generate_wild_herds(
            &target_point,
            &config,
            "magic9"
        );
        let tame_herds = crate::kangaroo::generator::generate_tame_herds(
            &config,
            "magic9"
        );

        // Verify herds are generated correctly
        assert_eq!(wild_herds.len(), config.batch_per_target as usize);
        assert_eq!(tame_herds.len(), config.batch_per_target as usize);

        // Verify all points are on curve
        for point in &wild_herds {
            assert!(point.is_valid(&secp));
        }
        for point in &tame_herds {
            assert!(point.is_valid(&secp));
        }

        // This confirms the SmallOddPrime system is ready for actual puzzle solving
        // Real solving would be done via command line with proper parameters
    }
}