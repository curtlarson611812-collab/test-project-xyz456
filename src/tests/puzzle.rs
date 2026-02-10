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
}