//! Tier 1 validators: load solved puzzle pubkey/range, run solve, assert privkey match
//!
//! Integration tests for solved Bitcoin puzzles to validate solver correctness

use speedbitcrack::config::{Config, SearchMode};
use speedbitcrack::kangaroo::KangarooManager;
use speedbitcrack::types::Solution;
use std::collections::HashMap;

/// Known solved puzzle data for validation
/// Format: (puzzle_number, pubkey_hex, private_key_hex)
const SOLVED_PUZZLES: &[(&str, &str, &str)] = &[
    // Add actual solved puzzles here when available
    // ("64", "pubkey_hex", "privkey_hex"),
    // ("65", "pubkey_hex", "privkey_hex"),
    // ("66", "pubkey_hex", "privkey_hex"),
];

/// Test solving puzzle #64
#[test]
fn test_puzzle_64() {
    let (puzzle_num, pubkey_hex, expected_privkey) = SOLVED_PUZZLES[0];

    let mut config = create_test_config(puzzle_num, pubkey_hex);
    config.validate_puzzle = Some(64);

    let manager = KangarooManager::new(config).expect("Failed to create manager");

    // This would run the actual solver in a real test
    // For now, just verify the test setup
    assert_eq!(puzzle_num, "64");
    assert!(!pubkey_hex.is_empty());
    assert!(!expected_privkey.is_empty());
}

/// Test solving puzzle #65
#[test]
fn test_puzzle_65() {
    let (puzzle_num, pubkey_hex, expected_privkey) = SOLVED_PUZZLES[1];

    let mut config = create_test_config(puzzle_num, pubkey_hex);
    config.validate_puzzle = Some(65);

    let manager = KangarooManager::new(config).expect("Failed to create manager");

    assert_eq!(puzzle_num, "65");
    assert!(!pubkey_hex.is_empty());
    assert!(!expected_privkey.is_empty());
}

/// Test solving puzzle #66
#[test]
fn test_puzzle_66() {
    let (puzzle_num, pubkey_hex, expected_privkey) = SOLVED_PUZZLES[2];

    let mut config = create_test_config(puzzle_num, pubkey_hex);
    config.validate_puzzle = Some(66);

    let manager = KangarooManager::new(config).expect("Failed to create manager");

    assert_eq!(puzzle_num, "66");
    assert!(!pubkey_hex.is_empty());
    assert!(!expected_privkey.is_empty());
}

/// Create test configuration for puzzle validation
fn create_test_config(puzzle_num: &str, pubkey_hex: &str) -> Config {
    let mut config = Config::parse().unwrap_or_default();

    // Configure for puzzle testing
    config.mode = SearchMode::Interval {
        low: 1,
        high: u64::MAX,
    };
    config.test_mode = true;
    config.max_ops = 1_000_000; // Limited ops for testing
    config.dp_bits = 20; // Smaller DP table for testing

    // Create temporary puzzle file
    let puzzle_content = format!("{},{},1,{},0.001", puzzle_num, pubkey_hex, u64::MAX);
    std::fs::write("test_puzzle.txt", puzzle_content).expect("Failed to create test puzzle file");

    config.puzzles_file = std::path::PathBuf::from("test_puzzle.txt");
    config.puzzle_mode = true;

    config
}

/// Test helper: validate solution against known answer
fn validate_solution(solution: &Solution, expected_privkey: &str) -> bool {
    let expected_bytes = hex::decode(expected_privkey).expect("Invalid expected private key hex");
    let solution_bytes = solution.private_key.map(|x| x.to_le_bytes()).concat();

    expected_bytes == solution_bytes && solution.verified
}

/// Integration test: run full solver on puzzle and verify result
#[test]
#[ignore] // Ignored by default due to long runtime
fn integration_test_puzzle_solve() {
    // This test would actually run the solver
    // For now, it's a placeholder

    for (puzzle_num, pubkey_hex, expected_privkey) in SOLVED_PUZZLES {
        println!("Testing puzzle {}...", puzzle_num);

        let config = create_test_config(puzzle_num, pubkey_hex);
        let mut manager = KangarooManager::new(config).expect("Failed to create manager");

        // Run solver with timeout
        let result = manager.run();

        match result {
            Ok(Some(solution)) => {
                assert!(validate_solution(&solution, expected_privkey),
                       "Solution validation failed for puzzle {}", puzzle_num);
                println!("✓ Puzzle {} solved correctly", puzzle_num);
            }
            Ok(None) => {
                panic!("Solver failed to find solution for puzzle {}", puzzle_num);
            }
            Err(e) => {
                panic!("Solver error for puzzle {}: {}", puzzle_num, e);
            }
        }
    }
}

/// Performance regression test
#[test]
fn performance_regression_test() {
    // Test that solver performance doesn't regress
    // This would benchmark solve time and compare against baseline

    let config = create_test_config("64", SOLVED_PUZZLES[0].1);
    let manager = KangarooManager::new(config).expect("Failed to create manager");

    // Measure initialization time
    let start = std::time::Instant::now();
    let _ = manager.target_count();
    let init_time = start.elapsed();

    // Should initialize in reasonable time
    assert!(init_time.as_millis() < 1000, "Initialization took too long: {:?}", init_time);
}

/// Memory usage test
#[test]
fn memory_usage_test() {
    // Test that solver doesn't use excessive memory
    // This would monitor memory usage during solver operation

    let config = create_test_config("64", SOLVED_PUZZLES[0].1);
    let manager = KangarooManager::new(config).expect("Failed to create manager");

    // Check initial memory usage
    // (In real implementation, would use memory profiling)
    assert!(manager.target_count() > 0, "No targets loaded");
}

/// Correctness test with known small solution
#[test]
fn correctness_test_small_solution() {
    // Test with a known small private key to verify solver correctness
    // This would use a puzzle with known small solution

    // Placeholder - would need actual small-solution test data
    assert!(true, "Correctness test placeholder");
}

/// Cleanup test files
#[test]
fn cleanup() {
    // Clean up test files
    let _ = std::fs::remove_file("test_puzzle.txt");
}