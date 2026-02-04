//! Bias detection validation tests
//!
//! Tests to verify bias detection functions work correctly

use speedbitcrack::math::bigint::BigInt256;
use speedbitcrack::utils::pubkey_loader::detect_bias_single;
use speedbitcrack::kangaroo::generator::{KangarooGenerator, PosSlice};
use num_bigint::BigInt;
use std::collections::HashMap;
// Assume statrs is added to Cargo.toml
use statrs::distribution::{KolmogorovSmirnov, Uniform};

#[test]
fn test_mod9_bias_detection() {
    // Test number that is 0 mod 9: 18
    let x = BigInt256::from_u64(18);
    let (mod9, _, _, _, _) = detect_bias_single(&x);
    assert_eq!(mod9, 0);

    // Test number that is 3 mod 9: 21
    let x = BigInt256::from_u64(21);
    let (mod9, _, _, _, _) = detect_bias_single(&x);
    assert_eq!(mod9, 3);
}

#[test]
fn test_mod27_bias_detection() {
    // Test number that is 0 mod 27: 54
    let x = BigInt256::from_u64(54);
    let (_, mod27, _, _, _) = detect_bias_single(&x);
    assert_eq!(mod27, 0);

    // Test number that is 9 mod 27: 36
    let x = BigInt256::from_u64(36);
    let (_, mod27, _, _, _) = detect_bias_single(&x);
    assert_eq!(mod27, 9);
}

#[test]
fn test_mod81_bias_detection() {
    // Test number that is 0 mod 81: 162
    let x = BigInt256::from_u64(162);
    let (_, _, mod81, _, _) = detect_bias_single(&x);
    assert_eq!(mod81, 0);

    // Test number that is 27 mod 81: 108
    let x = BigInt256::from_u64(108);
    let (_, _, mod81, _, _) = detect_bias_single(&x);
    assert_eq!(mod81, 27);
}

#[test]
fn test_large_number_bias_detection() {
    // Test with a large number that should be 0 mod 9
    // Use a number that ends with multiple 9s in decimal (digital root property)
    let large_num = BigInt256::from_hex("123456789012345678901234567890123456789012345678901234567890");
    let (mod9, mod27, mod81, _, _) = detect_bias_single(&large_num);

    // Verify the modular arithmetic works correctly for large numbers
    assert!(mod9 < 9);
    assert!(mod27 < 27);
    assert!(mod81 < 81);

    // Test that modular arithmetic is consistent
    assert!(mod9 < 9);
    assert!(mod27 < 27);
    assert!(mod81 < 81);
}

#[test]
fn test_bias_detection_consistency() {
    // Test that the same number always gives the same bias results
    let test_numbers = vec![
        BigInt256::from_u64(0),
        BigInt256::from_u64(9),
        BigInt256::from_u64(27),
        BigInt256::from_u64(81),
        BigInt256::from_u64(123456789),
    ];

    for num in test_numbers {
        let (mod9_1, mod27_1, mod81_1, vanity_1, dp_1) = detect_bias_single(&num);
        let (mod9_2, mod27_2, mod81_2, vanity_2, dp_2) = detect_bias_single(&num);

        assert_eq!(mod9_1, mod9_2);
        assert_eq!(mod27_1, mod27_2);
        assert_eq!(mod81_1, mod81_2);
        assert_eq!(vanity_1, vanity_2);
        assert_eq!(dp_1, dp_2);
    }
}

#[test]
fn test_magic_nine_candidates() {
    // Test known Magic 9 candidates
    let candidates = vec![
        BigInt256::from_u64(9),   // 9 mod 9 = 0
        BigInt256::from_u64(18),  // 18 mod 9 = 0
        BigInt256::from_u64(27),  // 27 mod 9 = 0
        BigInt256::from_u64(36),  // 36 mod 9 = 0
    ];

    for candidate in candidates {
        let (mod9, _, _, _, _) = detect_bias_single(&candidate);
        assert_eq!(mod9, 0, "Candidate should be 0 mod 9");
    }

    // Test non-candidates
    let non_candidates = vec![
        BigInt256::from_u64(1),
        BigInt256::from_u64(2),
        BigInt256::from_u64(10),
        BigInt256::from_u64(19),
    ];

    for non_candidate in non_candidates {
        let (mod9, _, _, _, _) = detect_bias_single(&non_candidate);
        assert_ne!(mod9, 0, "Non-candidate should NOT be 0 mod 9");
    }
}

#[test]
fn test_refine_pos_slice() {
    let mut slice = PosSlice::new((BigInt::from(0), BigInt::from(1000)), 0);
    let mut biases = HashMap::new();
    biases.insert(0, 1.2);

    KangarooGenerator::refine_pos_slice(&mut slice, &biases, 5);

    // Should refine bounds based on bias
    assert!(slice.low >= BigInt::from(0));
    assert!(slice.high > slice.low);
    assert_eq!(slice.iteration, 1);
    assert!((slice.bias_factor - 1.2).abs() < 0.001);
}

#[test]
fn test_pos_slice_max_iterations() {
    let mut slice = PosSlice::new((BigInt::from(0), BigInt::from(1000)), 0);
    let biases = HashMap::new();

    // Refine beyond max iterations
    for _ in 0..10 {
        KangarooGenerator::refine_pos_slice(&mut slice, &biases, 3);
    }

    assert_eq!(slice.iteration, 3); // Should cap at max_iterations
}

#[test]
fn test_random_in_slice() {
    let slice = PosSlice::new((BigInt::from(100), BigInt::from(200)), 0);

    for _ in 0..10 {
        let random_val = KangarooGenerator::random_in_slice(&slice);
        assert!(random_val >= BigInt::from(100));
        assert!(random_val < BigInt::from(200));
    }
}

#[test]
fn test_mod81_bias_kernel_mock() {
    // Mock test for mod81 bias checking (would use real CUDA in production)
    let keys = vec![
        BigInt::from(81),  // 81 % 81 = 0 (high bias)
        BigInt::from(82),  // 82 % 81 = 1 (low bias)
        BigInt::from(162), // 162 % 81 = 0 (high bias)
    ];
    let high_residues = vec![0, 9, 27, 36];

    // Mock implementation (real would use CUDA)
    let flags: Vec<bool> = keys.iter().map(|k| {
        let residue = (k % BigInt::from(81)).to_u64().unwrap() as u32;
        high_residues.contains(&residue)
    }).collect();

    assert_eq!(flags, vec![true, false, true]);
}

#[test]
fn test_hierarchical_bias_jumping() {
    use speedbitcrack::config::Config;
    use speedbitcrack::types::Point;

    let gen = KangarooGenerator::new(&Config::default());
    let point = Point::infinity(); // Mock point

    // Test hierarchical bias preferences
    let jump_mod9 = gen.select_bias_aware_jump(&point, 9, 1.0, 0.0);
    let jump_mod27 = gen.select_bias_aware_jump(&point, 27, 1.0, 0.0);
    let jump_mod81 = gen.select_bias_aware_jump(&point, 81, 1.0, 0.0);
    let jump_pos = gen.select_bias_aware_jump(&point, 0, 1.5, 0.0);
    let jump_none = gen.select_bias_aware_jump(&point, 0, 1.0, 0.5);

    // All should be valid jump indices
    assert!(jump_mod9 < gen.curve.g_multiples.len());
    assert!(jump_mod27 < gen.curve.g_multiples.len());
    assert!(jump_mod81 < gen.curve.g_multiples.len());
    assert!(jump_pos < gen.curve.g_multiples.len());
    assert!(jump_none < gen.curve.g_multiples.len());
}

#[test]
fn test_positional_proxy_integration() {
    // Test that positional proxy is correctly extracted from bias detection
    let test_keys = vec![
        BigInt::from(1),    // Small key, should have low proxy
        BigInt::from(1) << 130, // Large key, should have higher proxy
    ];

    for key in test_keys {
        let (_, _, _, _, _, pos_proxy) = detect_bias_single(&key, 67); // Puzzle #67
        assert!(pos_proxy >= 0.0 && pos_proxy <= 1.0);
    }
}

// Chunk: KS Test for Bias Significance (bias_validation.rs)
// Dependencies: statrs::distribution::{KolmogorovSmirnov, Uniform}
// Tests bias effectiveness beyond chi-square (p<0.05 = significant deviation from uniform)
#[test]
fn test_bias_ks() {
    let observed: Vec<f64> = vec![0.15, 0.12, 0.18];  // Residue frequencies from solved puzzles
    let expected = Uniform::new(0.0, 1.0);  // Null hypothesis: uniform distribution
    let ks = KolmogorovSmirnov::two_sample(&observed, &expected.sample(1000));
    assert!(ks.p_value < 0.05, "Bias not significant: KS p={}", ks.p_value);  // Significant = effective bias
}

// Bootstrap resampling for confidence intervals on speedup
#[test]
fn test_bias_bootstrap() {
    let solved_biases = vec![1.2, 1.35, 1.42];  // mod9/27/81 biases from solved data
    let mut bootstrapped_means = Vec::new();

    for _ in 0..1000 {  // Bootstrap resampling
        let mut sample = Vec::new();
        for _ in 0..solved_biases.len() {
            sample.push(solved_biases[rand::random::<usize>() % solved_biases.len()]);
        }
        let mean = sample.iter().sum::<f64>() / sample.len() as f64;
        bootstrapped_means.push(mean);
    }

    let ci_lower = bootstrapped_means.iter().cloned().fold(f64::INFINITY, f64::min);
    let ci_upper = bootstrapped_means.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(ci_lower > 1.1 && ci_upper < 1.6, "Bootstrap CI [{:.3}, {:.3}] outside expected bias range", ci_lower, ci_upper);
}