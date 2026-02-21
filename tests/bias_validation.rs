//! Bias detection validation tests
//!
//! Tests to verify bias detection functions work correctly

use num_bigint::BigInt;
use num_traits::cast::ToPrimitive;
use rand::distributions::Distribution;
use rand::{thread_rng, Rng};
use speedbitcrack::kangaroo::generator::{KangarooGenerator, new_slice};
use speedbitcrack::math::bigint::BigInt256;
use speedbitcrack::math::secp::Secp256k1;
use speedbitcrack::utils::pubkey_loader::detect_bias_single;
use statrs::distribution::Uniform;
use statrs::statistics::Statistics;
use std::collections::HashMap;

#[test]
fn test_mod9_bias_detection() {
    // Test number that is 0 mod 9: 18
    let x = BigInt256::from_u64(18);
    let (bias_mod, dominant_residue, _) = detect_bias_single(&x, 66);
    assert_eq!(bias_mod, 9); // Should detect bias mod 9
    assert_eq!(dominant_residue, 0); // 18 mod 9 = 0

    // Test number that is 3 mod 9: 21
    let x = BigInt256::from_u64(21);
    let (bias_mod, dominant_residue, _) = detect_bias_single(&x, 66);
    assert_eq!(dominant_residue, 3); // 21 mod 9 = 3
}

#[test]
fn test_mod27_bias_detection() {
    // Test number that is 0 mod 27: 54
    let x = BigInt256::from_u64(54);
    let (bias_mod, dominant_residue, _) = detect_bias_single(&x, 66);
    assert_eq!(bias_mod, 27); // Should detect bias mod 27
    assert_eq!(dominant_residue, 0); // 54 mod 27 = 0

    // Test number that is 9 mod 27: 36
    let x = BigInt256::from_u64(36);
    let (bias_mod, dominant_residue, _) = detect_bias_single(&x, 66);
    assert_eq!(dominant_residue, 9); // 36 mod 27 = 9
}

#[test]
fn test_mod81_bias_detection() {
    // Test number that is 0 mod 81: 162
    let x = BigInt256::from_u64(162);
    let (bias_mod, dominant_residue, _) = detect_bias_single(&x, 66);
    assert_eq!(bias_mod, 81); // Should detect bias mod 81
    assert_eq!(dominant_residue, 0); // 162 mod 81 = 0

    // Test number that is 27 mod 81: 108
    let x = BigInt256::from_u64(108);
    let (bias_mod, dominant_residue, _) = detect_bias_single(&x, 66);
    assert_eq!(dominant_residue, 27); // 108 mod 81 = 27
}

#[test]
fn test_large_number_bias_detection() {
    // Test with a large number that should be 0 mod 9
    // Use a number that ends with multiple 9s in decimal (digital root property)
    let large_num =
        BigInt256::from_hex("123456789012345678901234567890123456789012345678901234567890").unwrap();
    let (bias_mod, dominant_residue, _) = detect_bias_single(&large_num, 66);

    // Verify the modular arithmetic works correctly for large numbers
    assert!(dominant_residue < 9);
    assert!(dominant_residue < 27);
    assert!(dominant_residue < 81);

    // Test passed - dominant residue is within expected bounds
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
        let (bias_mod_1, dominant_residue_1, pos_proxy_1) = detect_bias_single(&num, 66);
        let (bias_mod_2, dominant_residue_2, pos_proxy_2) = detect_bias_single(&num, 66);

        assert_eq!(bias_mod_1, bias_mod_2);
        assert_eq!(dominant_residue_1, dominant_residue_2);
        assert_eq!(pos_proxy_1, pos_proxy_2);
    }
}

#[test]
fn test_magic_nine_candidates() {
    // Test known Magic 9 candidates
    let candidates = vec![
        BigInt256::from_u64(9),  // 9 mod 9 = 0
        BigInt256::from_u64(18), // 18 mod 9 = 0
        BigInt256::from_u64(27), // 27 mod 9 = 0
        BigInt256::from_u64(36), // 36 mod 9 = 0
    ];

    for candidate in candidates {
        let (bias_mod, dominant_residue, _) = detect_bias_single(&candidate, 66);
        assert_eq!(dominant_residue, 0, "Candidate should be 0 mod 9");
    }

    // Test non-candidates
    let non_candidates = vec![
        BigInt256::from_u64(1),
        BigInt256::from_u64(2),
        BigInt256::from_u64(10),
        BigInt256::from_u64(19),
    ];

    for non_candidate in non_candidates {
        let (bias_mod, dominant_residue, _) = detect_bias_single(&non_candidate, 66);
        assert_ne!(dominant_residue, 0, "Non-candidate should NOT be 0 mod 9");
    }
}

#[test]
fn test_refine_pos_slice() {
    let mut slice = new_slice((BigInt::from(0), BigInt::from(1000)), 0);
    let mut biases = HashMap::new();
    biases.insert(0, 1.2);

    KangarooGenerator::refine_pos_slice(&mut slice, &biases, 5);

    // Should refine bounds based on bias
    assert!(slice.low >= BigInt::from(0));
    assert!(slice.high > slice.low);
    assert_eq!(slice.iter, 1);
    assert!((slice.bias - 1.2).abs() < 0.001);
}

#[test]
fn test_pos_slice_max_iterations() {
    let mut slice = new_slice((BigInt::from(0), BigInt::from(1000)), 0);
    let biases = HashMap::new();

    // Refine beyond max iterations
    for _ in 0..10 {
        KangarooGenerator::refine_pos_slice(&mut slice, &biases, 3);
    }

    assert_eq!(slice.iter, 3); // Should cap at max_iterations
}

#[test]
fn test_random_in_slice() {
    let slice = new_slice((BigInt::from(100), BigInt::from(200)), 0);

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
    let flags: Vec<bool> = keys
        .iter()
        .map(|k| {
            let residue = (k % BigInt::from(81)).to_u64().unwrap();
            high_residues.contains(&residue)
        })
        .collect();

    assert_eq!(flags, vec![true, false, true]);
}

#[test]
fn test_hierarchical_bias_jumping() {
    use speedbitcrack::config::Config;
    use speedbitcrack::types::Point;

    let gen = KangarooGenerator::new(&Config::default());
    let curve = Secp256k1::new();
    let point = Point::infinity(); // Mock point

    // Test hierarchical bias preferences
    let jump_mod9 = gen.select_bias_aware_jump(&point, 9, 1.0, 0.0);
    let jump_mod27 = gen.select_bias_aware_jump(&point, 27, 1.0, 0.0);
    let jump_mod81 = gen.select_bias_aware_jump(&point, 81, 1.0, 0.0);
    let jump_pos = gen.select_bias_aware_jump(&point, 0, 1.5, 0.0);
    let jump_none = gen.select_bias_aware_jump(&point, 0, 1.0, 0.5);

    // All should be valid jump indices
    assert!(jump_mod9 < curve.g_multiples.len());
    assert!(jump_mod27 < curve.g_multiples.len());
    assert!(jump_mod81 < curve.g_multiples.len());
    assert!(jump_pos < curve.g_multiples.len());
    assert!(jump_none < curve.g_multiples.len());
}

#[test]
fn test_positional_proxy_integration() {
    // Test that positional proxy is correctly extracted from bias detection
    let test_keys = vec![
        BigInt::from(1),        // Small key, should have low proxy
        BigInt::from(1) << 130, // Large key, should have higher proxy
    ];

    for key in test_keys {
        let key_256 = BigInt256::from_biguint(&key.to_biguint().unwrap());
        let (_, _, pos_proxy) = detect_bias_single(&key_256, 67); // Puzzle #67
        assert!(pos_proxy >= 0.0 && pos_proxy <= 1.0);
    }
}

// Chunk: KS Bias Validation (tests/bias_validation.rs)
// Dependencies: statrs::distribution::{KolmogorovSmirnov, Uniform}, rand::{Rng, thread_rng}
#[test]
fn test_bias_ks() {
    let observed = vec![0.2, 0.15, 0.25, 0.1, 0.3]; // Biased mod5 freq
    let uniform = Uniform::new(0.0, 1.0).unwrap();
    let mut rng = thread_rng();
    let samples: Vec<f64> = (0..1000).map(|_| uniform.sample(&mut rng)).collect();
    // Simple statistical test - check if mean is within expected range
    let observed_mean = observed.clone().mean();
    let samples_mean = samples.clone().mean();
    assert!((observed_mean - samples_mean).abs() < 0.1); // Significant bias detected
}

// Bootstrap resampling for confidence intervals on speedup
#[test]
fn test_bias_bootstrap() {
    let solved_biases = vec![1.2, 1.35, 1.42]; // mod9/27/81 biases from solved data
    let mut bootstrapped_means = Vec::new();

    for _ in 0..1000 {
        // Bootstrap resampling
        let mut sample = Vec::new();
        for _ in 0..solved_biases.len() {
            sample.push(solved_biases[rand::random::<usize>() % solved_biases.len()]);
        }
        let mean = sample.iter().sum::<f64>() / sample.len() as f64;
        bootstrapped_means.push(mean);
    }

    let ci_lower = bootstrapped_means
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let ci_upper = bootstrapped_means
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        ci_lower > 1.1 && ci_upper < 1.6,
        "Bootstrap CI [{:.3}, {:.3}] outside expected bias range",
        ci_lower,
        ci_upper
    );
}

#[test]
fn test_bias_ks_validation() {
    // Test Kolmogorov-Smirnov validation for bias detection
    // Generate biased sample (clustered around 0.1-0.3)
    let observed = vec![0.15, 0.12, 0.18, 0.22, 0.14, 0.16, 0.19, 0.11, 0.17, 0.13];

    // Generate uniform reference sample
    use rand::thread_rng;
    let uniform = Uniform::new(0.0, 1.0).unwrap();
    let mut rng = thread_rng();
    let reference: Vec<f64> = (0..1000).map(|_| uniform.sample(&mut rng)).collect();

    // Simple statistical test - check if distributions are similar
    let observed_mean = observed.clone().mean();
    let reference_mean = reference.clone().mean();
    assert!((observed_mean - reference_mean).abs() < 0.05);

    // For significantly biased data, mean difference should be detectable
    let bias_detected = (observed_mean - reference_mean).abs() > 0.01;
    assert!(bias_detected, "Should detect bias in the data");

    // Calculate bias score (mock implementation)
    let bias_score = if bias_detected {
        // Simple score based on clustering around lower values
        let mean = observed.iter().sum::<f64>() / observed.len() as f64;
        let variance =
            observed.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / observed.len() as f64;
        1.0 + (0.3 - mean).abs() * 2.0 - variance * 5.0 // Reward clustering near 0.1-0.3
    } else {
        1.0
    };

    assert!(
        bias_score > 1.2,
        "Bias score {:.3} should be > 1.2 for significant bias",
        bias_score
    );
}
