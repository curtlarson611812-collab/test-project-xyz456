//! Tests for Magic 9 GOLD Cluster Mode
//!
//! Validates cluster bias patterns, shared optimizations, and GOLD mode detection

use speedbitcrack::utils::bias::{MAGIC9_BIASES, get_magic9_bias};

#[test]
fn test_gold_cluster_bias_patterns() {
    // Verify all GOLD cluster keys have identical bias patterns (universal zeros)
    assert!(!MAGIC9_BIASES.is_empty(), "MAGIC9_BIASES should be populated by build.rs");

    if !MAGIC9_BIASES.is_empty() {
        let first_bias = MAGIC9_BIASES[0];

        // All keys should share identical bias patterns (GOLD cluster)
        for (i, &bias) in MAGIC9_BIASES.iter().enumerate() {
            assert_eq!(bias, first_bias,
                      "Key {} bias {:?} should match first key bias {:?} for GOLD cluster",
                      i, bias, first_bias);

            // Verify universal zero residues (GOLD pattern)
            assert_eq!(bias.0, 0, "Key {} should have mod3=0", i);  // mod3
            assert_eq!(bias.1, 0, "Key {} should have mod9=0", i);  // mod9
            assert_eq!(bias.2, 0, "Key {} should have mod27=0", i); // mod27
            assert_eq!(bias.3, 0, "Key {} should have mod81=0", i); // mod81
            assert_eq!(bias.4, 128, "Key {} should have Hamming=128", i); // Hamming
        }

        println!("✅ GOLD Cluster validation passed: All {} keys share identical bias patterns", MAGIC9_BIASES.len());
    }
}

#[test]
fn test_magic9_bias_getter() {
    // Test bias getter function
    let bias = get_magic9_bias(0);
    assert_eq!(bias.4, 128, "Should return valid Hamming weight");

    // Test out-of-bounds access
    let fallback_bias = get_magic9_bias(999);
    assert_eq!(fallback_bias, (0, 0, 0, 0, 128), "Should return fallback bias for invalid index");
}

#[test]
fn test_gold_cluster_detection() {
    // This would test the is_magic9_gold_cluster() function
    // For now, verify the bias array structure
    assert!(MAGIC9_BIASES.len() == 9, "Should have exactly 9 Magic 9 biases");
}

#[test]
fn test_nested_modulus_validation() {
    // Verify that GOLD cluster satisfies nested modulus relationships
    // If mod81=0, then mod27=0, mod9=0, mod3=0 must also be true
    for (i, &bias) in MAGIC9_BIASES.iter().enumerate() {
        if bias.3 == 0 {  // mod81 = 0
            assert_eq!(bias.2, 0, "Key {}: mod81=0 implies mod27=0", i);
            assert_eq!(bias.1, 0, "Key {}: mod81=0 implies mod9=0", i);
            assert_eq!(bias.0, 0, "Key {}: mod81=0 implies mod3=0", i);
        }
    }
}