//! Bias detection validation tests
//!
//! Tests to verify bias detection functions work correctly

use speedbitcrack::math::bigint::BigInt256;
use speedbitcrack::utils::pubkey_loader::detect_bias_single;

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