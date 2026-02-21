#[cfg(test)]
mod tests {
    use crate::config::{BiasMode, Config};
    use crate::kangaroo::collision::Trap;
    use crate::kangaroo::generator::select_bucket;
    use crate::kangaroo::generator::{generate_tame_herds, generate_wild_herds};
    use crate::kangaroo::stepper::KangarooStepper;
    use crate::kangaroo::CollisionDetector;
    use crate::kangaroo::SearchConfig;
    use crate::math::constants::CURVE_ORDER_BIGINT;
    use crate::math::BigInt256;
    use crate::math::Secp256k1;
    use crate::types::KangarooState;
    use crate::SmallOddPrime_Precise_code as sop;

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

    // === SmallOddPrime_Precise_code.rs Integration Tests ===

    #[test]
    fn test_prime_multipliers() {
        assert_eq!(sop::PRIME_MULTIPLIERS.len(), 32);
        assert_eq!(sop::PRIME_MULTIPLIERS[0], 131); // First sacred prime
        assert_eq!(sop::PRIME_MULTIPLIERS[31], 307); // Last sacred prime

        // Verify all are >128, odd, and low Hamming weight
        for &prime in &sop::PRIME_MULTIPLIERS {
            assert!(prime > 128, "Prime {} should be >128", prime);
            assert!(prime % 2 == 1, "Prime {} should be odd", prime);
        }
    }

    #[test]
    fn test_get_biased_prime() {
        // Test basic cycling - all should return PRIME_MULTIPLIERS[0] = 179 for index 0
        assert_eq!(sop::get_biased_prime(0, 81), 179); // (0 % 81) % 32 = 0 -> 179
        assert_eq!(sop::get_biased_prime(0, 9), 179); // (0 % 9) % 32 = 0 -> 179
        assert_eq!(sop::get_biased_prime(0, 27), 179); // (0 % 27) % 32 = 0 -> 179

        // Test cycling
        assert_eq!(sop::get_biased_prime(32, 81), 179); // (32 % 81) % 32 = 0 -> 179

        // Test different indices
        assert_eq!(sop::get_biased_prime(1, 81), 257); // (1 % 81) % 32 = 1 -> 257
        assert_eq!(sop::get_biased_prime(31, 81), 1583); // (31 % 81) % 32 = 31 -> 1583

        // Test edge case: bias_mod = 1
        assert_eq!(sop::get_biased_prime(0, 1), 179); // (0 % 1) % 32 = 0 -> 179
        assert_eq!(sop::get_biased_prime(1, 1), 179); // (1 % 1) % 32 = 0 -> 179
    }

    #[test]
    fn test_prime_properties() {
        // Verify low Hamming weight (fast GPU multiplication)
        for &prime in &sop::PRIME_MULTIPLIERS {
            let hamming = (prime as u64).count_ones();
            assert!(
                hamming <= 8,
                "Prime {} has high Hamming weight: {}",
                prime,
                hamming
            );
        }

        // Verify primes are distinct
        let mut seen = std::collections::HashSet::new();
        for &prime in &sop::PRIME_MULTIPLIERS {
            assert!(!seen.contains(&prime), "Duplicate prime: {}", prime);
            seen.insert(prime);
        }
    }

    #[test]
    fn test_multiplicative_wild_herds() {
        let config = SearchConfig {
            batch_per_target: 3,
            ..Default::default()
        };
        let curve = Secp256k1::new();
        let target = curve.g;
        let herds = generate_wild_herds(&target, &config, "magic9");
        assert_eq!(herds.len(), 3);
        let expected_first = curve
            .mul_constant_time(&BigInt256::from_u64(179), &target)
            .unwrap();
        assert_eq!(herds[0], expected_first);
        for herd in &herds {
            assert!(herd.is_valid(&curve)); // From secp.rs
        }
    }

    #[test]
    fn test_additive_tame_herds() {
        let config = SearchConfig {
            batch_per_target: 3,
            ..Default::default()
        };
        let curve = Secp256k1::new();
        let herds = generate_tame_herds(&config, "magic9");
        assert_eq!(herds.len(), 3);
        // Verify accumulation: 179 + 257 + 281 = 717 * G for last
        let expected_sum = 179 + 257 + 281;
        let expected_last = curve
            .mul_constant_time(&BigInt256::from_u64(expected_sum), &curve.g)
            .unwrap();
        let expected_affine = curve.to_affine(&expected_last);
        let actual_affine = curve.to_affine(&herds[2]);
        assert_eq!(expected_affine.x, actual_affine.x);
        assert_eq!(expected_affine.y, actual_affine.y);
        for herd in &herds {
            assert!(herd.is_valid(&curve));
        }
    }

    // === Phase 5: Integration Testing Stepping and Bucket ===

    #[test]
    fn test_select_sop_bucket() {
        let curve = Secp256k1::new();
        let point = curve.g; // Mock point
        let dist = BigInt256::from_u64(123);
        let seed = 456u32;
        let step = 789u32;

        // Test tame bucket (deterministic)
        let tame_bucket = select_bucket(&point, &dist, seed, step, true);
        assert_eq!(tame_bucket, (step % 32) as u32); // Tame deterministic per code

        // Test wild bucket (state-mixed)
        let wild_bucket = select_bucket(&point, &dist, seed, step, false);
        // Wild should be mixed, not necessarily different but state-dependent
        assert!(wild_bucket < 32); // Valid bucket range
    }

    #[test]
    fn test_step_kangaroo_with_bias() {
        let curve = Secp256k1::new();

        // Test tame kangaroo step
        let tame_state = KangarooState::new(
            curve.g.clone(),
            BigInt256::zero(), // distance as BigInt256
            [0u64; 4],         // alpha
            [0u64; 4],         // beta
            true,              // is_tame
            false,             // is_dp
            0,                 // id
            0,                 // step
            0,                 // kangaroo_type
        );
        let stepper = KangarooStepper::new(false);
        let new_tame_state = stepper.step_kangaroo_with_bias(&tame_state, None, 81);

        // Tame should have moved position and increased distance (additive)
        assert_ne!(new_tame_state.position, tame_state.position);
        // Distance comparison for BigInt256 - check if greater than zero
        assert!(!new_tame_state.distance.is_zero());

        let wild_state = KangarooState::new(
            curve.g.clone(),
            BigInt256::zero(), // distance as BigInt256
            [0u64; 4],         // alpha
            [0u64; 4],         // beta
            false,             // is_tame
            false,             // is_dp
            1,                 // id
            0,                 // step
            0,                 // kangaroo_type
        );

        let new_wild_state = stepper.step_kangaroo_with_bias(&wild_state, Some(&curve.g), 81);

        // Wild should have moved position (multiplicative with target)
        assert_ne!(new_wild_state.position, wild_state.position);
        // Distance may or may not change depending on implementation
    }

    #[test]
    fn test_collision_inversion_large() {
        use crate::math::constants::CURVE_ORDER_BIGINT;
        use num_bigint::BigUint;

        let detector = CollisionDetector::new();

        let prime = 179u64;
        let d_tame = BigInt256::from_hex("10000000000000000").expect("Invalid hex"); // Large >u64
        let d_wild = BigInt256::from_hex("5000000000000000").expect("Invalid hex");

        let k = detector
            .solve_collision_inversion(prime, d_tame.clone(), d_wild.clone(), &CURVE_ORDER_BIGINT)
            .expect("Inversion failed");

        // Verify: k * prime ≡ (d_tame - d_wild) mod n
        // Expected: k = inv(prime) * (d_tame - d_wild) mod n
        let prime_big = BigUint::from_bytes_be(&BigInt256::from_u64(prime).to_bytes_be());
        let n_big = BigUint::from_bytes_be(&CURVE_ORDER_BIGINT.to_bytes_be());
        let inv_prime = prime_big.modinv(&n_big).expect("Prime inverse failed");

        let diff = d_tame - d_wild;
        let diff_big = BigUint::from_bytes_be(&diff.to_bytes_be());
        let expected_big = (inv_prime * diff_big) % &n_big;

        let k_big = BigUint::from_bytes_be(&k.to_bytes_be());
        assert_eq!(k_big, expected_big);
    }

    #[test]
    fn test_collision_inversion_non_co_prime() {
        let prime = 2u64; // Even, non-co-prime to n (n odd)
        let d_tame = BigInt256::zero();
        let d_wild = BigInt256::zero();
        let _n = &CURVE_ORDER_BIGINT;
        let detector = CollisionDetector::new();
        let k = detector.solve_collision_inversion(prime, d_tame, d_wild, &CURVE_ORDER_BIGINT);
        assert!(k.is_none()); // Fails inv
    }
}
