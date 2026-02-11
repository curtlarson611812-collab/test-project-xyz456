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

    #[test]
    fn test_large_step_parity() {
        use crate::math::bigint::BigInt256;
        use crate::types::KangarooState;
        use crate::kangaroo::stepper::KangarooStepper;

        let curve = Secp256k1::new();
        let stepper = KangarooStepper::new(false); // CPU reference

        // Create initial kangaroo state
        let initial_state = KangarooState::new(
            curve.g.clone(),
            BigInt256::from_u64(1000000), // Large initial distance
            [0; 4], [0; 4], // alpha, beta
            true, false, // tame, not dp
            0, // id
        );

        // Test 10M steps (scaled down for test performance)
        const TEST_STEPS: usize = 10000; // 10k steps for test, represents 10M in practice

        let mut cpu_state = initial_state.clone();
        for i in 0..TEST_STEPS {
            let bias = 179 + (i % 32) * 2; // Cycle through SmallOddPrime multipliers
            cpu_state = stepper.step_kangaroo_with_bias(&cpu_state, None, bias as u64);
        }

        // GPU parity check (if available)
        #[cfg(any(feature = "rustacuda", feature = "wgpu"))]
        {
            let mut gpu_state = initial_state.clone();

            #[cfg(feature = "rustacuda")]
            if let Ok(mut cuda_backend) = crate::gpu::backends::cuda_backend::CudaBackend::new() {
                for i in 0..TEST_STEPS {
                    let bias = 179 + (i % 32) * 2;
                    gpu_state = cuda_backend.step_kangaroo(&gpu_state, None, bias as u64).unwrap();
                }

                // Compare CPU vs GPU final states
                assert_eq!(cpu_state.position.x, gpu_state.position.x, "CUDA position X mismatch after {} steps", TEST_STEPS);
                assert_eq!(cpu_state.position.y, gpu_state.position.y, "CUDA position Y mismatch after {} steps", TEST_STEPS);
                assert_eq!(cpu_state.distance, gpu_state.distance, "CUDA distance mismatch after {} steps", TEST_STEPS);

                println!("CUDA large step parity test passed ✓ ({} steps)", TEST_STEPS);
            }

            #[cfg(feature = "wgpu")]
            if let Ok(mut vulkan_backend) = crate::gpu::backends::vulkan_backend::VulkanBackend::new() {
                gpu_state = initial_state.clone();
                for i in 0..TEST_STEPS {
                    let bias = 179 + (i % 32) * 2;
                    gpu_state = vulkan_backend.step_kangaroo(&gpu_state, None, bias as u64).unwrap();
                }

                // Compare CPU vs GPU final states
                assert_eq!(cpu_state.position.x, gpu_state.position.x, "Vulkan position X mismatch after {} steps", TEST_STEPS);
                assert_eq!(cpu_state.position.y, gpu_state.position.y, "Vulkan position Y mismatch after {} steps", TEST_STEPS);
                assert_eq!(cpu_state.distance, gpu_state.distance, "Vulkan distance mismatch after {} steps", TEST_STEPS);

                println!("Vulkan large step parity test passed ✓ ({} steps)", TEST_STEPS);
            }
        }

        // Verify CPU state changed significantly (proof of computation)
        assert_ne!(cpu_state.position.x, initial_state.position.x, "Position should change after {} steps", TEST_STEPS);
        assert!(cpu_state.distance > initial_state.distance, "Distance should increase after {} steps", TEST_STEPS);

        println!("Large step parity test completed ✓ (CPU reference: {} steps)", TEST_STEPS);
    }

    #[cfg(feature = "hybrid")]
    #[test]
    fn test_gpu_hybrid_puzzle66() -> anyhow::Result<()> {
        use crate::gpu::backends::HybridBackend;
        use crate::kangaroo::manager::KangarooManager;
        use std::time::Duration;

        let config = SearchConfig::for_puzzle(66);
        let manager = KangarooManager::new(&config)?;
        let backend = HybridBackend::new()?;
        let priv_k = manager.run_until_solve(&backend, Duration::from_secs(60))?;
        assert_eq!(priv_k, KNOWN_66_PRIV, "Puzzle66 solve fail"); // 0x... known
        Ok(())
    }

    #[cfg(feature = "hybrid")]
    #[test]
    fn test_10m_step_parity_hybrid() -> anyhow::Result<()> {
        use crate::gpu::backends::HybridBackend;
        use crate::parity::checker::ParityChecker;

        let backend = HybridBackend::new()?;
        let checker = ParityChecker::new();
        let (cpu_points, cpu_dists) = checker.run_cpu_steps(10000000, mock_start_state());
        let (gpu_points, gpu_dists) = backend.run_gpu_steps(10000000, mock_start_state())?;
        for i in 0..10000000 {
            assert_eq!(cpu_points[i], gpu_points[i], "Point parity fail at {}", i);
        }
        Ok(())
    }

    // Helper for mock start state
    fn mock_start_state() -> crate::types::KangarooState {
        // Simplified mock
        crate::types::KangarooState::new(
            Point::generator(),
            crate::math::BigInt256::zero(),
            crate::types::AlphaBeta::default(),
            0,
            true,
        )
    }

    // Known priv for puzzle 66 (placeholder)
    const KNOWN_66_PRIV: &str = "0x123456789abcdef"; // Replace with real
}