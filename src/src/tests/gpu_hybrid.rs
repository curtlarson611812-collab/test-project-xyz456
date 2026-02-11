use anyhow::Result;
use crate::gpu::backends::{Backend, HybridBackend};
use crate::math::{BigInt256, Scalar, BigInt512};
use crate::types::KangarooState;
use k256::ProjectivePoint;
use std::time::Duration;

// Suite entry
#[test]
fn gpu_hybrid_suite() -> Result<()> {
    let backend = HybridBackend::new()?;
    // Call sub-tests
    test_parity_barrett(&backend)?;
    test_parity_bigint_mul(&backend)?;
    test_gpu_mod_bias(&backend)?;
    test_gpu_collision_solve(&backend)?;
    #[cfg(feature = "near_collisions")]
    test_gpu_rho_walk(&backend)?;
    test_gpu_multi_target()?;
    test_hybrid_fallback()?;
    test_gpu_hybrid_puzzle66()?;
    test_10m_step_parity_hybrid()?;
    Ok(())
}

// Helper for mock large x (2^512 + 42)
fn mock_large_x() -> BigInt512 {
    let mut limbs = [0u32; 16];
    limbs[0] = 42; // Low limb
    limbs[8] = 1; // 2^256 bit
    BigInt512 { limbs }
}

// Helper for mock modulus 81
fn mock_mod81() -> BigInt256 {
    BigInt256::from_u64(81)
}

// Helper for mu precomp (floor(2^512 / 81))
fn compute_mu(modulus: &BigInt256) -> BigInt512 {
    // Simplified: in real impl use full Barrett precomp
    BigInt512::from_u64(1) // Placeholder
}

// CPU fallback for parity tests
fn cpu_barrett_reduce(x: &BigInt512, modulus: &BigInt256, _mu: &BigInt512) -> BigInt256 {
    // Use num_bigint for exact mod
    use num_bigint::BigUint;
    let x_big = BigUint::from_bytes_le(&x.to_bytes_le());
    let mod_big = BigUint::from_bytes_le(&modulus.to_bytes_le());
    let r_big = &x_big % &mod_big;
    BigInt256::from_bytes_le(&r_big.to_bytes_le())
}

// CPU bigint mul for parity
fn cpu_bigint_mul(a: &BigInt256, b: &BigInt256) -> BigInt512 {
    let a_big = num_bigint::BigUint::from_bytes_le(&a.to_bytes_le());
    let b_big = num_bigint::BigUint::from_bytes_le(&b.to_bytes_le());
    let res = &a_big * &b_big;
    BigInt512::from_bytes_le(&res.to_bytes_le())
}

fn test_parity_barrett(backend: &dyn Backend) -> Result<()> {
    let x_wide = mock_large_x(); // 2^512 + 42
    let modulus = mock_mod81(); // mod81 bias
    let mu = compute_mu(&modulus); // Precomp floor(2^512/mod)
    let gpu_res = backend.barrett_reduce(&x_wide, &modulus, &mu)?;
    let cpu_res = cpu_barrett_reduce(&x_wide, &modulus, &mu); // From bigint.rs
    assert_eq!(gpu_res, cpu_res, "Barrett parity fail");
    // Edge: Overflow loop
    let large_x = BigInt512::max() + BigInt512::one(); // Sim overflow
    let gpu_large = backend.barrett_reduce(&large_x, &modulus, &mu)?;
    assert!(gpu_large < modulus, "Overflow not handled");
    Ok(())
}

fn test_parity_bigint_mul(backend: &dyn Backend) -> Result<()> {
    let a = BigInt256::from_u64(42);
    let b = BigInt256::from_u64(179);
    let gpu_res = backend.bigint_mul(&a, &b)?;
    let cpu_res = cpu_bigint_mul(&a, &b);
    assert_eq!(gpu_res, cpu_res, "Bigint mul parity fail");
    // Then reduce mod81 for bias test
    let mod81 = mock_mod81();
    let mu = compute_mu(&mod81);
    let reduced = backend.barrett_reduce(&gpu_res, &mod81, &mu)?;
    assert_eq!(reduced, BigInt256::from_u64(42), "Mul + reduce fail"); // 42*179 %81 = 7518 %81 = 42
    Ok(())
}

fn test_gpu_collision_solve(backend: &dyn Backend) -> Result<()> {
    use crate::math::constants::CURVE_ORDER;

    let tame_dist = BigInt256::from_u64(100).to_limbs();
    let wild_dist = BigInt256::from_u64(58).to_limbs();
    let n = CURVE_ORDER.to_limbs();
    let diff = backend.safe_diff_mod_n(&tame_dist, &wild_dist, &n)?;
    assert_eq!(diff, [42,0,0,0,0,0,0,0], "Diff fail"); // 100-58=42
    // Phase 7 inv
    let inv = backend.mod_inverse(&diff, &n)?;
    let mul_back = backend.bigint_mul(&inv, &diff)?;
    let one_mod = backend.modulo(&mul_back, &n)?;
    assert_eq!(one_mod, BigInt256::one().to_limbs(), "Inv mul !=1 mod N");
    Ok(())
}

#[cfg(feature = "near_collisions")]
fn test_gpu_rho_walk(backend: &dyn Backend) -> Result<()> {
    // Mock cycle: tortoise=hare at step 42
    let mock_tort = mock_point(42 * BigInt256::one()); // Simplified
    let mock_hare = mock_tort.clone();
    let walk_res = backend.rho_walk(&mock_tort, &mock_hare, 100000)?;
    assert_eq!(walk_res.cycle_len, 42, "Brent cycle fail");
    // Phase 4/8: Multi-target solve post-walk
    let targets = load_mock_targets(5);
    let k = backend.solve_post_walk(&walk_res, &targets)?;
    assert!(k.is_some(), "Near solve fail");
    Ok(())
}

// Helpers for rho test
fn mock_point(_scalar: BigInt256) -> crate::types::Point {
    // Simplified: return generator
    crate::types::Point::generator()
}

fn load_mock_targets(_count: usize) -> Vec<crate::types::Point> {
    vec![crate::types::Point::generator(); _count]
}

fn test_gpu_multi_target() -> Result<()> {
    use crate::kangaroo::manager::KangarooManager;
    use crate::config::SearchConfig;

    let config = SearchConfig::default().with_puzzle_mode(true);
    let manager = KangarooManager::new(&config)?;
    let backend = HybridBackend::new()?;
    manager.step_batch_multi(10000, &backend)?; // GPU steps
    // Assert full load
    assert_eq!(manager.herds.len(), 34353 + puzzles_count, "Incomplete multi");
    // Phase 8 eviction
    config.enable_target_eviction = true;
    manager.check_eviction(&config);
    assert!(manager.herds.len() < 34353, "Stagnant not evicted");
    Ok(())
}

fn test_hybrid_fallback() -> Result<()> {
    let mut backend = HybridBackend::new()?;
    backend.simulate_cuda_fail(true); // Mock OOM
    let x = mock_large_x();
    let mod81 = mock_mod81();
    let mu = compute_mu(&mod81);
    let res = backend.barrett_reduce(&x, &mod81, &mu)?;
    let cpu_res = cpu_barrett_reduce(&x, &mod81, &mu);
    assert_eq!(res, cpu_res, "Fallback fail");
    // Mixed: CUDA inv + Vulkan step
    let mock_a = BigInt256::from_u64(42).to_limbs();
    let inv = backend.mod_inverse(&mock_a, &mock_mod81().to_limbs())?;
    let point = backend.scalar_mul_glv(&mock_point(BigInt256::one()), &inv)?;
    assert_eq!(point, mock_point(BigInt256::one()), "Mixed fail"); // Placeholder assert
    Ok(())
}

fn test_gpu_hybrid_puzzle66() -> Result<()> {
    use crate::kangaroo::manager::KangarooManager;
    use crate::config::SearchConfig;

    let config = SearchConfig::for_puzzle(66);
    let manager = KangarooManager::new(&config)?;
    let backend = HybridBackend::new()?;
    let priv_k = manager.run_until_solve(&backend, Duration::from_secs(60))?;
    assert_eq!(priv_k, crate::math::BigInt256::from_hex("123456789abcdef"), "Puzzle66 solve fail"); // Placeholder
    Ok(())
}

fn test_10m_step_parity_hybrid() -> Result<()> {
    use crate::parity::checker::ParityChecker;

    let backend = HybridBackend::new()?;
    let checker = ParityChecker::new();
    let (cpu_points, _cpu_dists) = checker.run_cpu_steps(10000000, mock_start_state());
    let (gpu_points, _gpu_dists) = backend.run_gpu_steps(10000000, mock_start_state())?;
    for i in 0..10000000 {
        assert_eq!(cpu_points[i], gpu_points[i], "Point parity fail at {}", i);
    }
    Ok(())
}

// Helpers
fn mock_start_state() -> crate::types::KangarooState {
    crate::types::KangarooState::new(
        crate::types::Point::generator(),
        crate::math::BigInt256::zero(),
        crate::types::AlphaBeta::default(),
        0,
        true,
    )
}

// Helper: puzzles count (assume from loader)
const puzzles_count: usize = 100; // Placeholder