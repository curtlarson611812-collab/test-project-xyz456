//! GPU Backend Trait Definition
//!
//! Unified interface for GPU acceleration backends (CUDA, Vulkan, CPU)

use crate::kangaroo::collision::Trap;
use anyhow::Result;

#[derive(Debug, Clone)]
pub struct RhoWalkResult {
    pub cycle_len: u32,
    pub cycle_point: [[u32;8];3],
    pub cycle_dist: [u32;8],
}

/// Unified GPU backend trait for hybrid Vulkan+CUDA acceleration
/// Enables dispatch of kangaroo operations to appropriate GPU backends
#[async_trait::async_trait]
pub trait GpuBackend {
    /// Create a new backend instance
    async fn new() -> Result<Self> where Self: Sized;

    /// Precompute jump table for kangaroo algorithm (G * 2^i for i in 0..32)
    /// Returns (positions, distances) for optimized jump operations
    fn precomp_table(&self, primes: Vec<[u32;8]>, base: [u32;8]) -> Result<(Vec<[[u32;8];3]>, Vec<[u32;8]>)>;

    /// Execute batch kangaroo stepping (parallel walk operations)
    /// Updates positions/distances, returns collision traps
    fn step_batch(&self, positions: &mut Vec<[[u32;8];3]>, distances: &mut Vec<[u32;8]>, types: &Vec<u32>) -> Result<Vec<Trap>>;

    /// Execute bias-enhanced batch kangaroo stepping with config parameters
    /// Supports Magic9 skewing, Primes factoring, and GOLD hierarchical nudging
    fn step_batch_bias(&self, positions: &mut Vec<[[u32;8];3]>, distances: &mut Vec<[u32;8]>, types: &Vec<u32>, config: &crate::config::Config) -> Result<Vec<Trap>>;

    /// Batch modular inverse for discrete logarithm denominator calculation
    fn batch_inverse(&self, inputs: Vec<[u32;8]>, modulus: [u32;8]) -> Result<Vec<[u32;8]>>;

    /// Batch collision solving for private key recovery
    fn batch_solve(&self, alphas: Vec<[u32;8]>, betas: Vec<[u32;8]>) -> Result<Vec<[u64;4]>>;

    /// Advanced batch collision solving with target points
    fn batch_solve_collision(&self, alpha_t: Vec<[u32;8]>, alpha_w: Vec<[u32;8]>, beta_t: Vec<[u32;8]>, beta_w: Vec<[u32;8]>, target: Vec<[u32;8]>, n: [u32;8]) -> Result<Vec<[u32;8]>>;

    /// Batch BSGS solving for near-collision resolution
    fn batch_bsgs_solve(&self, deltas: Vec<[[u32;8];3]>, alphas: Vec<[u32;8]>, distances: Vec<[u32;8]>, config: &crate::config::Config) -> Result<Vec<Option<[u32;8]>>>;

    /// Barrett modular reduction for 512-bit to 256-bit reduction
    fn batch_barrett_reduce(&self, x: Vec<[u32;16]>, mu: [u32;9], modulus: [u32;8], use_montgomery: bool) -> Result<Vec<[u32;8]>>;

    /// Batch 256-bit modular multiplication
    fn batch_mul(&self, a: Vec<[u32;8]>, b: Vec<[u32;8]>) -> Result<Vec<[u32;16]>>;

    /// Convert Jacobian coordinates to affine for output
    fn batch_to_affine(&self, positions: Vec<[[u32;8];3]>, modulus: [u32;8]) -> Result<(Vec<[u32;8]>, Vec<[u32;8]>)>;

    /// Safe diff mod N for collision detection (Phase 4)
    fn safe_diff_mod_n(&self, tame_dist: &[u32;8], wild_dist: &[u32;8], n: &[u32;8]) -> Result<[u32;8]>;

    /// Barrett reduction for large modular operations (Phase 5)
    fn barrett_reduce(&self, x: &[u32;16], modulus: &[u32;8], mu: &[u32;16]) -> Result<[u32;8]>;

    /// GLV optimized scalar multiplication (Phase 6)
    fn mul_glv_opt(&self, p: &[[u32;8];3], k: &[u32;8]) -> Result<[[u32;8];3]>;

    /// Modular inverse for collision solving (Phase 7)
    fn mod_inverse(&self, a: &[u32;8], modulus: &[u32;8]) -> Result<[u32;8]>;

    /// Bigint multiplication for parity tests
    fn bigint_mul(&self, a: &[u32;8], b: &[u32;8]) -> Result<[u32;16]>;

    /// Modulo operation for parity tests
    fn modulo(&self, a: &[u32;16], modulus: &[u32;8]) -> Result<[u32;8]>;

    /// Scalar multiplication with GLV for hybrid tests
    fn scalar_mul_glv(&self, p: &[[u32;8];3], k: &[u32;8]) -> Result<[[u32;8];3]>;

    /// Mod small for bias tests
    fn mod_small(&self, x: &[u32;8], modulus: u32) -> Result<u32>;

    /// Batch mod small for multi-target bias
    fn batch_mod_small(&self, points: &Vec<[[u32;8];3]>, modulus: u32) -> Result<Vec<u32>>;

    /// Rho walk for near-collisions
    fn rho_walk(&self, tortoise: &[[u32;8];3], hare: &[[u32;8];3], max_steps: u32) -> Result<RhoWalkResult>;

    /// Solve post walk for near-collisions
    fn solve_post_walk(&self, walk_result: &RhoWalkResult, targets: &Vec<[[u32;8];3]>) -> Result<Option<[u32;8]>>;

    /// Run GPU steps for parity
    fn run_gpu_steps(&self, num_steps: usize, start_state: crate::types::KangarooState) -> Result<(Vec<crate::types::Point>, Vec<crate::math::BigInt256>)>;

    /// Simulate CUDA fail for fallback tests
    fn simulate_cuda_fail(&mut self, fail: bool);

    /// Generate pre-seed positional bias points
    fn generate_preseed_pos(&self, range_min: &crate::math::BigInt256, range_width: &crate::math::BigInt256) -> Result<Vec<f64>>;

    /// Blend pre-seed positions with random and empirical data
    fn blend_proxy_preseed(&self, preseed_pos: Vec<f64>, num_random: usize, empirical_pos: Option<Vec<f64>>, weights: (f64, f64, f64)) -> Result<Vec<f64>>;

    /// Analyze pre-seed cascade for histogram generation
    fn analyze_preseed_cascade(&self, proxy_pos: &[f64], bins: usize) -> Result<(Vec<f64>, Vec<f64>)>;
}