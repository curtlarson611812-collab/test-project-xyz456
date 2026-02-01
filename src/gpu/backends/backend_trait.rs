//! GPU Backend Trait Definition
//!
//! Unified interface for GPU acceleration backends (CUDA, Vulkan, CPU)

use crate::kangaroo::collision::Trap;
use anyhow::Result;

/// Unified GPU backend trait for hybrid Vulkan+CUDA acceleration
/// Enables dispatch of kangaroo operations to appropriate GPU backends
#[async_trait::async_trait]
pub trait GpuBackend: Send + Sync {
    /// Create a new backend instance
    async fn new() -> Result<Self> where Self: Sized;

    /// Precompute jump table for kangaroo algorithm (G * 2^i for i in 0..32)
    /// Returns (positions, distances) for optimized jump operations
    fn precomp_table(&self, primes: Vec<[u32;8]>, base: [u32;8]) -> Result<(Vec<[[u32;8];3]>, Vec<[u32;8]>)>;

    /// Execute batch kangaroo stepping (parallel walk operations)
    /// Updates positions/distances, returns collision traps
    fn step_batch(&self, positions: &mut Vec<[[u32;8];3]>, distances: &mut Vec<[u32;8]>, types: &Vec<u32>) -> Result<Vec<Trap>>;

    /// Batch modular inverse for discrete logarithm denominator calculation
    fn batch_inverse(&self, inputs: Vec<[u32;8]>, modulus: [u32;8]) -> Result<Vec<[u32;8]>>;

    /// Batch collision solving for private key recovery
    fn batch_solve(&self, alphas: Vec<[u32;8]>, betas: Vec<[u32;8]>) -> Result<Vec<[u64;4]>>;

    /// Advanced batch collision solving with target points
    fn batch_solve_collision(&self, alpha_t: Vec<[u32;8]>, alpha_w: Vec<[u32;8]>, beta_t: Vec<[u32;8]>, beta_w: Vec<[u32;8]>, target: Vec<[u32;8]>, n: [u32;8]) -> Result<Vec<[u32;8]>>;

    /// Barrett modular reduction for 512-bit to 256-bit reduction
    fn batch_barrett_reduce(&self, x: Vec<[u32;16]>, mu: [u32;9], modulus: [u32;8], use_montgomery: bool) -> Result<Vec<[u32;8]>>;

    /// Batch 256-bit modular multiplication
    fn batch_mul(&self, a: Vec<[u32;8]>, b: Vec<[u32;8]>) -> Result<Vec<[u32;16]>>;

    /// Convert Jacobian coordinates to affine for output
    fn batch_to_affine(&self, positions: Vec<[[u32;8];3]>, modulus: [u32;8]) -> Result<(Vec<[u32;8]>, Vec<[u32;8]>)>;
}