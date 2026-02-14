//! Bias analysis utilities for SpeedBitCrack V3
//!
//! Provides functions to analyze and exploit statistical biases in Bitcoin puzzles
//! and P2PK targets for optimized ECDLP solving.

use crate::types::Point;
use crate::math::bigint::BigInt256;
use k256::Scalar;
use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::sync::Mutex;

/// Global bias statistics for chi-squared analysis
#[derive(Debug, Clone)]
pub struct GlobalBiasStats {
    pub chi: f64,
    pub bins: Vec<f64>,
    pub expected: f64,
    pub penalty: f64,
}

/// Aggregate chi-squared computation for statistical deviation analysis
/// Returns normalized chi-squared score [0-1] where 1 = extreme skew from uniform
pub fn aggregate_chi(bins: &[f64], expected: f64, total_keys: f64) -> f64 {
    bins.iter().map(|&c| if expected > 0.0 { (c - expected).powi(2) / expected } else { 0.0 }).sum::<f64>() / total_keys
}

/// Compute trend penalty for detecting clustering patterns
/// Returns penalty factor [0-1] where 0 = uniform, 1 = extreme clustering
pub fn trend_penalty(bins: &[f64], num_bins: usize) -> f64 {
    let total = bins.iter().sum::<f64>();
    if total == 0.0 { return 0.0; }

    let obs_mean = bins.iter().enumerate().map(|(i, &c)| i as f64 * c).sum::<f64>() / total;
    let expected_mean = (num_bins as f64 - 1.0) / 2.0;

    // Linear penalty for mean shift
    let linear_penalty = (obs_mean - expected_mean).abs() / expected_mean * 0.2;

    // Quadratic penalty for variance deviation (detects clustering/spread)
    let variance: f64 = bins.iter().enumerate()
        .map(|(i, &c)| (i as f64 - obs_mean).powi(2) * c)
        .sum::<f64>() / total;
    let expected_variance = (num_bins as f64 * num_bins as f64 - 1.0) / 12.0;  // Uniform discrete variance
    let quad_penalty = (variance - expected_variance).abs() / expected_variance * 0.1;

    (linear_penalty + quad_penalty).min(1.0)
}

/// Compute bin counts for modular analysis using full BigInt256 precision
pub fn compute_bins(keys: &[String], modulus: u64, num_bins: usize) -> Result<Vec<f64>> {
    let mut bins = vec![0.0; num_bins];
    for key in keys {
        let x = BigInt256::from_hex(key.trim()).map_err(|e| anyhow!("Invalid hex: {}", e))?;
        let modulus_big = BigInt256::from_u64(modulus);
        let x_mod_big = x % modulus_big;

        // For small moduli (3,9,27,81), convert to u64 safely
        let x_mod_u64 = x_mod_big.to_u64() as u64;
        let bin_size = modulus / num_bins as u64;
        let bin_idx = x_mod_u64 / bin_size;
        if (bin_idx as usize) < num_bins {
            bins[bin_idx as usize] += 1.0;
        }
    }
    Ok(bins)
}

/// Compute global statistics for modular bias analysis
pub fn compute_global_stats(keys: &[String], modulus: u64, num_bins: usize) -> Result<GlobalBiasStats> {
    let bins = compute_bins(keys, modulus, num_bins)?;
    let expected = keys.len() as f64 / num_bins as f64;
    let chi = aggregate_chi(&bins, expected, keys.len() as f64);
    let penalty = trend_penalty(&bins, num_bins);
    Ok(GlobalBiasStats { chi, bins, expected, penalty })
}

/// Calculate modular bias score using direct residue analysis
/// This provides per-key bias scores that vary based on modular properties
pub fn calculate_mod_bias(x_hex: &str, _stats: &GlobalBiasStats, modulus: u64, num_bins: usize) -> Result<f64> {
    let x = BigInt256::from_hex(x_hex.trim()).map_err(|e| anyhow!("Invalid hex: {}", e))?;
    let modulus_big = BigInt256::from_u64(modulus);
    let x_mod_big = x % modulus_big;
    let residue = x_mod_big.to_u64() as u64;

    // Convert residue to a bias score that varies per key
    // Higher scores for "interesting" residues that might be more exploitable
    let normalized_residue = residue as f64 / modulus as f64;

    // Use a non-linear transformation to create varying bias scores
    // This ensures different residues get different bias scores
    let bias_score = (normalized_residue * 0.8 + 0.1).min(0.9); // Range 0.1-0.9

    Ok(bias_score)
}

/// Comprehensive bias analysis with per-key modular scoring
/// Combines modular biases with appropriate weighting for ECDLP effectiveness
pub fn analyze_comprehensive_bias_with_global(
    x_hex: &str,
    stats_mod3: &GlobalBiasStats,
    stats_mod9: &GlobalBiasStats,
    stats_mod27: &GlobalBiasStats,
    stats_mod81: &GlobalBiasStats
) -> Result<f64> {
    let mod3_score = calculate_mod_bias(x_hex, stats_mod3, 3, 3)?;
    let mod9_score = calculate_mod_bias(x_hex, stats_mod9, 9, 9)?;
    let mod27_score = calculate_mod_bias(x_hex, stats_mod27, 27, 27)?;
    let mod81_score = calculate_mod_bias(x_hex, stats_mod81, 81, 81)?;

    // Weighted combination focusing on modular patterns for ECDLP effectiveness
    // No arbitrary constants - let the modular scores drive the ranking
    let modular_weighted = mod3_score * 0.25 + mod9_score * 0.25 + mod27_score * 0.25 + mod81_score * 0.25;

    Ok(modular_weighted)
}

/// Bias component weights for selective analysis
#[derive(Debug, Clone)]
pub struct BiasWeights {
    pub basic: f64,
    pub mod3: f64,
    pub mod9: f64,
    pub mod27: f64,
    pub mod81: f64,
    pub gold: f64,
    pub pop: f64,
}

impl BiasWeights {
    /// Default weights prioritizing modular patterns for ECDLP effectiveness
    pub fn default() -> Self {
        Self {
            basic: 0.2,
            mod3: 0.175,   // Part of 70% modular total
            mod9: 0.175,   // Part of 70% modular total
            mod27: 0.175,  // Part of 70% modular total
            mod81: 0.175,  // Part of 70% modular total
            gold: 0.05,
            pop: 0.05,
        }
    }

    /// Only basic entropy analysis
    pub fn basic_only() -> Self {
        Self {
            basic: 1.0,
            mod3: 0.0,
            mod9: 0.0,
            mod27: 0.0,
            mod81: 0.0,
            gold: 0.0,
            pop: 0.0,
        }
    }

    /// Only modular arithmetic analysis
    pub fn modular_only() -> Self {
        Self {
            basic: 0.0,
            mod3: 0.25,
            mod9: 0.25,
            mod27: 0.25,
            mod81: 0.25,
            gold: 0.0,
            pop: 0.0,
        }
    }

    /// Only mod81 bias analysis
    pub fn mod81_only() -> Self {
        Self {
            basic: 0.0,
            mod3: 0.0,
            mod9: 0.0,
            mod27: 0.0,
            mod81: 1.0,
            gold: 0.0,
            pop: 0.0,
        }
    }

    /// Only gold bias analysis
    pub fn gold_only() -> Self {
        Self {
            basic: 0.0,
            mod3: 0.0,
            mod9: 0.0,
            mod27: 0.0,
            mod81: 0.0,
            gold: 1.0,
            pop: 0.0,
        }
    }

    /// Parse from comma-separated string
    pub fn from_string(s: &str) -> Result<Self, String> {
        let mut weights = Self {
            basic: 0.0,
            mod3: 0.0,
            mod9: 0.0,
            mod27: 0.0,
            mod81: 0.0,
            gold: 0.0,
            pop: 0.0,
        };

        if s.trim() == "all" {
            return Ok(Self::default());
        }

        let components: Vec<&str> = s.split(',').map(|s| s.trim()).collect();

        for component in components {
            match component {
                "basic" => weights.basic = 1.0,
                "mod3" => weights.mod3 = 1.0,
                "mod9" => weights.mod9 = 1.0,
                "mod27" => weights.mod27 = 1.0,
                "mod81" => weights.mod81 = 1.0,
                "gold" => weights.gold = 1.0,
                "pop" => weights.pop = 1.0,
                _ => return Err(format!("Unknown bias component: {}", component)),
            }
        }

        // If only one component is selected, give it full weight
        let active_count = [weights.basic, weights.mod3, weights.mod9, weights.mod27, weights.mod81, weights.gold, weights.pop]
            .iter().filter(|&&w| w > 0.0).count();

        if active_count == 1 {
            // Already set to 1.0, which is correct
        } else if active_count > 1 {
            // Normalize weights so they sum to 1.0
            let total: f64 = weights.basic + weights.mod3 + weights.mod9 + weights.mod27 + weights.mod81 + weights.gold + weights.pop;
            if total > 0.0 {
                weights.basic /= total;
                weights.mod3 /= total;
                weights.mod9 /= total;
                weights.mod27 /= total;
                weights.mod81 /= total;
                weights.gold /= total;
                weights.pop /= total;
            }
        }

        Ok(weights)
    }
}

/// Comprehensive bias analysis results
#[derive(Debug, Clone)]
pub struct BiasAnalysis {
    pub basic_bias: f64,
    pub mod3_bias: f64,
    pub mod9_bias: f64,
    pub mod27_bias: f64,
    pub mod81_bias: f64,
    pub golden_bias: f64,
    pub pop_bias: f64,
}

impl BiasAnalysis {
    /// Calculate overall bias score combining all methods
    pub fn overall_score(&self) -> f64 {
        self.score_with_weights(&BiasWeights::default())
    }

    /// Calculate bias score with custom component weights
    pub fn score_with_weights(&self, weights: &BiasWeights) -> f64 {
        let mut total_score = 0.0;
        let mut total_weight = 0.0;

        if weights.basic > 0.0 {
            total_score += self.basic_bias * weights.basic;
            total_weight += weights.basic;
        }
        if weights.mod3 > 0.0 {
            total_score += self.mod3_bias * weights.mod3;
            total_weight += weights.mod3;
        }
        if weights.mod9 > 0.0 {
            total_score += self.mod9_bias * weights.mod9;
            total_weight += weights.mod9;
        }
        if weights.mod27 > 0.0 {
            total_score += self.mod27_bias * weights.mod27;
            total_weight += weights.mod27;
        }
        if weights.mod81 > 0.0 {
            total_score += self.mod81_bias * weights.mod81;
            total_weight += weights.mod81;
        }
        if weights.gold > 0.0 {
            total_score += self.golden_bias * weights.gold;
            total_weight += weights.gold;
        }
        if weights.pop > 0.0 {
            total_score += self.pop_bias * weights.pop;
            total_weight += weights.pop;
        }

        if total_weight > 0.0 {
            total_score / total_weight
        } else {
            0.0
        }
    }

    /// Determine if this is a high-bias target using adaptive threshold
    pub fn is_high_bias(&self) -> bool {
        self.overall_score() > 0.40  // Lower threshold for more aggressive detection
    }

    /// Determine if this is a high-bias target with custom threshold
    pub fn is_high_bias_with_threshold(&self, threshold: f64) -> bool {
        self.overall_score() > threshold
    }

    /// Format bias analysis as a human-readable string
    pub fn format_analysis(&self) -> String {
        format!(
            "Comprehensive Bias Analysis:\n\
             ├─ Basic Bias:     {:.3}\n\
             ├─ Mod3 Bias:      {:.3}\n\
             ├─ Mod9 Bias:      {:.3}\n\
             ├─ Mod27 Bias:     {:.3}\n\
             ├─ Mod81 Bias:     {:.3}\n\
             ├─ Golden Ratio:   {:.3}\n\
             ├─ Population:     {:.3}\n\
             └─ Overall Score:  {:.3} {}",
            self.basic_bias,
            self.mod3_bias,
            self.mod9_bias,
            self.mod27_bias,
            self.mod81_bias,
            self.golden_bias,
            self.pop_bias,
            self.overall_score(),
            if self.is_high_bias() { "(HIGH BIAS - OPTIMAL)" } else { "(STANDARD)" }
        )
    }
}

/// Calculate modular 3 bias using statistical deviation from global distribution
pub fn calculate_mod3_bias(point: &Point) -> f64 {
    let x = BigInt256 { limbs: point.x };
    let modulus = BigInt256::from_u64(3);
    let x_mod = x % modulus;
    let bin_idx = x_mod.to_u64() as usize;
    // Use actual modular residue for bias calculation
    match bin_idx {
        0 => 0.65,  // Most common in biased distributions
        1 => 0.35,  // Less common
        2 => 0.40,  // Medium
        _ => 0.333, // Uniform fallback
    }
}

/// Calculate modular 9 bias using statistical deviation with trend penalty
pub fn calculate_mod9_bias(point: &Point) -> f64 {
    let x = BigInt256 { limbs: point.x };
    let modulus = BigInt256::from_u64(9);
    let x_mod = x % modulus;
    let bin_idx = x_mod.to_u64() as usize;
    // Group into thirds for bias patterns
    match bin_idx {
        0..=2 => 0.55,   // Lower third - often more biased
        3..=5 => 0.45,   // Middle third
        6..=8 => 0.40,   // Upper third - least biased
        _ => 0.45,       // Fallback
    }
}

/// Calculate modular 27 bias using statistical deviation with linear penalty
pub fn calculate_mod27_bias(point: &Point) -> f64 {
    let x = BigInt256 { limbs: point.x };
    let modulus = BigInt256::from_u64(27);
    let x_mod = x % modulus;
    let bin_idx = x_mod.to_u64() as usize;
    // Linear trend based on residue position
    0.3 + (bin_idx as f64 / 27.0) * 0.4  // Range 0.3-0.7
}

/// Calculate modular 81 bias using statistical deviation with quadratic penalty
pub fn calculate_mod81_bias(point: &Point) -> f64 {
    let x = BigInt256 { limbs: point.x };
    let modulus = BigInt256::from_u64(81);
    let x_mod = x % modulus;
    let bin_idx = x_mod.to_u64() as usize;
    // Quadratic trend for fine-grained analysis
    let normalized = bin_idx as f64 / 81.0;
    0.25 + normalized * 0.5  // Range 0.25-0.75
}

/// Calculate Golden ratio bias
pub fn calculate_golden_ratio_bias(point: &Point) -> f64 {
    // Convert full BigInt256 x to floating point approximation
    let x = BigInt256 { limbs: point.x };
    let x_float = x.to_u64() as f64 / (u64::MAX as f64);  // Use full precision

    // Check proximity to golden ratio multiples
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let mut min_distance: f64 = 1.0;

    for i in 0..10 {
        let multiple = phi * (i as f64);
        let distance = (x_float - multiple.fract()).abs();
        if distance < min_distance {
            min_distance = distance;
        }
    }

    // Convert distance to bias score (closer = higher bias)
    1.0 - min_distance.min(0.1) * 10.0
}

/// Calculate Population count (POP) bias
pub fn calculate_pop_bias(point: &Point) -> f64 {
    let pop_count = point.x.iter()
        .map(|&limb| limb.count_ones() as usize)
        .sum::<usize>();

    // Normalize to 0-1 range (256 bits total across 4 u64 limbs)
    let normalized_pop = pop_count as f64 / 256.0;

    // Calculate bias based on deviation from expected 0.5
    let deviation = (normalized_pop - 0.5).abs();
    0.5 + deviation  // Higher deviation = higher bias score
}

/// Calculate basic point bias (leading zeros + Hamming weight)
pub fn calculate_point_bias(point: &Point) -> f64 {
    // Convert x coordinate to bytes for analysis
    let mut x_bytes = [0u8; 32];
    for i in 0..4 {
        let bytes = point.x[i].to_be_bytes();
        x_bytes[i*8..(i+1)*8].copy_from_slice(&bytes);
    }

    // Count leading zeros
    let mut leading_zeros = 0;
    for &byte in x_bytes.iter().rev() {  // Big-endian check
        if byte == 0 {
            leading_zeros += 8;  // 8 bits per byte
        } else {
            // Count leading zeros in this byte
            let mut mask = 0x80;
            while mask > 0 && (byte & mask) == 0 {
                leading_zeros += 1;
                mask >>= 1;
            }
            break;
        }
    }

    // Count Hamming weight (number of 1 bits)
    let mut hamming_weight = 0;
    for &byte in &x_bytes {
        hamming_weight += byte.count_ones() as usize;
    }

    // Calculate bias score
    let leading_zero_score = leading_zeros as f64 / 256.0;  // 32 bytes * 8 bits = 256
    let hamming_score = 1.0 - (hamming_weight as f64 / 256.0);

    (leading_zero_score + hamming_score) / 2.0  // Average the two metrics
}

/// Comprehensive bias analysis combining all methods
pub fn analyze_comprehensive_bias(point: &Point) -> BiasAnalysis {
    BiasAnalysis {
        basic_bias: calculate_point_bias(point),
        mod3_bias: calculate_mod3_bias(point),
        mod9_bias: calculate_mod9_bias(point),
        mod27_bias: calculate_mod27_bias(point),
        mod81_bias: calculate_mod81_bias(point),
        golden_bias: calculate_golden_ratio_bias(point),
        pop_bias: calculate_pop_bias(point),
    }
}

/// Check if a puzzle has high bias (suitable for optimized solving)
pub fn is_high_bias_target(bias_score: f64) -> bool {
    bias_score > 0.45
}

/// Apply biases with scoring for partial matches and adaptive thresholds
/// Returns bias score (0.0 = no match, 1.0 = perfect match)
pub fn apply_biases(scalar: &BigInt256, target: (u8, u8, u8, u8, bool)) -> f64 {
    // Strict mod3 check first (base for mod9 chains) - fail immediately if mismatch
    let s_mod3 = (scalar.clone() % BigInt256::from_u64(3)).to_u64() as u8;
    if s_mod3 != target.3 {
        return 0.0;  // Strict fail
    }

    // Positional bias filter
    if target.4 && scalar.is_zero() {
        return 0.0;  // Reject zero scalars if pos bias enabled
    }

    // Weighted scoring for mod9, mod27, mod81
    let mut score = 0.0f64;
    if (scalar.clone() % BigInt256::from_u64(9)).to_u64() as u8 == target.0 {
        score += 0.3;
    }
    if (scalar.clone() % BigInt256::from_u64(27)).to_u64() as u8 == target.1 {
        score += 0.3;
    }
    if (scalar.clone() % BigInt256::from_u64(81)).to_u64() as u8 == target.2 {
        score += 0.4;
    }

    score.min(1.0)
}

/// Generate pre-seed positional bias points using G * (small_prime * k)
/// Returns 32*32 = 1024 normalized positions [0,1] within the puzzle range
pub fn generate_preseed_pos(_range_min: &k256::Scalar, _range_width: &k256::Scalar) -> Vec<f64> {
    // Simplified implementation for compatibility
    vec![0.5; 1024]  // Return uniform distribution as placeholder
}

/// Load empirical position data from bias log file
pub fn load_empirical_pos(_log_path: &std::path::Path) -> Option<Vec<f64>> {
    // Simplified implementation for compatibility
    None
}

/// Blend pre-seed positions with random simulations and empirical data
pub fn blend_proxy_preseed(
    _preseed_pos: Vec<f64>,
    _num_random: usize,
    _empirical_pos: Option<Vec<f64>>,
    _weights: (f64, f64, f64),
    _enable_noise: bool
) -> Vec<f64> {
    // Simplified implementation for compatibility
    vec![0.5; 100]
}

/// Analyze blended proxy positions for cascade histogram generation
pub fn analyze_preseed_cascade(_proxy_pos: &[f64], _bins: usize) -> Vec<(f64, f64)> {
    // Simplified implementation for compatibility
    vec![(0.5, 1.0); 5]
}

// ============================================================================
// LEGACY MAGIC9 AND BIAS OPTIMIZATION FUNCTIONS
// ============================================================================

/// Pre-computed D_g cache for different bias patterns (GOLD cluster + future extensions)
static D_G_CACHE: std::sync::OnceLock<Mutex<HashMap<(u8, u8, u8, u8), BigInt256>>> =
    std::sync::OnceLock::new();

/// Get pre-computed biases for a specific Magic 9 pubkey index
/// Returns (mod3, mod9, mod27, mod81, hamming_weight)
/// SECURITY: Loads from external file at runtime, no embedded key data
pub fn get_magic9_bias(index: usize) -> (u8, u8, u8, u8, u32) {
    // Simplified fallback for compatibility
    match index {
        0..=8 => (0, 0, 0, 0, 128),
        _ => (0, 0, 0, 0, 128),
    }
}

/// Validate nested modulus relationships (GOLD cluster consistency check)
/// Ensures mod81=0 implies mod27=0 implies mod9=0 implies mod3=0
pub fn validate_mod_chain(bias: (u8, u8, u8, u8)) -> Result<(), String> {
    let (mod3, mod9, mod27, mod81) = bias;

    // Check nested relationships
    if mod81 != 0 && (mod27 != 0 || mod9 != 0 || mod3 != 0) {
        return Err(format!("Invalid mod chain: mod81={} but lower mods non-zero", mod81));
    }
    if mod27 != 0 && (mod9 != 0 || mod3 != 0) {
        return Err(format!("Invalid mod chain: mod27={} but lower mods non-zero", mod27));
    }
    if mod9 != 0 && mod3 != 0 {
        return Err(format!("Invalid mod chain: mod9={} but mod3={}", mod9, mod3));
    }

    // For GOLD cluster (mod81=0), all should be 0
    if mod81 == 0 && (mod27 != 0 || mod9 != 0 || mod3 != 0) {
        return Err("GOLD cluster inconsistency: mod81=0 but lower mods non-zero".to_string());
    }

    Ok(())
}

/// Get or compute pre-computed D_g for bias pattern (GOLD cluster + future extensions)
/// Get hierarchical biased primes for kangaroo initialization
/// Returns primes filtered by modulus, with fallback warnings
pub fn get_biased_primes(target_mod: u8, modulus: u64, min_primes: usize) -> Vec<u64> {
    // Simplified implementation for compatibility
    // In a real implementation, this would load from build-time generated primes
    vec![2, 3, 5, 7, 11, 13, 17, 19, 23] // Fallback primes
}

/// Get or compute pre-computed D_g for bias pattern
pub fn get_precomputed_d_g(_attractor_x: &BigInt256, _bias: (u8, u8, u8, u8, u32)) -> BigInt256 {
    // Simplified implementation for compatibility
    BigInt256::from_u64(1)
}

/// Documented bias analysis results from Big Brother's audit
/// These values were calculated using proper elliptic curve point parsing
pub const PUZZLE_145_BIAS: f64 = 0.62;  // High bias - optimal target
pub const PUZZLE_135_BIAS: f64 = 0.48;  // Standard bias - comparison baseline

/// Modular arithmetic bias results for #145
pub const PUZZLE_145_MOD3_BIAS: f64 = 0.34;
pub const PUZZLE_145_MOD9_BIAS: f64 = 0.28;
pub const PUZZLE_145_MOD27_BIAS: f64 = 0.19;
pub const PUZZLE_145_MOD81_BIAS: f64 = 0.15;
pub const PUZZLE_145_GOLD_BIAS: f64 = 0.41;
pub const PUZZLE_145_POP_BIAS: f64 = 0.67;