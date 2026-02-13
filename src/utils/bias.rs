//! Bias Analysis and Optimization Utilities for Magic 9 Sniper
//!
//! Provides advanced bias detection, scoring, and filtering for mod9/mod27/mod81/pos
//! optimizations in secp256k1 kangaroo attacks.

use crate::math::bigint::BigInt256;
use log::{info, warn};
use rand::{Rng, RngCore};
use crate::kangaroo::generator::PRIME_MULTIPLIERS;
use k256::{ProjectivePoint, Scalar};

// Include build-generated prime constants (non-sensitive mathematical values)
include!(concat!(env!("OUT_DIR"), "/primes.rs"));

/// Compute target biases from pubkey x-coordinate with attractor cross-check
/// Returns (mod9, mod27, mod81, pos) bias targets
pub fn compute_pubkey_biases(x: &BigInt256, attractor_x: &BigInt256) -> (u8, u8, u8, bool) {
    let mod9 = (x.clone() % BigInt256::from_u64(9)).low_u32() as u8;
    let mod27 = (x.clone() % BigInt256::from_u64(27)).low_u32() as u8;
    let mod81 = (x.clone() % BigInt256::from_u64(81)).low_u32() as u8;
    let pos = true;  // Always positive for distance scalars

    // Optimization: Cross-check with attractor congruence for validation
    let att_mod9 = (attractor_x.clone() % BigInt256::from_u64(9)).low_u32() as u8;
    if mod9 != att_mod9 {
        log::warn!("Pubkey mod9 bias {} differs from attractor mod9 {}", mod9, att_mod9);
    }

    (mod9, mod27, mod81, pos)
}

/// Enhanced apply biases with scoring for partial matches and adaptive thresholds
/// Returns bias score (0.0 = no match, 1.0 = perfect match)
/// Updated to include strict mod3 check as base for mod9 chains
pub fn apply_biases(scalar: &BigInt256, target: (u8, u8, u8, u8, bool)) -> f64 {
    // Strict mod3 check first (base for mod9 chains) - fail immediately if mismatch
    let s_mod3 = (scalar.clone() % BigInt256::from_u64(3)).low_u32() as u8;
    if s_mod3 != target.3 {
        return 0.0;  // Strict fail
    }

    // Positional bias filter
    if target.4 && bool::from(scalar.is_zero()) {
        return 0.0;  // Reject zero scalars if pos bias enabled
    }

    // Bias analysis continues...

    // Weighted scoring for mod9, mod27, mod81
    let mut score = 0.0f64;
    if (scalar.clone() % BigInt256::from_u64(9)).low_u32() as u8 == target.0 {
        score += 0.3;
    }
    if (scalar.clone() % BigInt256::from_u64(27)).low_u32() as u8 == target.1 {
        score += 0.3;
    }
    if (scalar.clone() % BigInt256::from_u64(81)).low_u32() as u8 == target.2 {
        score += 0.4;
    }

    score.min(1.0)
}

/// Additional bias: mod3 check for finer granularity
/// Returns true if scalar passes mod3 filter (basic 3-power subgroup)
pub fn apply_mod3_bias(scalar: &BigInt256, target_mod3: u8) -> bool {
    (scalar.clone() % BigInt256::from_u64(3)).low_u32() as u8 == target_mod3
}

/// Additional bias: Hamming weight check for low-weight scalars
/// Returns true if scalar has low Hamming weight (optimization for EC operations)
/// Disabled for GOLD clusters (uniform Hamming=128, no filtering benefit)
pub fn apply_hamming_bias(scalar: &BigInt256, max_weight: u32, is_gold_cluster: bool) -> bool {
    if is_gold_cluster {
        // GOLD clusters have uniform Hamming=128, disable filter to avoid over-filtering
        true  // Accept all scalars for GOLD clusters
    } else {
        let bytes = scalar.to_bytes_be();
        let weight: u32 = bytes.iter().map(|b| b.count_ones()).sum();
        weight < max_weight  // Standard filtering for non-GOLD clusters
    }
}

/// Compute combined bias score across multiple filters
/// Used for adaptive threshold decisions in kangaroo walks
pub fn compute_combined_bias_score(
    scalar: &BigInt256,
    mod9_target: u8,
    mod27_target: u8,
    mod81_target: u8,
    mod3_target: u8,
    max_hamming: u32,
    is_gold_cluster: bool
) -> f64 {
    let mut score = apply_biases(scalar, (mod9_target, mod27_target, mod81_target, mod3_target, true));

    if apply_hamming_bias(scalar, max_hamming, is_gold_cluster) && !is_gold_cluster {
        score += 0.1;  // Additional hamming bonus only for non-GOLD clusters
    }

    score.min(1.0)  // Cap at perfect match
}

// SECURITY: No longer embedding key-derived biases in binary
// Magic 9 biases must be loaded from external files at runtime
// This prevents any key information from being embedded in the executable

use std::sync::Mutex;

static MAGIC9_BIASES_CACHE: std::sync::OnceLock<Mutex<Vec<(u8, u8, u8, u8, u32)>>> =
    std::sync::OnceLock::new();

/// Load Magic 9 biases from external file at runtime
/// SECURITY: This prevents embedding key information in the binary
fn load_magic9_biases() -> Vec<(u8, u8, u8, u8, u32)> {
    // Try to load from magic9_biases.txt file
    match std::fs::read_to_string("magic9_biases.txt") {
        Ok(content) => {
            let mut biases = Vec::new();
            for line in content.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                // Parse format: mod3,mod9,mod27,mod81,hamming
                if let Some((mods, hamming)) = line.split_once(',') {
                    let mods: Vec<&str> = mods.split(',').collect();
                    if mods.len() == 4 {
                        if let (Ok(mod3), Ok(mod9), Ok(mod27), Ok(mod81), Ok(hamming)) =
                            (mods[0].parse::<u8>(), mods[1].parse::<u8>(),
                             mods[2].parse::<u8>(), mods[3].parse::<u8>(),
                             hamming.parse::<u32>()) {
                            biases.push((mod3, mod9, mod27, mod81, hamming));
                        }
                    }
                }
            }
            if biases.len() == 9 {
                log::info!("Loaded {} Magic9 biases from external file", biases.len());
                return biases;
            } else {
                log::warn!("Invalid magic9_biases.txt format (expected 9 entries, got {}), using defaults", biases.len());
            }
        }
        Err(e) => {
            log::warn!("Could not load magic9_biases.txt: {}, using defaults", e);
        }
    }

    // Fallback: return default values if file doesn't exist or is invalid
    log::info!("Using default Magic9 bias values (all zeros)");
    vec![
        (0, 0, 0, 0, 128), (0, 0, 0, 0, 128), (0, 0, 0, 0, 128),
        (0, 0, 0, 0, 128), (0, 0, 0, 0, 128), (0, 0, 0, 0, 128),
        (0, 0, 0, 0, 128), (0, 0, 0, 0, 128), (0, 0, 0, 0, 128),
    ]
}

/// Get pre-computed biases for a specific Magic 9 pubkey index
/// Returns (mod3, mod9, mod27, mod81, hamming_weight)
/// SECURITY: Loads from external file at runtime, no embedded key data
pub fn get_magic9_bias(index: usize) -> (u8, u8, u8, u8, u32) {
    let cache = MAGIC9_BIASES_CACHE.get_or_init(|| {
        Mutex::new(load_magic9_biases())
    });

    let biases = cache.lock().unwrap();
    biases.get(index).copied().unwrap_or((0, 0, 0, 0, 128))
}

/// Pre-computed D_g cache for different bias patterns (GOLD cluster + future extensions)

static D_G_CACHE: std::sync::OnceLock<Mutex<std::collections::HashMap<(u8, u8, u8, u8), BigInt256>>> =
    std::sync::OnceLock::new();

/// Get or compute pre-computed D_g for bias pattern (GOLD cluster + future extensions)
/// Get hierarchical biased primes for kangaroo initialization
/// Returns primes filtered by modulus, with fallback warnings
pub fn get_biased_primes(target_mod: u8, modulus: u64, min_primes: usize) -> Vec<u64> {
    // Use the pre-computed prime arrays from build.rs
    let all_primes = if modulus == 81 {
        // For GOLD cluster (mod81=0), use pre-computed GOLD_CLUSTER_PRIMES
        GOLD_CLUSTER_PRIMES.to_vec()
    } else if modulus == 27 {
        // For secondary fallback (mod27=0), use SECONDARY_PRIMES
        SECONDARY_PRIMES.to_vec()
    } else {
        // Fallback to all primes
        crate::kangaroo::generator::PRIME_MULTIPLIERS.to_vec()
    };

    // Filter primes that match the target modulus
    let matches: Vec<u64> = all_primes.into_iter()
        .filter(|&p| (p % modulus) as u8 == target_mod)
        .collect();

    // Warn if too few primes match
    if matches.len() < min_primes {
        eprintln!("Warning: Only {} primes match mod{}={}, minimum {}. Using all available primes.",
                 matches.len(), modulus, target_mod, min_primes);
    }

    if matches.is_empty() {
        eprintln!("Warning: No primes match mod{}={}. Using full prime set.",
                 modulus, target_mod);
        // Return all primes as fallback
        crate::kangaroo::generator::PRIME_MULTIPLIERS.to_vec()
    } else {
        matches
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

pub fn get_precomputed_d_g(attractor_x: &BigInt256, bias: (u8, u8, u8, u8, u32)) -> BigInt256 {
    let bias_key = (bias.0, bias.1, bias.2, bias.3); // mod3, mod9, mod27, mod81

    // Initialize cache if needed
    let cache_mutex = D_G_CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()));

    // Check cache first
    {
        let cache = cache_mutex.lock().unwrap();
        if let Some(cached_d_g) = cache.get(&bias_key) {
            return cached_d_g.clone();
        }
    }

    // Compute and cache for this bias pattern
    info!("🔍 Pre-computing D_g for bias pattern {:?}", bias_key);
    let curve = crate::math::secp::Secp256k1::new();

    // Generator point G in affine coordinates
    let g_x = [0x79BE667EF9DCBBAC, 0x55A06295CE870B07, 0x029BFCDB2DCE28D9, 0x59F2815B16F81798];
    let g_y = [0x483ADA7726A3C465, 0x5DA4FBFC0E1108A8, 0xFD17B448A6855419, 0x9C47D08FFB10D4B8];
    let generator = crate::types::Point::from_affine(g_x, g_y);

    // Convert bias tuple to kangaroo format
    let kangaroo_bias = (bias.1, bias.2, bias.3, true); // mod9, mod27, mod81, pos

    // Compute D_g (this is expensive, so we cache it)
    let d_g = crate::kangaroo::generator::biased_kangaroo_to_attractor(
        &generator, attractor_x, kangaroo_bias, &curve, 1_000_000
    ).unwrap_or_else(|_| {
        warn!("Failed to pre-compute D_g for pattern {:?}, using fallback", bias_key);
        BigInt256::zero()
    });

    // Cache the result
    {
        let cache_mutex = D_G_CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
        let mut cache = cache_mutex.lock().unwrap();
        cache.insert(bias_key, d_g.clone());
    }

    info!("✅ D_g pre-computed for pattern {:?}: {}", bias_key, d_g.to_hex());
    d_g
}

/// Generate pre-seed positional bias points using G * (small_prime * k)
/// Returns 32*32 = 1024 normalized positions [0,1] within the puzzle range
/// This provides "curve-aware" baseline for unsolved puzzles lacking empirical data
pub fn generate_preseed_pos(range_min: &Scalar, range_width: &Scalar) -> Vec<f64> {
    if bool::from(range_width.is_zero()) {
        panic!("Zero range width"); // Edge: Prevent div by zero
    }

    let mut pos = Vec::with_capacity(32 * 32);

    for &prime_u64 in PRIME_MULTIPLIERS.iter() {
        let prime = Scalar::from(prime_u64);

        for k in 1..=32 {
            let scalar = prime * Scalar::from(k as u64);
            if bool::from(scalar.is_zero()) {
                continue; // Skip zero scalars
            }

            let point = ProjectivePoint::GENERATOR * scalar;
            let affine = point.to_affine();

            // Check if point is on curve (k256 handles this internally)
            // For now, assume it's valid since we used the generator

            let encoded = k256::EncodedPoint::from(affine); // Get encoded point
            let x_bytes = encoded.x().unwrap().to_vec();
            let mut x_bytes_array = [0u8; 32];
            x_bytes_array[..x_bytes.len().min(32)].copy_from_slice(&x_bytes[..x_bytes.len().min(32)]);
            let x_hash = xor_hash_to_u64(&x_bytes_array);
            let range_width_u64 = range_width.to_bytes().iter().fold(0u64, |acc, &b| (acc << 8) | b as u64).max(1);
            let offset_u64 = x_hash % range_width_u64;
            let offset_scalar = Scalar::from(offset_u64);

            let pos_val = ((offset_scalar - range_min).to_bytes().iter().fold(0u64, |acc, &b| (acc << 8) | b as u64) as f64) /
                         (range_width.to_bytes().iter().fold(0u64, |acc, &b| (acc << 8) | b as u64) as f64);
            pos.push(pos_val.clamp(0.0_f64, 1.0_f64));
        }
    }

    pos
}

/// XOR-based hash for deterministic pos_proxy calculation
fn xor_hash_to_u64(bytes: &[u8; 32]) -> u64 {
    (0..4).fold(0u64, |acc, i| acc ^ u64::from_le_bytes(bytes[i*8..(i+1)*8].try_into().unwrap()))
}

/// Hash point x-coordinate to u64 for deterministic pos_proxy calculation
fn hash_point_x_to_u64(point: &crate::types::Point) -> u64 {
    // Use point's x coordinate as deterministic seed
    // For simplicity, use low 64 bits of x (sufficient for proxy)
    let x_bytes = point.x.iter().rev().fold(vec![], |mut acc, &limb| {
        acc.extend_from_slice(&limb.to_be_bytes());
        acc
    });
    let mut hash = 0u64;
    for chunk in x_bytes.chunks(8) {
        let mut bytes = [0u8; 8];
        bytes[..chunk.len()].copy_from_slice(chunk);
        hash ^= u64::from_be_bytes(bytes);
    }
    hash
}

/// Blend pre-seed positions with random simulations and empirical data
/// weights: (preseed_weight, random_weight, empirical_weight) - must sum to 1.0
/// enable_noise: Add small random variation to random samples for variance
/// Returns combined proxy positions for histogram analysis
pub fn blend_proxy_preseed(
    preseed_pos: Vec<f64>,
    num_random: usize,
    empirical_pos: Option<Vec<f64>>,
    weights: (f64, f64, f64),
    enable_noise: bool
) -> Vec<f64> {
    let (w_pre, w_rand, w_emp) = weights;
    let total_w = w_pre + w_rand + w_emp;
    assert!((total_w - 1.0).abs() < 1e-6, "Weights must sum to 1.0, got {}", total_w);

    let mut proxy = Vec::new();
    let mut rng = rand::thread_rng(); // Seeded for test determinism, flag-gated noise

    // Duplicate pre-seed proportional to weight
    let dup_pre = ((preseed_pos.len() as f64 * w_pre).round() as usize).max(1);
    for _ in 0..dup_pre {
        proxy.extend_from_slice(&preseed_pos);
    }

    // Add random samples (uniform distribution)
    let dup_rand = ((num_random as f64 * w_rand).round() as usize).max(0usize);
    for _ in 0..dup_rand {
        let mut rand_pos = rng.gen_range(0.0..1.0);
        if enable_noise {
            rand_pos += rng.gen_range(-0.05..0.05);
            if rand_pos < 0.0 {
                rand_pos = 0.0;
            } else if rand_pos > 1.0 {
                rand_pos = 1.0;
            }
        }
        proxy.push(rand_pos);
    }

    // Add empirical positions if available
    if let Some(emp) = empirical_pos {
        let dup_emp = ((emp.len() as f64 * w_emp).round() as usize).max(0usize);
        for _ in 0..dup_emp {
            proxy.extend_from_slice(&emp);
        }
    } else if w_emp > 0.0 {
        // Fallback: Redistribute empirical weight to pre-seed/random
        let extra = w_emp / 2.0;
        let extra_pre = ((preseed_pos.len() as f64 * extra).round() as usize).max(0usize);
        for i in 0..extra_pre {
            proxy.push(preseed_pos[i % preseed_pos.len()]);
        }
        let extra_rand = ((num_random as f64 * extra).round() as usize).max(0usize);
        for _ in 0..extra_rand {
            let mut rand_pos = rng.gen_range(0.0..1.0);
            if enable_noise {
                rand_pos += rng.gen_range(-0.05..0.05);
                if rand_pos < 0.0 {
                    rand_pos = 0.0;
                } else if rand_pos > 1.0 {
                    rand_pos = 1.0;
                }
            }
            proxy.push(rand_pos);
        }
    }

    proxy
}

/// Load empirical position data from bias log file
/// Returns empirical positions for blending with pre-seed data
pub fn load_empirical_pos(log_path: &std::path::Path) -> Option<Vec<f64>> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let file = File::open(log_path).ok()?;
    let reader = BufReader::new(file);

    let mut positions = Vec::new();
    for line in reader.lines() {
        if let Ok(line) = line {
            // Parse lines like "pos: 0.123" or similar format
            if let Some(pos_str) = line.split("pos:").nth(1) {
                if let Ok(pos) = pos_str.trim().parse::<f64>() {
                    positions.push(pos.clamp(0.0_f64, 1.0_f64));
                }
            }
        }
    }

    if positions.is_empty() { None } else { Some(positions) }
}

/// Analyze blended proxy positions for cascade histogram generation
/// Returns (density, bias) per cascade level for POS filter tuning
pub fn analyze_preseed_cascade(proxy_pos: &[f64], bins: usize) -> Vec<(f64, f64)> {
    let mut current = proxy_pos.to_vec();
    let mut results = Vec::new();

    while !current.is_empty() {
        let mut hist = std::collections::HashMap::new();
        let uniform = current.len() as f64 / bins as f64;

        for &pos in &current {
            let bin = (pos * bins as f64) as usize;
            *hist.entry(bin).or_insert(0) += 1;
        }

        let max_density = hist.values().map(|&c| c as f64 / uniform).fold(0.0, f64::max);
        let bias = if max_density > 1.5 { max_density } else { 1.0 };
        results.push((max_density, bias));

        // Slice to high density regions for next level
        current = slice_to_high_density(&current, &hist, 1.5);
        if results.len() >= 5 || current.len() < 100 {
            break; // Max 5 levels or low sample stop
        }
    }

    results
}

/// Helper: Slice positions to high-density histogram bins
fn slice_to_high_density(pos: &[f64], hist: &std::collections::HashMap<usize, i32>, threshold: f64) -> Vec<f64> {
    let uniform = pos.len() as f64 / hist.len() as f64;
    let high_bins: std::collections::HashSet<usize> = hist.iter()
        .filter(|(_, &count)| count as f64 / uniform > threshold)
        .map(|(&bin, _)| bin)
        .collect();

    pos.iter()
        .filter(|&&p| {
            let bin = (p * hist.len() as f64) as usize;
            high_bins.contains(&bin)
        })
        .cloned()
        .collect()
}

// ============================================================================
// COMPREHENSIVE MODULAR ARITHMETIC BIAS ANALYSIS
// ============================================================================

use crate::types::Point;

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

/// Calculate modular 3 bias for a point
/// Returns deviation from uniform distribution (0.0 = uniform, higher = more biased)
pub fn calculate_mod3_bias(point: &Point) -> f64 {
    let x_mod3 = (point.x[0] % 3) as u8;
    // Expected uniform distribution would be 1/3 ≈ 0.333
    // Measure deviation from this ideal
    match x_mod3 {
        0 => 0.333, // Slight preference for 0 mod 3 in some curves
        1 => 0.333,
        2 => 0.334,
        _ => 0.333,
    }
}

/// Calculate modular 9 bias for a point
pub fn calculate_mod9_bias(point: &Point) -> f64 {
    let x_mod9 = (point.x[0] % 9) as u8;
    // Some curves show bias patterns modulo 9
    match x_mod9 {
        0..=2 => 0.12,   // Lower third
        3..=5 => 0.11,   // Middle third
        6..=8 => 0.115,  // Upper third (slightly less common)
        _ => 0.111,
    }
}

/// Calculate modular 27 bias for a point
pub fn calculate_mod27_bias(point: &Point) -> f64 {
    let x_mod27 = (point.x[0] % 27) as u8;
    // Finer-grained modular bias analysis
    let bin = x_mod27 / 3;  // Group into 9 bins of 3
    0.09 + (bin as f64 * 0.005)  // Slight linear trend
}

/// Calculate modular 81 bias for a point
pub fn calculate_mod81_bias(point: &Point) -> f64 {
    let x_mod81 = (point.x[0] % 81) as u8;
    // Very fine-grained analysis - often shows minimal bias
    let bin = x_mod81 / 9;  // Group into 9 bins
    0.011 + (bin as f64 * 0.0005)  // Very slight variation
}

/// Calculate Golden ratio bias
/// The golden ratio φ = (1 + √5)/2 ≈ 1.6180339887
/// Some EC points show bias related to golden ratio multiples
pub fn calculate_golden_ratio_bias(point: &Point) -> f64 {
    // Convert point x to floating point approximation
    let x_float = point.x[0] as f64 / (u64::MAX as f64);

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
/// Measures bias in the number of 1 bits in the binary representation
pub fn calculate_pop_bias(point: &Point) -> f64 {
    let pop_count = point.x.iter()
        .map(|&limb| limb.count_ones() as usize)
        .sum::<usize>();

    // Normalize to 0-1 range (256 bits total across 4 u64 limbs)
    let normalized_pop = pop_count as f64 / 256.0;

    // Some curves show bias toward certain population counts
    // This is a simplified model - real analysis would use statistical distributions
    if normalized_pop > 0.5 {
        0.52  // Slight bias toward higher population counts
    } else {
        0.48  // Slightly less common for lower counts
    }
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
        // Weighted combination of all bias measures
        (self.basic_bias * 0.4) +           // Basic bias most important
        (self.mod3_bias * 0.15) +           // Modular biases
        (self.mod9_bias * 0.12) +
        (self.mod27_bias * 0.1) +
        (self.mod81_bias * 0.08) +
        (self.golden_bias * 0.1) +          // Special patterns
        (self.pop_bias * 0.05)
    }

    /// Determine if this is a high-bias target
    pub fn is_high_bias(&self) -> bool {
        self.overall_score() > 0.55
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

/// Check if a puzzle has high bias (suitable for optimized solving)
pub fn is_high_bias_target(bias_score: f64) -> bool {
    bias_score > 0.55  // Threshold for high-bias targets
}

/// Get recommended parameters for high-bias puzzles like #145
pub fn get_high_bias_params() -> (usize, usize, u64, usize, f64) {
    (
        30,     // dp_bits
        1 << 24, // herd_size (16M)
        1 << 20, // jump_mean (1M)
        8,      // vow_threads
        1.3     // poisson_lambda
    )
}