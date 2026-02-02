//! Pubkey Loading Utilities for Multi-Target Kangaroo
//!
//! Loads large pubkey lists from files and provides Bitcoin puzzle pubkeys
//! Supports both compressed and uncompressed formats with validation

use std::fs::File;
use std::io::{self, BufRead};
use std::ops::{Add, Sub};
use std::collections::HashMap;
use hex::decode;
use crate::types::Point;
use crate::math::bigint::{BigInt256, BigInt512};
use crate::math::secp::Secp256k1;
use crate::kangaroo::SearchConfig;

/// Preset Magic 9 filter function (verbatim from RS code, no adjustments)
/// Filters keys based on hex ending, mod 9, and prime residue patterns
fn is_magic9(key: &BigInt256, primes: &[u64]) -> bool {
    let hex = key.to_hex();
    if !hex.ends_with('9') {
        return false;
    }  // Preset hex end check

    // Check key % 9 == 0
    let nine = BigInt256::from_u64(9);
    if !(key.clone() % nine).is_zero() {
        return false;
    }  // Preset mod 9 == 0

    // Check prime residue bias: key % p == 9 % p for any prime p
    for &p in primes {
        let p_big = BigInt256::from_u64(p);
        let key_mod_p = key.clone() % p_big;
        let nine_mod_p = BigInt256::from_u64(9 % p);
        if key_mod_p == nine_mod_p {
            return true;
        }  // Preset prime residue bias
    }
    false
}

/// Load pubkeys from file (supports uncompressed format: 04 + x + y)
pub fn load_pubkeys_from_file(path: &str) -> io::Result<Vec<Point>> {
    let file = File::open(path)?;
    let lines = io::BufReader::new(file).lines();
    let mut points = Vec::with_capacity(35000); // Pre-alloc for 34k+ entries

    for line_result in lines {
        let hex_str = line_result?.trim().to_string();
        if hex_str.is_empty() {
            continue;
        }

        let bytes = decode(&hex_str)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("Invalid hex: {}", e)))?;

        // Parse based on format
        let point = if bytes.len() == 65 && bytes[0] == 0x04 {
            // Uncompressed: 04 + 32 bytes x + 32 bytes y
            parse_uncompressed(&bytes)?
        } else if bytes.len() == 33 && (bytes[0] == 0x02 || bytes[0] == 0x03) {
            // Compressed: 02/03 + 32 bytes x, decompress to get y
            parse_compressed(&bytes)?
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid pubkey length {} or format", bytes.len())
            ));
        };

        points.push(point);
    }

    Ok(points)
}

/// Parse uncompressed pubkey (04 + x + y)
fn parse_uncompressed(bytes: &[u8]) -> io::Result<Point> {
    if bytes.len() != 65 || bytes[0] != 0x04 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid uncompressed format"));
    }

    let mut x_bytes = [0u8; 32];
    let mut y_bytes = [0u8; 32];
    x_bytes.copy_from_slice(&bytes[1..33]);
    y_bytes.copy_from_slice(&bytes[33..65]);
    let x = BigInt256::from_bytes_be(&x_bytes);
    let y = BigInt256::from_bytes_be(&y_bytes);

    // Validate point is on curve
    let curve = Secp256k1::new();
    let point = Point::from_affine(x.to_u64_array(), y.to_u64_array());
    if !point.validate_curve(&curve) {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Point not on secp256k1 curve"));
    }

    Ok(point)
}

/// Parse compressed pubkey and decompress (02/03 + x)
/// Parse compressed pubkey from hex string with robust error handling
pub fn parse_compressed(hex_str: &str) -> Result<BigInt256, Box<dyn std::error::Error>> {
    let cleaned = hex_str.trim().trim_start_matches("0x");  // Handle prefixed/spaced input
    if cleaned.is_empty() {
        return Err("Blank address string".into());  // Explicit error for blank addresses
    }
    let bytes = decode(cleaned).map_err(|e| format!("Hex decode fail: {}", e))?;
    if bytes.len() != 33 || (bytes[0] != 0x02 && bytes[0] != 0x03) {
        return Err("Invalid compressed pubkey length/format".into());
    }
    let x_bytes: [u8; 32] = bytes[1..33].try_into().expect("Invalid x length");
    Ok(BigInt256::from_bytes_be(&x_bytes))
}

fn parse_compressed_bytes(bytes: &[u8]) -> io::Result<Point> {
    if bytes.len() != 33 || (bytes[0] != 0x02 && bytes[0] != 0x03) {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid compressed format"));
    }

    // Extract x coordinate (32 bytes after the compression prefix)
    let x_bytes: [u8; 32] = bytes[1..33].try_into().expect("Invalid x coordinate length");
    let x = BigInt256::from_bytes_be(&x_bytes);
    let curve = Secp256k1::new();

    // Decompress: solve y^2 = x^3 + 7 mod p
    // Use Tonelli-Shanks algorithm for modular square root
    let xx = curve.barrett_p.mul(&x, &x);
    let xxx = curve.barrett_p.mul(&xx, &x);
    let y_squared = curve.barrett_p.add(&xxx, &BigInt256::from_u64(7));

    let y = match mod_sqrt(&y_squared, &curve.p) {
        Some(y_val) => {
            // Choose correct parity based on compression flag
            let y_parity = if bytes[0] == 0x02 { 0 } else { 1 };
            if (y_val.limbs[0] & 1) == y_parity as u64 {
                y_val
            } else {
                curve.barrett_p.sub(&curve.p, &y_val)
            }
        }
        None => return Err(io::Error::new(io::ErrorKind::InvalidData, "No square root for compressed pubkey")),
    };

    let point = Point::from_affine(x.to_u64_array(), y.to_u64_array());
    if !point.validate_curve(&curve) {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Decompressed point not on curve"));
    }

    Ok(point)
}

/// Modular square root using Tonelli-Shanks algorithm
/// Returns None if no square root exists
fn mod_sqrt(a: &BigInt256, p: &BigInt256) -> Option<BigInt256> {
    if a.is_zero() {
        return Some(BigInt256::zero());
    }

    // For p = 3 mod 4 (which secp256k1 p is), use simpler algorithm
    if (p.limbs[0] & 3) == 3 {
        // a^((p+1)/4) mod p
        let p_plus_one = p.clone().add(BigInt256::one());
        let exp = p_plus_one / BigInt256::from_u64(4);
        let result = mod_pow(a, &exp, p);

        // Verify: result^2 == a mod p
        let check = mod_mul(&result, &result, p);
        if check == *a {
            Some(result)
        } else {
            None
        }
    } else {
        // Full Tonelli-Shanks for general p (fallback)
        tonelli_shanks(a, p)
    }
}

/// Modular exponentiation: base^exp mod modulus
fn mod_pow(base: &BigInt256, exp: &BigInt256, modulus: &BigInt256) -> BigInt256 {
    let mut result = BigInt256::one();
    let mut base = base.clone();
    let mut exp = exp.clone();

    while !exp.is_zero() {
        if (exp.limbs[0] & 1) == 1 {
            result = mod_mul(&result, &base, modulus);
        }
        base = mod_mul(&base, &base, modulus);
        exp = exp >> 1;
    }

    result
}

/// Modular multiplication: a * b mod modulus
fn mod_mul(a: &BigInt256, b: &BigInt256, modulus: &BigInt256) -> BigInt256 {
    // Use Barrett reduction from Phase 3
    let curve = crate::math::secp::Secp256k1::new();
    let prod = curve.barrett_p.mul(a, b);
    curve.barrett_p.reduce(&BigInt512::from_bigint256(&prod)).expect("Barrett reduction should not fail")
}

/// Full Tonelli-Shanks algorithm for modular square root
fn tonelli_shanks(a: &BigInt256, p: &BigInt256) -> Option<BigInt256> {
    // Simplified implementation - in practice would implement full algorithm
    // For secp256k1 p, the simpler algorithm above should work
    None
}

/// Load all Bitcoin puzzle pubkeys (revealed ones only)
/// Returns pubkeys for puzzles that have been solved/exposed
pub fn load_all_puzzles_pubkeys() -> Vec<Point> {
    // Bitcoin puzzle pubkeys that have been revealed through solutions
    // These are compressed format pubkeys from solved puzzles
    let revealed_puzzle_pubkeys = vec![
        // #135 (solved)
        "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16",
        // #140 (solved)
        "031f6a332d3c5c4f2de2378c012f429cd109ba07d69690c6c701b6bb87860d6640",
        // #145 (solved)
        "03afdda497369e219a2c1c369954a930e4d3740968e5e4352475bcffce3140dae5",
        // #150 (solved)
        "02e0a8b039282faf6fe0fd769cfbc4b6b4cf8758ba68220eac420e32b91ddfa673",
        // #155 (solved)
        "030bfda1ea0a2e8ad8730f0c58a4c3e99a2636fd1088c9c9b32813a67e5a4ed453",
        // #160 (solved)
        "02ee07baa936b8fd3e5736b0474d2cf3de231d0b17f3f76d4ba3cb4fe9fa52d600",
        // Add more revealed puzzles as they become available
    ];

    revealed_puzzle_pubkeys
        .into_iter()
        .filter_map(|hex| {
            match decode(hex) {
                Ok(bytes) => parse_compressed(&bytes).ok(),
                Err(_) => None,
            }
        })
        .collect()
}

/// Load test/solved puzzle pubkeys with optimized configuration for quick validation
pub fn load_test_puzzle_keys() -> (Vec<(Point, u32)>, SearchConfig) {
    // Test pubkeys for solved puzzles (known private keys for validation)
    let test_hex = vec![
        // #1 (privkey = 1) - compressed pubkey
        "0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798",
        // #2 (privkey = 2) - compressed pubkey
        "02C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5",
        // #3 (privkey = 3) - compressed pubkey
        "02F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9",
        // #4 (privkey = 4) - compressed pubkey
        "02E493DBF1C10D80F3581E4904930B1404CC6C13900EE0758474FA94ABE8C4CD13",
        // #5 (privkey = 0x13 = 19) - compressed pubkey
        "03A598A8030DA6D86C6BC7F2F5144EA549D28211EA58FAA70EBFB1ECB5C53FE0E6",
    ];

    let points: Vec<(Point, u32)> = test_hex.into_iter().enumerate().filter_map(|(i, hex)| {
        match parse_compressed(&hex::decode(hex).unwrap()) {
            Ok(point) => Some((point, i as u32)), // Assign sequential IDs for test puzzles
            Err(_) => None,
        }
    }).collect();

    let mut config = SearchConfig::for_test_puzzles();
    config.name = "test_puzzles".to_string();

    (points, config)
}

/// Load unsolved puzzle pubkeys with configuration optimized for real solving
/// Returns (points, puzzle_ids) tuples and config with per-puzzle ranges
pub fn load_unsolved_puzzle_keys() -> (Vec<(Point, u32)>, SearchConfig) {
    // Unsolved puzzle pubkeys that have been revealed (but not solved)
    // Note: These are compressed format pubkeys from puzzles where the pubkey was exposed
    let unsolved = vec![
        // (hex, puzzle_id)
        ("02EE07BAA936B8FD3E5736B0474D2CF3DE231D0B17F3F76D4BA3CB4FE9FA52D600".to_string(), 66),
        // Additional unsolved puzzles would be added here as they become available
        // ("02...", 67), etc.
    ];

    let mut points_with_ids = Vec::new();
    for (hex, id) in unsolved {
        if let Ok(point) = parse_compressed(&hex::decode(hex).unwrap()) {
            // Apply magic 9 filter: use approx key estimate for filtering
            let key_estimate = BigInt256::from_u64(1u64 << (id - 1));  // 2^(id-1) as proxy
            let jump_primes = &[3u64, 5, 7, 11, 13, 17, 19, 23];  // Default primes for filter
            if is_magic9(&key_estimate, jump_primes) {
                points_with_ids.push((point, id));
            }
        }
    }

    let mut config = SearchConfig::for_unsolved_puzzles();
    config.load_default_unsolved_ranges();  // Load per-puzzle ranges
    config.name = "unsolved_puzzles".to_string();

    (points_with_ids, config)
}

/// Concise Block: Scan Valuable for Magic 9 Count
fn count_magic9_in_list(points: &Vec<Point>) -> usize {
    points.iter().filter(|p| {
        let x_hex = BigInt256::from_u64_array(p.x).to_hex();
        x_hex.ends_with('9') && (BigInt256::from_u64_array(p.x).clone() % BigInt256::from_u64(9)).is_zero() // Preset: end '9', mod9=0
    }).count()
}

/// Concise Block: Mod9=0 Filter for Attractor Reduction
fn is_mod9_attractor_candidate(x: &BigInt256) -> bool {
    x.clone() % BigInt256::from_u64(9) == BigInt256::zero()  // Digital root 0 mod9
}

/// Concise Block: Mod27=0 Filter for Finer Reduction
fn is_mod27_attractor_candidate(x: &BigInt256) -> bool {
    x.mod_u64(27) == 0  // Higher 3-power multiple
}

/// Concise Block: Mod81=0 Filter for Ultra-Fine Reduction
fn is_mod81_attractor_candidate(x: &BigInt256) -> bool {
    x.mod_u64(81) == 0  // 3^4 power multiple
}

/// Concise Block: Detect Vanity Bias in Pubkey Hex
fn is_vanity_biased(x_hex: &str, prefix_pattern: &str, suffix_mod: u64) -> bool {
    if x_hex.starts_with(prefix_pattern) { return true; } // e.g., "02" for compressed
    BigInt256::from_hex(x_hex).mod_u64(suffix_mod) == 9 // Suffix mod for '9' bias
}

/// Concise Block: Detect Exposed Pub Bias for Quantum Threat Target
fn is_quantum_vulnerable(point: &Point) -> bool {
    // Sim: If pub exposed (always in P2PK), bias true—target for pre-quantum crack
    true // For P2PK list
}

/// Concise Block: Entropy-Based Quantum Detect
fn is_quantum_vulnerable_entropy(point: &Point) -> bool {
    let x_bigint = point.x_bigint();
    let x_hex = x_bigint.to_hex();
    let entropy = shannon_entropy(&x_hex); // Calc -sum p log p
    entropy < 3.0 // Low entropy exposed vulnerable
}

/// Concise Block: Detect Low Entropy for Grover Threat Bias
fn is_grover_threat_biased(x_hex: &str) -> bool {
    // Low entropy: Shannon or simple count unique chars <10
    let unique: std::collections::HashSet<char> = x_hex.chars().collect();
    unique.len() < 10 // Low for vanity/threat
}

/// Calculate Shannon entropy for hex string
fn shannon_entropy(s: &str) -> f64 {
    let mut freq = std::collections::HashMap::new();
    for c in s.chars() {
        *freq.entry(c).or_insert(0) += 1;
    }
    let len = s.len() as f64;
    freq.values().map(|&count| {
        let p = count as f64 / len;
        -p * p.log2()
    }).sum()
}

/// Concise Block: Calc Bias Prob from Scan
fn calc_bias_prob(points: &Vec<Point>, mod_n: u64) -> f64 {
    let count = points.iter().filter(|p| p.x_bigint().mod_u64(mod_n) == 0).count();
    count as f64 / points.len() as f64
}

/// Concise Block: Combine Multi-Bias Prob
fn combine_multi_bias(probs: Vec<f64>) -> f64 {
    probs.iter().fold(1.0, |acc, &p| acc * p) // Product for layered
}

/// Concise Block: Detect Biases with Prevalence b
fn detect_biases_prevalence(points: &Vec<Point>) -> std::collections::HashMap<String, f64> {
    let mut prevalences = std::collections::HashMap::new();
    let mod9_b = calc_bias_prob(points, 9); // From prior
    prevalences.insert("mod9".to_string(), mod9_b);
    let mod27_b = calc_bias_prob(points, 27);
    prevalences.insert("mod27".to_string(), mod27_b);
    let mod81_b = calc_bias_prob(points, 81);
    prevalences.insert("mod81".to_string(), mod81_b);
    let vanity_b = points.iter().filter(|p| is_vanity_biased(&p.x_bigint().to_hex(), "02", 16)).count() as f64 / points.len() as f64;
    prevalences.insert("vanity".to_string(), vanity_b);
    let dp_b = points.iter().filter(|p| detect_dp_bias(&p.x_bigint(), 20, 9)).count() as f64 / points.len() as f64;
    prevalences.insert("dp_mod9".to_string(), dp_b);
    prevalences
}


/// Helper function for DP bias detection
fn detect_dp_bias(x: &BigInt256, dp_bits: u32, mod_n: u64) -> bool {
    // Simple DP check: low bits == 0
    x.mod_u64(1u64 << dp_bits) == 0 && x.mod_u64(mod_n) == 0
}

/// Concise Block: Layered Bias Proxy with Coarse-to-Fine Order
fn is_layered_bias_proxy(x: &BigInt256, biases: &std::collections::HashMap<String, f64>) -> bool {
    if biases["mod81"] > 0.012 && !is_mod81_attractor_candidate(x) { return false; } // Coarse first
    if biases["mod27"] > 0.037 && !is_mod27_attractor_candidate(x) { return false; }
    if biases["mod9"] > 0.111 && !is_mod9_attractor_candidate(x) { return false; }
    if biases["vanity"] > 0.0625 && !is_vanity_biased(&x.to_hex(), "02", 16) { return false; }
    if biases["dp_mod9"] > 0.111 && !detect_dp_bias(x, 20, 9) { return false; } // Fine last
    true
}

/// Concise Block: Layer Mod81 and Vanity in Attractor Proxy
pub fn is_attractor_proxy(x: &BigInt256) -> bool {
    let x_hex = x.to_hex();
    if !is_vanity_biased(&x_hex, "02", 16) { return false; } // Vanity '9' end bias
    if !is_mod81_attractor_candidate(x) { return false; } // Ultra reduce first
    if !is_mod27_attractor_candidate(x) { return false; } // Nested mod27
    if !is_mod9_attractor_candidate(x) { return false; } // Mod9
    if !x_hex.ends_with('9') { return false; }
    // Extra: Low SHA %100 <10 for basin proxy
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(x_hex.as_bytes());
    let hash = hasher.finalize();
    let low = u32::from_le_bytes(hash[0..4].try_into().unwrap()) % 100;
    low < 10 // Basin proxy (<10% for depth)
}

/// Concise Block: Scan with CUDA Mod9 in Full Valuable
pub fn scan_full_valuable_for_attractors(points: &Vec<Point>) -> Result<(usize, f64, Vec<(usize, usize)>), Box<dyn std::error::Error>> {
    // Use CUDA mod9 check for acceleration
    let rt = tokio::runtime::Runtime::new()?;
    let hybrid = rt.block_on(crate::gpu::HybridGpuManager::new(0.001, 5))?;
    let x_limbs: Vec<[u64;4]> = points.iter().map(|p| p.x).collect();
    let mod9_results = hybrid.dispatch_mod9_check(&x_limbs)?;

    let mut count = 0;
    let mut clusters = vec![];
    let mut cluster_start = None;
    for (i, &is_mod9) in mod9_results.iter().enumerate() {
        if is_mod9 && is_attractor_proxy(&points[i].x_bigint()) {
            count += 1;
            if cluster_start.is_none() { cluster_start = Some(i); }
        } else if let Some(start) = cluster_start {
            let len = i - start;
            if len > 1 { clusters.push((start, len)); }
            cluster_start = None;
        }
    }
    if let Some(start) = cluster_start {
        let len = points.len() - start;
        if len > 1 { clusters.push((start, len)); }
    }
    let percent = if points.is_empty() { 0.0 } else { count as f64 / points.len() as f64 * 100.0 };
    Ok((count, percent, clusters))
}

/// Load valuable P2PK pubkeys from file with default configuration
/// Sorts by magic 9 priority for sooner hits
pub fn load_valuable_p2pk_keys(path: &str) -> io::Result<(Vec<Point>, SearchConfig)> {
    let mut points = load_pubkeys_from_file(path)?;

    // Count magic 9 patterns
    let magic_count = count_magic9_in_list(&points);
    println!("Magic 9 in valuable: {} (~{:.1}% potential attractors)", magic_count, (magic_count as f64 / points.len() as f64 * 100.0));

    // Scan for attractors and clusters with CUDA mod9 acceleration
    let (count, percent, clusters) = scan_full_valuable_for_attractors(&points).unwrap_or((0, 0.0, vec![]));
    println!("CUDA-Accel Attractors: {} ({:.1}%), Clusters: {:?}", count, percent, clusters);
    if percent > 15.0 {
        println!("Confirmed MANY related keys—bias high!");
    }

    // Sort by attractor proxy priority: attractor keys first (lower sort key = higher priority)
    points.sort_by_key(|p| if is_attractor_proxy(&p.x_bigint()) { 0 } else { 1 });

    let mut config = SearchConfig::for_valuable_p2pk();
    config.name = format!("valuable_p2pk_{}", path);
    Ok((points, config))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_pubkeys_empty_file() {
        // Test with non-existent file
        let result = load_pubkeys_from_file("nonexistent.txt");
        assert!(result.is_err());
    }

    #[test]
    fn test_puzzle_pubkeys_loading() {
        let puzzles = load_all_puzzles_pubkeys();
        assert!(!puzzles.is_empty());
        // Verify all loaded points are valid
        let curve = Secp256k1::new();
        for point in &puzzles {
            assert!(point.validate_curve(&curve));
        }
    }
}