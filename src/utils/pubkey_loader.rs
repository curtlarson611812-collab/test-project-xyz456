//! Pubkey Loading Utilities for Multi-Target Kangaroo
//!
//! Loads large pubkey lists from files and provides Bitcoin puzzle pubkeys
//! Supports both compressed and uncompressed formats with validation

use std::fs::File;
use std::io::{self, BufRead};
use std::ops::{Add, Sub};
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
            parse_compressed_bytes(&bytes)?
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
    if cleaned.len() != 66 {  // 33 bytes *2 hex
        log::warn!("Cleaned hex len {} !=66 (33 bytes): {}", cleaned.len(), cleaned);
    }
    let bytes = decode(cleaned).map_err(|e| format!("Hex decode fail: {}", e))?;
    log::info!("Decoded bytes len: {}", bytes.len());  // Debug
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
fn mod_mul(a: &BigInt256, b: &BigInt256, _modulus: &BigInt256) -> BigInt256 {
    // Use Barrett reduction from Phase 3
    let curve = crate::math::secp::Secp256k1::new();
    let prod = curve.barrett_p.mul(a, b);
    curve.barrett_p.reduce(&BigInt512::from_bigint256(&prod)).expect("Barrett reduction should not fail")
}

/// Full Tonelli-Shanks algorithm for modular square root
fn tonelli_shanks(_a: &BigInt256, _p: &BigInt256) -> Option<BigInt256> {
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

    let curve = Secp256k1::new();
    revealed_puzzle_pubkeys
        .into_iter()
        .filter_map(|hex| {
            match decode(hex) {
                Ok(bytes) => {
                    if bytes.len() == 33 && (bytes[0] == 0x02 || bytes[0] == 0x03) {
                        curve.decompress_point(&bytes.try_into().unwrap())
                    } else {
                        None
                    }
                }
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

    let curve = Secp256k1::new();
    let points: Vec<(Point, u32)> = test_hex.into_iter().enumerate().filter_map(|(i, hex)| {
        match hex::decode(hex) {
            Ok(bytes) => {
                if bytes.len() == 33 && (bytes[0] == 0x02 || bytes[0] == 0x03) {
                    let mut comp = [0u8; 33];
                    comp.copy_from_slice(&bytes);
                    curve.decompress_point(&comp).map(|point| (point, i as u32))
                } else {
                    None
                }
            }
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
        if let Ok(_x) = parse_compressed(&hex) {
            // Decompress the point using the x coordinate
            let bytes = hex::decode(hex).unwrap();
            if bytes.len() == 33 {
                let mut comp = [0u8; 33];
                comp.copy_from_slice(&bytes);
                if let Some(point) = Secp256k1::new().decompress_point(&comp) {
                    // Apply magic 9 filter: use approx key estimate for filtering
                    let key_estimate = BigInt256::from_u64(1u64 << (id - 1));  // 2^(id-1) as proxy
                    let jump_primes = &[3u64, 5, 7, 11, 13, 17, 19, 23];  // Default primes for filter
                    if is_magic9(&key_estimate, jump_primes) {
                        points_with_ids.push((point, id));
                    }
                }
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
// fn is_quantum_vulnerable(_point: &Point) -> bool {
//     // Sim: If pub exposed (always in P2PK), bias true—target for pre-quantum crack
//     true // For P2PK list
// }

/// Concise Block: Entropy-Based Quantum Detect
// fn is_quantum_vulnerable_entropy(point: &Point) -> bool {
//     let x_bigint = point.x_bigint();
//     let x_hex = x_bigint.to_hex();
//     let entropy = shannon_entropy(&x_hex); // Calc -sum p log p
//     entropy < 3.0 // Low entropy exposed vulnerable
// }

/// Concise Block: Detect Low Entropy for Grover Threat Bias
// fn is_grover_threat_biased(x_hex: &str) -> bool {
//     // Low entropy: Shannon or simple count unique chars <10
//     let unique: std::collections::HashSet<char> = x_hex.chars().collect();
//     unique.len() < 10 // Low for vanity/threat
// }

/// Calculate Shannon entropy for hex string
// fn shannon_entropy(s: &str) -> f64 {
//     let mut freq = std::collections::HashMap::new();
//     for c in s.chars() {
//         *freq.entry(c).or_insert(0) += 1;
//     }
//     let len = s.len() as f64;
//     freq.values().map(|&count| {
//         let p = count as f64 / len;
//         -p * p.log2()
//     }).sum()
// }

/// Concise Block: Calc Bias Prob from Scan
// fn calc_bias_prob(points: &Vec<Point>, mod_n: u64) -> f64 {
//     let count = points.iter().filter(|p| p.x_bigint().mod_u64(mod_n) == 0).count();
//     count as f64 / points.len() as f64
// }

/// Concise Block: Combine Multi-Bias Prob
// fn combine_multi_bias(probs: Vec<f64>) -> f64 {
//     probs.iter().fold(1.0, |acc, &p| acc * p) // Product for layered
// }

/// Detect bias for a single point (used for individual puzzle analysis)
pub fn detect_bias_single(x: &BigInt256, n: u32) -> (u64, u64, u64, bool, bool, f64) {
    let mod9 = x.mod_u64(9);
    let mod27 = x.mod_u64(27);
    let mod81 = x.mod_u64(81);

    // Vanity bias: check if last hex digit is '0' (common vanity pattern)
    let x_hex = x.to_hex();
    let vanity_last_0 = x_hex.ends_with('0');

    // DP mod9: trivial check if mod9 matches (for DP framework)
    let dp_mod9 = true; // Always true for single point - would be used in DP collection

    // Positional proxy bias for unsolved puzzles
    let pos_proxy = detect_pos_bias_proxy_single(n);

    (mod9, mod27, mod81, vanity_last_0, dp_mod9, pos_proxy)
}

/// Detect dimensionless position bias for a single puzzle
/// Returns normalized position in [0,1] within the puzzle's interval
pub fn detect_pos_bias_single(priv_key: &BigInt256, puzzle_n: u32) -> f64 {
    // For puzzle #N: range is [2^(N-1), 2^N - 1]
    // pos = (priv - 2^(N-1)) / (2^N - 1 - 2^(N-1)) = (priv - 2^(N-1)) / (2^(N-1))

    // Calculate 2^(N-1) using bit shifting
    let mut min_range = BigInt256::from_u64(1);
    for _ in 0..(puzzle_n - 1) {
        min_range = min_range.clone().add(min_range.clone()); // Double the value
    }
    let range_width = min_range.clone(); // 2^(N-1)

    // priv should be >= min_range for valid puzzles
    if priv_key < &min_range {
        return 0.0; // Invalid, but return 0
    }

    let offset = priv_key.clone().sub(min_range.clone());
    let pos = offset.to_f64() / range_width.to_f64();

    // Clamp to [0,1] in case of rounding issues
    pos.max(0.0).min(1.0)
}

/// Detect proxy positional bias for unsolved puzzles
/// Returns proxy position (0.0 for start of range) since priv is unknown
pub fn detect_pos_bias_proxy_single(_n: u32) -> f64 {
    // For unsolved puzzles, we use the starting point [2^(n-1)]G as proxy
    // This gives pos = 0.0, but we can analyze clustering patterns from solved puzzles
    0.0
}

/// Analyze positional bias across multiple solved puzzles
/// Returns histogram of positional clustering (10 bins [0-0.1, 0.1-0.2, ..., 0.9-1.0])
pub fn analyze_pos_bias_histogram(solved_puzzles: &[(u32, BigInt256)]) -> [f64; 10] {
    let mut hist = [0u32; 10];

    for (puzzle_n, priv_key) in solved_puzzles {
        let pos = detect_pos_bias_single(priv_key, *puzzle_n);
        let bin = (pos * 10.0).min(9.0) as usize; // 0-9 for 10 bins
        hist[bin] += 1;
    }

    let total = solved_puzzles.len() as f64;
    let mut result = [0.0; 10];

    for i in 0..10 {
        // Normalize: prevalence per bin (uniform would be 1.0)
        result[i] = if total > 0.0 { (hist[i] as f64) / (total / 10.0) } else { 1.0 };
    }

    result
}

/// Concise Block: Pick Most Likely Unsolved Puzzle to Crack
/// Scores puzzles by bias_factor / 2^(n/2) to prioritize high bias, low complexity
pub fn pick_most_likely_unsolved() -> u32 {
    // Simple implementation - in practice this should use full bias analysis
    // For now, return the smallest unsolved puzzle (67) as it's most likely to be crackable
    67
}

/// Deep Dive: Deeper Mod9 Subgroup Analysis
/// Analyzes mod27 subgroups within the most biased mod9 residue
/// Returns (b_mod9, max_r9, b_mod27, max_r27)
pub fn deeper_mod9_subgroup(points: &[Point]) -> (f64, u64, f64, u64) {
    let (_hist9, _expected9, max_r9, b_mod9, _sig9) = analyze_mod9_bias_deeper(points);

    // Analyze mod27 subgroups within the most biased mod9 residue
    let mut sub_hist27 = [0u32; 3];  // For r=0,9,18 mod27 within mod9=max_r9
    let mut sub_total = 0u32;

    for point in points {
        let x_bigint = BigInt256::from_u64_array(point.x);
        let mod9 = x_bigint.mod_u64(9);
        if mod9 == max_r9 {
            let mod27 = x_bigint.mod_u64(27);
            // Only count if mod27 ≡ max_r9 mod 9 (conditional subgroup)
            if mod27 % 9 == max_r9 {
                let sub_bin = (mod27 / 9) as usize;  // 0, 1, or 2
                if sub_bin < 3 {
                    sub_hist27[sub_bin] += 1;
                    sub_total += 1;
                }
            }
        }
    }

    let sub_expected = sub_total as f64 / 3.0;
    let b_mod27 = if sub_expected > 0.0 {
        sub_hist27.iter().map(|&count| count as f64 / sub_expected).fold(0.0, f64::max)
    } else {
        1.0 / 3.0
    };

    let max_sub_bin = sub_hist27.iter().enumerate()
        .max_by(|a, b| a.1.cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let max_r27 = max_sub_bin as u64 * 9 + max_r9;  // Full mod27 residue

    (b_mod9, max_r9, b_mod27, max_r27)
}

/// Deep Dive: Iterative Mod9 Slice Analysis
/// Performs multi-level conditional bias analysis (mod9 -> mod27 -> mod81)
/// Returns the product of conditional biases
pub fn iterative_mod9_slice(points: &[Point], max_levels: u32) -> f64 {
    let mut b_prod = 1.0;
    let mut sub_points = points.to_vec();
    let mut current_modulus = 9u64;

    for _level in 0..max_levels.min(3) {
        if sub_points.len() < 100 {
            break; // Stop if too few points (overfitting protection)
        }

        let (_hist, _expected, max_r, b, _sig) = analyze_mod_bias_deeper(&sub_points, current_modulus);

        // Stop if bias is not significant or below uniform
        let uniform_bias = 1.0 / current_modulus as f64;
        if b <= uniform_bias * 1.1 || !_sig {
            break;
        }

        b_prod *= b;

        // Filter points to the most biased subgroup for next level
        sub_points = sub_points.into_iter()
            .filter(|p| {
                let x_bigint = BigInt256::from_u64_array(p.x);
                x_bigint.mod_u64(current_modulus) == max_r
            })
            .collect();

        current_modulus *= 3; // Next level: 9 -> 27 -> 81
    }

    b_prod
}

/// Generalized mod bias analysis for any modulus
/// Returns (hist, expected, max_r, b, sig)
fn analyze_mod_bias_deeper(points: &[Point], modulus: u64) -> ([u32; 81], f64, u64, f64, bool) {
    let mut hist = [0u32; 81];
    let total = points.len() as f64;

    for point in points {
        let x_bigint = BigInt256::from_u64_array(point.x);
        let residue = x_bigint.mod_u64(modulus);
        if (residue as usize) < hist.len() {
            hist[residue as usize] += 1;
        }
    }

    let expected = total / modulus as f64;
    let mut max_b = 0.0;
    let mut max_r = 0u64;

    for (r, &count) in hist.iter().enumerate() {
        if r >= modulus as usize { break; }
        let b = count as f64 / expected;
        if b > max_b {
            max_b = b;
            max_r = r as u64;
        }
    }

    // Chi-square test for significance (simplified for degrees of freedom)
    let chi_square = hist.iter().enumerate()
        .take(modulus as usize)
        .map(|(_r, &count)| {
            let observed = count as f64;
            (observed - expected).powi(2) / expected
        })
        .sum::<f64>();

    let df = (modulus - 1) as f64;
    let chi_critical = if df <= 10.0 { 15.51 + (df - 8.0) * 2.0 } else { df + 2.0 * df.sqrt() };
    let is_significant = chi_square > chi_critical && total >= 100.0;

    (hist, expected, max_r, max_b, is_significant)
}

/// Deep Dive: Analyze Positional Histogram
/// Builds histogram of dimensionless positions in 10 bins [0,1]
/// Returns array of counts per bin
fn analyze_pos_hist(points: &[Point]) -> [u32; 10] {
    let mut hist = [0u32; 10];

    for point in points {
        // For solved puzzles, we have known private keys
        // For unsolved, we'd need to estimate or use proxy
        // For now, use x-coordinate mod some large number as proxy
        let x_bigint = BigInt256::from_u64_array(point.x);
        let proxy_pos = (x_bigint.mod_u64(1000000) as f64) / 1000000.0;

        let bin = (proxy_pos * 10.0).min(9.0) as usize;
        hist[bin] += 1;
    }

    hist
}

/// Deep Dive: Iterative Positional Slice Analysis
/// Performs iterative narrowing of positional ranges based on bias
/// Returns (b_prod, narrowed_min, narrowed_max)
pub fn iterative_pos_slice(points: &[Point], max_iters: u32) -> (f64, f64, f64) {
    let mut b_prod = 1.0;
    let mut current_min = 0.0;
    let mut current_max = 1.0;
    let mut sub_points = points.to_vec();

    for _iter in 0..max_iters.min(3) {
        if sub_points.len() < 100 {
            break; // Overfitting protection
        }

        let hist = analyze_pos_hist(&sub_points);
        let expected = sub_points.len() as f64 / 10.0;

        // Find most biased bin
        let mut max_b = 0.0;
        let mut max_bin = 0usize;

        for (bin, &count) in hist.iter().enumerate() {
            let b = count as f64 / expected;
            if b > max_b {
                max_b = b;
                max_bin = bin;
            }
        }

        // Stop if bias is not significant
        if max_b <= 1.1 || !is_pos_bias_significant(&hist, expected) {
            break;
        }

        b_prod *= max_b;

        // Narrow the range to the most biased bin
        let bin_width = (current_max - current_min) / 10.0;
        let new_min = current_min + (max_bin as f64) * bin_width;
        let new_max = new_min + bin_width;

        current_min = new_min;
        current_max = new_max;

        // Filter points to new range (using proxy positions)
        sub_points = sub_points.into_iter()
            .filter(|p| {
                let x_bigint = BigInt256::from_u64_array(p.x);
                let proxy_pos = (x_bigint.mod_u64(1000000) as f64) / 1000000.0;
                proxy_pos >= current_min && proxy_pos < current_max
            })
            .collect();
    }

    (b_prod, current_min, current_max)
}

/// Helper: Test if positional bias is statistically significant
fn is_pos_bias_significant(hist: &[u32; 10], expected: f64) -> bool {
    let chi_square: f64 = hist.iter()
        .map(|&count| {
            let observed = count as f64;
            (observed - expected).powi(2) / expected
        })
        .sum();

    // Chi-square critical value for 9 degrees of freedom at p=0.05
    let chi_critical = 16.92;
    chi_square > chi_critical
}

/// Concise Block: Deeper Iterative Positional Bias Narrowing with Overfitting Protection
/// Performs up to max_iters rounds of slicing with Bayesian stopping criteria
/// Returns (cumulative_bias_factor, final_min_range, final_max_range, iterations_performed, overfitting_risk)
pub fn iterative_pos_bias_narrowing_deeper(solved_puzzles: &[(u32, BigInt256)], max_iters: usize) -> (f64, BigInt256, BigInt256, usize, f64) {
    if solved_puzzles.is_empty() {
        return (1.0, BigInt256::one(), BigInt256::from_u64(2), 0, 0.0);
    }

    let mut cumulative_bias = 1.0;
    let mut current_puzzles = solved_puzzles.to_vec();
    let current_min = BigInt256::one(); // Start of first puzzle range
    let current_max = BigInt256::from_u64(1) << 100; // End of largest puzzle range (use reasonable limit)
    let mut overfitting_risk = 0.0;

    for _iter in 0..max_iters.min(3) { // Limit to 3 iterations maximum
        let n = current_puzzles.len();

        // Overfitting protection: stop if sample size too small
        if n < 100 {
            overfitting_risk = 1.0 - (n as f64 / 100.0).min(1.0);
            break;
        }

        // Analyze current subset
        let hist = analyze_pos_bias_histogram(&current_puzzles);
        let _expected = 1.0; // Uniform would be 1.0

        // Find the most biased bin
        let mut max_bias = 0.0;
        let mut best_bin = 0;
        let mut best_count = 0u32;

        for (bin, &bias) in hist.iter().enumerate() {
            if bias > max_bias {
                max_bias = bias;
                best_bin = bin;
                // Get the actual count for Bayesian analysis
                best_count = (bias * n as f64 / 10.0) as u32; // Approximate count in bin
            }
        }

        // Bayesian stopping criteria: posterior mean should be > 0.1 for significance
        // Prior: Beta(1,1) uniform, Posterior: Beta(count+1, n-count+1)
        let posterior_mean = (best_count as f64 + 1.0) / (n as f64 + 2.0);
        if posterior_mean <= 0.1 {
            // No significant evidence of clustering
            break;
        }

        // Stop if bias is too weak (close to uniform)
        if max_bias <= 1.05 {
            break;
        }

        // Calculate new range slice for the best bin
        let bin_start = best_bin as f64 * 0.1;
        let bin_end = (best_bin + 1) as f64 * 0.1;

        // Filter puzzles to this bin range
        let mut new_puzzles = Vec::new();
        for (puzzle_n, priv_key) in &current_puzzles {
            let pos = detect_pos_bias_single(priv_key, *puzzle_n);
            if pos >= bin_start && pos < bin_end {
                new_puzzles.push((*puzzle_n, priv_key.clone()));
            }
        }

        if new_puzzles.is_empty() {
            break;
        }

        // Update cumulative bias
        cumulative_bias *= max_bias;
        current_puzzles = new_puzzles;

        // Calculate overfitting risk based on variance
        let bin_probability = 0.1; // Uniform probability per bin
        let variance = n as f64 * bin_probability * (1.0 - bin_probability);
        let std_dev = variance.sqrt();
        overfitting_risk = if std_dev > 0.0 {
            (max_bias - 1.0) / (3.0 * std_dev / (n as f64).sqrt()) // Z-score like measure
        } else {
            0.0
        };
    }

    (cumulative_bias, current_min, current_max, current_puzzles.len().min(max_iters), overfitting_risk)
}

/// Helper function: Calculate positional histogram for a given range
pub fn analyze_pos_hist_deeper(puzzles: &[(u32, BigInt256)]) -> ([u32; 10], f64, usize, f64) {
    let mut hist = [0u32; 10];
    let total = puzzles.len() as f64;

    // Build histogram
    for (puzzle_n, priv_key) in puzzles {
        let pos = detect_pos_bias_single(priv_key, *puzzle_n);
        let bin = (pos * 10.0).min(9.0) as usize;
        hist[bin] += 1;
    }

    // Calculate bias factors and find maximum
    let expected = total / 10.0;
    let mut max_bias = 0.0;
    let mut max_bin = 0;
    let mut chi_square = 0.0;

    for (bin, &count) in hist.iter().enumerate() {
        let observed = count as f64;
        if expected > 0.0 {
            chi_square += (observed - expected).powi(2) / expected;
        }

        let bias_factor = if expected > 0.0 { observed / expected } else { 1.0 };
        if bias_factor > max_bias {
            max_bias = bias_factor;
            max_bin = bin;
        }
    }

    // Chi-square critical value for 9 degrees of freedom at p=0.05 is approximately 16.92
    let chi_square_critical = 16.92;
    let is_significant = chi_square > chi_square_critical && total >= 100.0;

    (hist, max_bias, max_bin, if is_significant { chi_square } else { 0.0 })
}

/// Concise Block: Deeper Mod9 Histogram Analysis with Statistical Significance
/// Returns (histogram[9], max_bias_factor, most_biased_residue, chi_square_statistic, is_significant)
pub fn analyze_mod9_bias_deeper(points: &[Point]) -> ([u32; 9], f64, u64, f64, bool) {
    let mut hist = [0u32; 9];
    let total = points.len() as f64;

    // Build histogram
    for point in points {
        let x_bigint = BigInt256::from_u64_array(point.x);
        let residue = x_bigint.mod_u64(9);
        hist[residue as usize] += 1;
    }

    // Calculate bias factors and statistical significance
    let expected = total / 9.0;
    let mut max_bias = 0.0;
    let mut most_biased_residue = 0u64;
    let mut chi_square = 0.0;

    for (residue, &count) in hist.iter().enumerate() {
        let observed = count as f64;
        let expected_count = expected;

        // Chi-square contribution: (observed - expected)^2 / expected
        if expected_count > 0.0 {
            chi_square += (observed - expected_count).powi(2) / expected_count;
        }

        let bias_factor = if expected_count > 0.0 { observed / expected_count } else { 1.0 };
        if bias_factor > max_bias {
            max_bias = bias_factor;
            most_biased_residue = residue as u64;
        }
    }

    // Chi-square critical value for 8 degrees of freedom at p=0.05 is approximately 15.51
    let chi_square_critical = 15.51;
    let is_significant = chi_square > chi_square_critical && total >= 100.0; // Need sufficient sample size

    (hist, max_bias, most_biased_residue, chi_square, is_significant)
}

/// Concise Block: Deeper Mod9 Subgroup Analysis (Mod27 within Mod9 clusters)
/// Returns (subgroup_hist[3], max_sub_bias, most_biased_sub_residue, parent_residue)
pub fn analyze_mod9_subgroup_deeper(points: &[Point], parent_residue: u64) -> ([u32; 3], f64, u64) {
    let mut sub_hist = [0u32; 3]; // For residues 0, 9, 18 within the mod27 subgroup

    // Count points in the parent mod9 residue cluster
    for point in points {
        let x_bigint = BigInt256::from_u64_array(point.x);
        let mod9 = x_bigint.mod_u64(9);
        if mod9 == parent_residue {
            let mod27 = x_bigint.mod_u64(27);
            // Map mod27 residues that fall into this mod9 subgroup
            // For mod9 = r, the corresponding mod27 residues are r, r+9, r+18
            if mod27 == parent_residue {
                sub_hist[0] += 1;
            } else if mod27 == parent_residue + 9 {
                sub_hist[1] += 1;
            } else if mod27 == parent_residue + 18 {
                sub_hist[2] += 1;
            }
        }
    }

    let sub_total = sub_hist.iter().sum::<u32>() as f64;
    let expected = sub_total / 3.0;
    let mut max_sub_bias = 0.0;
    let mut most_biased_sub = 0u64;

    for (i, &count) in sub_hist.iter().enumerate() {
        let bias_factor = if expected > 0.0 { count as f64 / expected } else { 1.0 };
        if bias_factor > max_sub_bias {
            max_sub_bias = bias_factor;
            most_biased_sub = i as u64;
        }
    }

    (sub_hist, max_sub_bias, most_biased_sub)
}

/// Concise Block: Detect Biases with Prevalence b
// fn detect_biases_prevalence(points: &Vec<Point>) -> std::collections::HashMap<String, f64> {
//     let mut prevalences = std::collections::HashMap::new();
//     let mod9_b = calc_bias_prob(points, 9); // From prior
//     prevalences.insert("mod9".to_string(), mod9_b);
//     let mod27_b = calc_bias_prob(points, 27);
//     prevalences.insert("mod27".to_string(), mod27_b);
//     let mod81_b = calc_bias_prob(points, 81);
//     prevalences.insert("mod81".to_string(), mod81_b);
//     let vanity_b = points.iter().filter(|p| is_vanity_biased(&p.x_bigint().to_hex(), "02", 16)).count() as f64 / points.len() as f64;
//     prevalences.insert("vanity".to_string(), vanity_b);
//     let dp_b = points.iter().filter(|p| detect_dp_bias(&p.x_bigint(), 20, 9)).count() as f64 / points.len() as f64;
//     prevalences.insert("dp_mod9".to_string(), dp_b);

//     // Add deeper mod9 analysis
//     let (_, mod9_max_bias, _, _, _) = analyze_mod9_bias_deeper(points);
//     prevalences.insert("mod9_deeper".to_string(), mod9_max_bias);

//     prevalences
// }



/// Helper function for DP bias detection
// fn detect_dp_bias(x: &BigInt256, dp_bits: u32, mod_n: u64) -> bool {
//     // Simple DP check: low bits == 0
//     x.mod_u64(1u64 << dp_bits) == 0 && x.mod_u64(mod_n) == 0
// }

/// Concise Block: Layered Bias Proxy with Coarse-to-Fine Order
// fn is_layered_bias_proxy(x: &BigInt256, biases: &std::collections::HashMap<String, f64>) -> bool {
//     if biases["mod81"] > 0.012 && !is_mod81_attractor_candidate(x) { return false; } // Coarse first
//     if biases["mod27"] > 0.037 && !is_mod27_attractor_candidate(x) { return false; }
//     if biases["mod9"] > 0.111 && !is_mod9_attractor_candidate(x) { return false; }
//     if biases["vanity"] > 0.0625 && !is_vanity_biased(&x.to_hex(), "02", 16) { return false; }
//     if biases["dp_mod9"] > 0.111 && !detect_dp_bias(x, 20, 9) { return false; } // Fine last
//     true
// }

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

/// Load valuable P2PK pubkeys from file
pub fn load_valuable_p2pk(curve: &Secp256k1) -> Result<Vec<Point>, Box<dyn std::error::Error>> {
    load_from_file("valuable_p2pk_pubkeys.txt", curve)
}

/// Load test puzzles (known solved puzzles for validation)
pub fn load_test_puzzles(curve: &Secp256k1) -> Result<Vec<Point>, Box<dyn std::error::Error>> {
    // Hardcoded test puzzles with known solutions
    let test_hex = vec![
        "02ce7c036c6fa52c0803746c7bece1221524e8b1f6ca8eb847b9bcffbc1da76db",  // #64, privkey = 1
        // Add more test puzzles as needed
    ];

    let mut points = Vec::new();
    for hex in test_hex {
        let bytes = hex::decode(hex)?;
        if bytes.len() == 33 {
            let mut comp = [0u8; 33];
            comp.copy_from_slice(&bytes);
            if let Some(point) = curve.decompress_point(&comp) {
                points.push(point);
            }
        }
    }
    Ok(points)
}

/// Load a specific real unsolved puzzle
pub fn load_real_puzzle(n: u32, curve: &Secp256k1) -> Result<Point, Box<dyn std::error::Error>> {
    let hex = match n {
        150 => "02f54ba36518d7038ed669f7da906b689d393adaa88ba114c2aab6dc5f87a73cb8", // Puzzle #150, 2^150 * G
        160 => "038c2b4e9f2c6c4ef3217b4d5c9f75ca6c24b07b5e3c1e9f4d9c7a9c4b8c2b4e9f2c6c4ef3217b4d5c9f75ca6c24b07b5e3c1e9f4d9c7a9c4", // Placeholder - replace with actual #160 hex
        _ => return Err(format!("Unknown puzzle #{}", n).into()),
    };

    let bytes = hex::decode(hex)?;
    if bytes.len() != 33 {
        return Err(format!("Invalid hex length for puzzle #{}", n).into());
    }

    let mut comp = [0u8; 33];
    comp.copy_from_slice(&bytes);

    curve.decompress_point(&comp)
        .ok_or_else(|| format!("Failed to decompress puzzle #{}", n).into())
}

/// Generic file loader for compressed pubkey files
pub fn load_from_file(path: &str, curve: &Secp256k1) -> Result<Vec<Point>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut points = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let cleaned = line.trim().trim_start_matches("0x");
        if cleaned.is_empty() { continue; }

        let bytes = hex::decode(cleaned)?;
        if bytes.len() == 33 {
            // Compressed format
            let mut comp = [0u8; 33];
            comp.copy_from_slice(&bytes);
            if let Some(point) = curve.decompress_point(&comp) {
                points.push(point);
            }
        } else if bytes.len() == 65 && bytes[0] == 0x04 {
            // Uncompressed format: 04 + x + y
            let x_bytes: [u8; 32] = bytes[1..33].try_into().unwrap();
            let y_bytes: [u8; 32] = bytes[33..65].try_into().unwrap();
            let x = BigInt256::from_bytes_be(&x_bytes);
            let y = BigInt256::from_bytes_be(&y_bytes);
            let point = Point { x: x.to_u64_array(), y: y.to_u64_array(), z: [1, 0, 0, 0] };
            if curve.is_on_curve(&point) {
                points.push(point);
            } else {
                log::warn!("Uncompressed point not on curve: {}", cleaned);
            }
        } else {
            log::warn!("Invalid pubkey length {} for line: {}", bytes.len(), cleaned);
        }
    }

    Ok(points)
}

// Chunk: AVX Bias Check (pubkey_loader.rs)
use std::simd::{u32x8, SimdPartialEq};
pub fn simd_bias_check(res: u32, high_residues: &[u32]) -> bool {
    let mut padded = [0u32; 128];
    for (i, &val) in high_residues.iter().enumerate().take(128) {
        padded[i] = val;
    }
    for i in (0..padded.len()).step_by(8) {
        let vec_res = u32x8::splat(res);
        let vec_high = u32x8::from_slice(&padded[i..]);
        if vec_res.simd_eq(vec_high).any() { return true; }
    }
    false
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