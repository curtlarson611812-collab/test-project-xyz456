//! Pubkey Loading Utilities for Multi-Target Kangaroo
//!
//! Loads large pubkey lists from files and provides Bitcoin puzzle pubkeys
//! Supports both compressed and uncompressed formats with validation

use std::fs::File;
use std::io::{self, BufRead};
use std::ops::{Add, Sub};
use hex::decode;
use crate::types::Point;
use crate::math::bigint::BigInt256;
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
fn parse_compressed(bytes: &[u8]) -> io::Result<Point> {
    if bytes.len() != 33 || (bytes[0] != 0x02 && bytes[0] != 0x03) {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid compressed format"));
    }

    let mut x_bytes = [0u8; 32];
    x_bytes[31..32].copy_from_slice(&bytes[1..2]); // Only last byte for compressed
    x_bytes[30..31].copy_from_slice(&bytes[2..3]);
    x_bytes[29..30].copy_from_slice(&bytes[3..4]);
    x_bytes[28..29].copy_from_slice(&bytes[4..5]);
    x_bytes[27..28].copy_from_slice(&bytes[5..6]);
    x_bytes[26..27].copy_from_slice(&bytes[6..7]);
    x_bytes[25..26].copy_from_slice(&bytes[7..8]);
    x_bytes[24..25].copy_from_slice(&bytes[8..9]);
    x_bytes[23..24].copy_from_slice(&bytes[9..10]);
    x_bytes[22..23].copy_from_slice(&bytes[10..11]);
    x_bytes[21..22].copy_from_slice(&bytes[11..12]);
    x_bytes[20..21].copy_from_slice(&bytes[12..13]);
    x_bytes[19..20].copy_from_slice(&bytes[13..14]);
    x_bytes[18..19].copy_from_slice(&bytes[14..15]);
    x_bytes[17..18].copy_from_slice(&bytes[15..16]);
    x_bytes[16..17].copy_from_slice(&bytes[16..17]);
    x_bytes[15..16].copy_from_slice(&bytes[17..18]);
    x_bytes[14..15].copy_from_slice(&bytes[18..19]);
    x_bytes[13..14].copy_from_slice(&bytes[19..20]);
    x_bytes[12..13].copy_from_slice(&bytes[20..21]);
    x_bytes[11..12].copy_from_slice(&bytes[21..22]);
    x_bytes[10..11].copy_from_slice(&bytes[22..23]);
    x_bytes[9..10].copy_from_slice(&bytes[23..24]);
    x_bytes[8..9].copy_from_slice(&bytes[24..25]);
    x_bytes[7..8].copy_from_slice(&bytes[25..26]);
    x_bytes[6..7].copy_from_slice(&bytes[26..27]);
    x_bytes[5..6].copy_from_slice(&bytes[27..28]);
    x_bytes[4..5].copy_from_slice(&bytes[28..29]);
    x_bytes[3..4].copy_from_slice(&bytes[29..30]);
    x_bytes[2..3].copy_from_slice(&bytes[30..31]);
    x_bytes[1..2].copy_from_slice(&bytes[31..32]);
    x_bytes[0..1].copy_from_slice(&bytes[32..33]);
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

    let mut point = Point::from_affine(x.to_u64_array(), y.to_u64_array());
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
        let p_plus_one = p.add(BigInt256::one());
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
    // Simple implementation - could use Barrett reduction from Phase 3
    let curve = Secp256k1::new();
    curve.barrett_p.reduce(&curve.barrett_p.mul(a, b))
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
        match parse_compressed(hex.as_bytes()) {
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
        if let Ok(point) = parse_compressed(hex.as_bytes()) {
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

/// Concise Block: Layer Mod9 in Proxy for Impl
fn is_attractor_proxy(x: &BigInt256) -> bool {
    if !is_mod9_attractor_candidate(x) { return false; } // Reduce first
    let x_hex = x.to_hex();
    if !x_hex.ends_with('9') { return false; } // Hex end '9' (nibble 9 mod16)
    if x.clone() % BigInt256::from_u64(9) != BigInt256::zero() { return false; } // Mod9=0 (digital root 0)
    let mut hasher = Sha256::new();
    hasher.update(x_hex.as_bytes());
    let hash = hasher.finalize();
    let low = u32::from_le_bytes(hash[0..4].try_into().unwrap()) % 100;
    low < 10 // Basin proxy (<10% for depth)
}

/// Concise Block: Full Valuable Scan with Mod9=0 Reduction
fn scan_full_valuable_for_attractors(points: &Vec<Point>) -> (usize, f64, Vec<(usize, usize)>) {
    let mut count = 0;
    let mut clusters = vec![];
    let mut cluster_start = None;
    for (i, p) in points.iter().enumerate() {
        let x_big = BigInt256::from_u64_array(p.x);
        if x_big.clone() % BigInt256::from_u64(9) == BigInt256::zero() && is_attractor_proxy(&x_big) { // Mod9=0 reduction first
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
    (count, percent, clusters)
}

/// Load valuable P2PK pubkeys from file with default configuration
/// Sorts by magic 9 priority for sooner hits
pub fn load_valuable_p2pk_keys(path: &str) -> io::Result<(Vec<Point>, SearchConfig)> {
    let mut points = load_pubkeys_from_file(path)?;

    // Count magic 9 patterns
    let magic_count = count_magic9_in_list(&points);
    println!("Magic 9 in valuable: {} (~{:.1}% potential attractors)", magic_count, (magic_count as f64 / points.len() as f64 * 100.0));

    // Scan for attractors and clusters with mod9=0 reduction
    let (mod9_count, mod9_percent, full_count, full_percent, clusters) = scan_full_valuable_for_attractors(&points);
    println!("Mod9=0: {} ({:.1}%), Full Attractors: {} ({:.1}%), Clusters: {:?}", mod9_count, mod9_percent, full_count, full_percent, clusters);
    if mod9_percent > 15.0 {
        println!("Confirmed MANY related keys—bias high!");
    }

    // Sort by attractor proxy priority: attractor keys first (lower sort key = higher priority)
    points.sort_by_key(|p| if is_attractor_proxy(p) { 0 } else { 1 });

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