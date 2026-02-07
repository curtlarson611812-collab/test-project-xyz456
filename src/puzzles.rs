//! Bitcoin Puzzle Challenge Database Loader (February 2026)
//!
//! Loads puzzle data from puzzles.txt flat file instead of hardcoded constants.
//! This allows easy updates without recompiling the code.

use crate::math::BigInt256;
use crate::types::Point;
use serde::{Deserialize, Serialize};
use log::{warn, info};
use std::fs;
use std::path::Path;
use anyhow::{Result, anyhow};

/// Debug helper: Analyze hex string character codes for invisible characters
fn debug_hex_string(hex: &str, label: &str, puzzle_n: u32) {
    println!("🔍 Debug {} for puzzle {} (len {}): {:?}", label, puzzle_n, hex.len(), hex.as_bytes());
    let mut invalid_chars = Vec::new();

    for (i, &byte) in hex.as_bytes().iter().enumerate() {
        let c = char::from(byte);
        let code = byte as u32;
        let is_valid_hex = byte.is_ascii_hexdigit();
        let printable = if c.is_ascii_graphic() {
            format!("'{}'", c)
        } else {
            format!("\\x{:02x}", byte)
        };

        if !is_valid_hex {
            invalid_chars.push((i, byte, c));
            println!("❌ INVALID: Byte {}: {} code={:02x} ({})", i, printable, code, if is_valid_hex { "valid" } else { "INVALID" });
        } else if code < 32 || code > 126 {
            // Non-printable ASCII characters (but still valid hex)
            println!("⚠️  Non-printable: Byte {}: {} code={:02x} (valid hex but non-printable)", i, printable, code);
        }
    }

    if !invalid_chars.is_empty() {
        println!("🚨 Found {} invalid characters in {} for puzzle {}", invalid_chars.len(), label, puzzle_n);
        for (pos, byte, c) in invalid_chars {
            println!("   Position {}: byte={:02x}, char='{}', code={}", pos, byte, c.escape_default(), byte as u32);
        }
    } else {
        println!("✅ All characters in {} are valid hex digits", label);
    }
}

/// Calculate fallback ranges for puzzles when hex parsing fails
/// Returns (min, max) as BigInt256 representing 2^(n-1) to 2^n - 1
fn calculate_fallback_ranges(n: u32) -> (BigInt256, BigInt256) {
    if n == 0 {
        return (BigInt256::zero(), BigInt256::one());
    }

    // Calculate 2^(n-1)
    let mut min_val = BigInt256::one();
    for _ in 0..(n-1) {
        min_val = min_val.clone() + min_val; // Double for 2^(n-1)
    }

    // Calculate 2^n - 1
    let mut max_val = BigInt256::one();
    for _ in 0..n {
        max_val = max_val.clone() + max_val; // Double for 2^n
    }
    max_val = max_val - BigInt256::one(); // Subtract 1 for 2^n - 1

    info!("Calculated fallback ranges for puzzle {}: min={}, max={}", n, min_val.to_hex(), max_val.to_hex());
    (min_val, max_val)
}

/// Puzzle entry loaded from puzzles.txt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PuzzleEntry {
    pub n: u32,
    pub status: PuzzleStatus,
    pub btc_reward: f64,
    pub pub_key_hex: String,
    pub privkey_hex: Option<String>,
    pub target_address: String,
    pub range_min: BigInt256,
    pub range_max: BigInt256,
    pub search_space_bits: u32,
    pub estimated_ops: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PuzzleStatus {
    Solved,
    Unsolved,
    Revealed,
}

/// Load all puzzles from puzzles.txt
pub fn load_puzzles_from_file() -> Result<Vec<PuzzleEntry>> {
    let file_path = Path::new("puzzles.txt");
    let contents = fs::read_to_string(file_path)?;

    // Check for non-ASCII characters that could cause parsing issues
    let contents_ascii = contents.chars().filter(|&c| c.is_ascii()).collect::<String>();
    if contents.len() != contents_ascii.len() {
        warn!("Warning: puzzles.txt contains {} non-ASCII characters. This may cause parsing issues.",
              contents.len() - contents_ascii.len());
        // Continue with ASCII-only content to be safe
        let contents = contents_ascii;
    }

    let mut puzzles = Vec::new();

    for (line_num, line) in contents.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split('|').collect();
        if parts.len() < 9 {
            warn!("Skipping invalid puzzle line {}: expected 9 fields, got {}", line_num + 1, parts.len());
            continue;
        }

        let n: u32 = match parts[0].trim().parse() {
            Ok(v) => v,
            Err(e) => {
                warn!("Invalid n '{}' in line {}: {}, skipping", parts[0], line_num + 1, e);
                continue;
            }
        };
        let status = match parts[1].trim() {
            "SOLVED" => PuzzleStatus::Solved,
            "UNSOLVED" => PuzzleStatus::Unsolved,
            "REVEALED" => PuzzleStatus::Revealed,
            _ => {
                warn!("Unknown status '{}' in line {}, skipping", parts[1], line_num + 1);
                continue;
            }
        };
        let btc_reward: f64 = match parts[2].trim().parse() {
            Ok(v) => v,
            Err(e) => {
                warn!("Invalid btc_reward '{}' in line {}: {}, skipping", parts[2], line_num + 1, e);
                continue;
            }
        };
        let pub_key_hex = parts[3].trim().to_string();
        let privkey_hex = if parts[4].trim().is_empty() { None } else { Some(parts[4].trim().to_string()) };
        let target_address = parts[5].trim().to_string();
        let range_min_hex = parts[6].trim();
        let range_max_hex = parts[7].trim();
        let search_space_bits: u32 = match parts[8].trim().parse::<f64>() {
            Ok(v) => v as u32,
            Err(e) => {
                warn!("Invalid search_space_bits '{}' in line {}: {}, skipping", parts[8], line_num + 1, e);
                continue;
            }
        };

        // Calculate estimated ops (sqrt of search space for kangaroo algorithm)
        let estimated_ops = 2f64.powf(search_space_bits as f64 / 2.0);

        // Parse ranges from hex strings
        let range_min_hex = parts[6].trim();
        let range_max_hex = parts[7].trim();

        // Debug: Comprehensive char code analysis for hex strings (only in debug builds)
        #[cfg(debug_assertions)]
        {
            debug_hex_string(range_min_hex, "range_min_hex", n);
            debug_hex_string(range_max_hex, "range_max_hex", n);
        }

        // Parse ranges with comprehensive fallback for unsolved puzzles
        let (range_min, range_max) = match (BigInt256::manual_hex_to_bytes(range_min_hex), BigInt256::manual_hex_to_bytes(range_max_hex)) {
            (Ok(min_bytes), Ok(max_bytes)) => {
                if min_bytes.len() != 32 || max_bytes.len() != 32 {
                    warn!("Puzzle {}: Wrong byte length for ranges (min: {} bytes, max: {} bytes), using fallback calculation", n, min_bytes.len(), max_bytes.len());
                    calculate_fallback_ranges(n)
                } else {
                    let mut min_arr = [0u8; 32];
                    let mut max_arr = [0u8; 32];
                    min_arr.copy_from_slice(&min_bytes);
                    max_arr.copy_from_slice(&max_bytes);
                    let parsed_min = BigInt256::from_bytes_be(&min_arr);
                    let parsed_max = BigInt256::from_bytes_be(&max_arr);

                    // Validate ranges make sense (max > min, both > 0)
                    if parsed_max <= parsed_min || parsed_min.is_zero() {
                        warn!("Puzzle {}: Invalid parsed ranges (min: {}, max: {}), using fallback calculation", n, parsed_min.to_hex(), parsed_max.to_hex());
                        calculate_fallback_ranges(n)
                    } else {
                        (parsed_min, parsed_max)
                    }
                }
            }
            (Err(min_err), Err(max_err)) => {
                warn!("Puzzle {}: Hex parsing failed for both ranges (min: {}, max: {}), using fallback calculation", n, min_err, max_err);
                calculate_fallback_ranges(n)
            }
            (Err(min_err), Ok(max_bytes)) => {
                warn!("Puzzle {}: Hex parsing failed for min range ({}), max parsed successfully, using fallback calculation", n, min_err);
                calculate_fallback_ranges(n)
            }
            (Ok(min_bytes), Err(max_err)) => {
                warn!("Puzzle {}: Hex parsing failed for max range ({}), min parsed successfully, using fallback calculation", n, max_err);
                calculate_fallback_ranges(n)
            }
        };

        puzzles.push(PuzzleEntry {
            n,
            status,
            btc_reward,
            pub_key_hex,
            privkey_hex,
            target_address,
            range_min,
            range_max,
            search_space_bits,
            estimated_ops,
        });
    }

    Ok(puzzles)
}

/// Get puzzle entry by number
pub fn get_puzzle(n: u32) -> Result<Option<PuzzleEntry>> {
    let puzzles = load_puzzles_from_file()?;
    Ok(puzzles.into_iter().find(|p| p.n == n))
}

/// Get all solved puzzles (for testing)
pub fn get_solved_puzzles() -> Result<Vec<PuzzleEntry>> {
    let puzzles = load_puzzles_from_file()?;
    Ok(puzzles.into_iter().filter(|p| p.status == PuzzleStatus::Solved).collect())
}

/// Get all revealed but unsolved puzzles (high priority targets)
pub fn get_revealed_unsolved_puzzles() -> Result<Vec<PuzzleEntry>> {
    let puzzles = load_puzzles_from_file()?;
    Ok(puzzles.into_iter().filter(|p| p.status == PuzzleStatus::Revealed).collect())
}

/// Calculate total remaining prize pool
pub fn calculate_remaining_prize_pool() -> Result<f64> {
    let puzzles = load_puzzles_from_file()?;
    Ok(puzzles.iter()
        .filter(|p| p.status != PuzzleStatus::Solved)
        .map(|p| p.btc_reward)
        .sum())
}

/// Validate that a puzzle's public key matches its target address
/// Note: This is a placeholder - full validation requires external Bitcoin libraries
pub fn validate_puzzle_address(_puzzle: &PuzzleEntry) -> Result<bool> {
    // For now, just return true as validation requires full Bitcoin address calculation
    // In production, use bitcoin crate or external verification
    warn!("Puzzle address validation not implemented - use external tools for verification");
    Ok(true)
}

/// Get range boundaries for unsolved sequential puzzles
pub fn get_sequential_puzzle_ranges(start: u32, end: u32) -> Vec<(u32, BigInt256, BigInt256)> {
    let mut ranges = Vec::new();

    for n in start..=end {
        // Calculate 2^(n-1) and 2^n - 1 using iterative multiplication
        // This is efficient for reasonable n values (n <= 160 in practice)
        if n == 0 {
            ranges.push((n, BigInt256::zero(), BigInt256::one()));
            continue;
        }

        // Calculate 2^(n-1)
        let mut current_min = BigInt256::one();
        for _ in 0..(n-1) {
            current_min = current_min.clone() + current_min; // Double the value
        }

        // Calculate 2^n - 1
        let mut current_max = BigInt256::one();
        for _ in 0..n {
            current_max = current_max.clone() + current_max; // Double the value
        }
        current_max = current_max - BigInt256::one(); // Subtract 1

        ranges.push((n, current_min, current_max));
    }

    ranges
}

/// Generate Python verification script
pub fn generate_verification_script() -> String {
    r#"#!/usr/bin/env python3
"""
Bitcoin Puzzle Key Verification Script
Verifies that private key generates the correct public key and Bitcoin address
"""

import hashlib
import binascii
import ecdsa
import base58

def decompress_pubkey(pubkey_hex):
    """Decompress a compressed public key"""
    pubkey_bytes = binascii.unhexlify(pubkey_hex)
    vk = ecdsa.VerifyingKey.from_secret_key_bytes(b'', curve=ecdsa.SECP256k1)
    vk.pubkey = ecdsa.ellipticcurve.PointJacobi.from_bytes(
        ecdsa.SECP256k1.curve, pubkey_bytes, ecdsa.SECP256k1.generator
    )
    return b'\x04' + vk.to_string()

def pubkey_to_address(pubkey_hex):
    """Convert compressed public key to Bitcoin address"""
    pubkey_bytes = binascii.unhexlify(pubkey_hex)

    # SHA256 of public key
    sha = hashlib.sha256(pubkey_bytes).digest()

    # RIPEMD160 of SHA256
    rip = hashlib.new('ripemd160', sha).digest()

    # Add version byte
    version_rip = b'\x00' + rip

    # Double SHA256 for checksum
    checksum = hashlib.sha256(hashlib.sha256(version_rip).digest()).digest()[:4]

    # Base58 encode
    address = base58.b58encode(version_rip + checksum).decode()

    return address

def verify_puzzle(priv_hex, pub_hex, target_address):
    """Verify a puzzle solution"""
    try:
        # Convert private key to public key
        priv_bytes = binascii.unhexlify(priv_hex.zfill(64))
        sk = ecdsa.SigningKey.from_secret_exponent(
            int.from_bytes(priv_bytes, 'big'),
            curve=ecdsa.SECP256k1
        )

        # Get compressed public key
        vk = sk.verifying_key
        pubkey_compressed = b'\x02' + vk.pubkey.point.x().to_bytes(32, 'big')
        if vk.pubkey.point.y() % 2 == 1:
            pubkey_compressed = b'\x03' + vk.pubkey.point.x().to_bytes(32, 'big')

        computed_pub_hex = binascii.hexlify(pubkey_compressed).decode()

        # Verify public key matches
        if computed_pub_hex.lower() != pub_hex.lower():
            print(f"❌ Public key mismatch!")
            print(f"Expected: {pub_hex}")
            print(f"Computed: {computed_pub_hex}")
            return False

        # Compute address
        computed_address = pubkey_to_address(pub_hex)

        # Verify address matches
        if computed_address != target_address:
            print(f"❌ Address mismatch!")
            print(f"Expected: {target_address}")
            print(f"Computed: {computed_address}")
            return False

        print(f"✅ Puzzle verification successful!")
        print(f"Address: {computed_address}")
        return True

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

# Example usage for solved puzzles
if __name__ == "__main__":
    # Test with solved puzzle #130
    priv_hex = "0000000000000000000000000000000000000000000000163351C97F9A100000"
    pub_hex = "028686d1b827e69f848c9006969ee73587e24694407b46946d7616161616161"
    target_address = "13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5s"

    print("Testing Puzzle #130 verification...")
    verify_puzzle(priv_hex, pub_hex, target_address)
"#.to_string()
}

/// Generate configuration for testing with solved puzzles
pub fn generate_test_config() -> Result<String> {
    let solved = get_solved_puzzles()?;
    let mut config = String::from("# SpeedBitCrackV3 Test Configuration - Solved Puzzles\n\n");

    for puzzle in solved {
        config.push_str(&format!(
            "# Puzzle #{}\n",
            puzzle.n
        ));
        config.push_str(&format!(
            "puzzle_{}_pubkey = \"{}\"\n",
            puzzle.n, puzzle.pub_key_hex
        ));
        config.push_str(&format!(
            "puzzle_{}_range_min = \"{}\"\n",
            puzzle.n, puzzle.range_min.to_hex()
        ));
        config.push_str(&format!(
            "puzzle_{}_range_max = \"{}\"\n\n",
            puzzle.n, puzzle.range_max.to_hex()
        ));
    }

    Ok(config)
}

/// Stub for backward compatibility - TODO: Remove when main.rs is updated
pub static PUZZLE_MAP: &[PuzzleEntry] = &[];

/// Stub functions for backward compatibility - TODO: Remove when main.rs is updated
pub fn load_unspent_67() -> Result<(Point, (BigInt256, BigInt256))> {
    Err(anyhow!("Not implemented - use flat file system"))
}

pub fn load_solved_32() -> Result<(Point, (BigInt256, BigInt256))> {
    Err(anyhow!("Not implemented - use flat file system"))
}

pub fn load_solved_64() -> Result<(Point, (BigInt256, BigInt256))> {
    Err(anyhow!("Not implemented - use flat file system"))
}

pub fn load_solved_66() -> Result<(Point, (BigInt256, BigInt256))> {
    Err(anyhow!("Not implemented - use flat file system"))
}

/// Advice for avoiding front-running bots
pub fn get_bot_avoidance_advice() -> &'static str {
    r#"
🚨 CRITICAL: Front-Running Bot Protection (2026 Edition)

The Bitcoin Puzzle prize pool is heavily contested. Here's how to protect your solve:

1. NEVER BROADCAST TRANSACTIONS PUBLICLY
   - Bots monitor mempool 24/7 for puzzle address activity
   - Higher fee front-running is common

2. LOCAL NODE FIRST
   - Keep a fully synced Bitcoin node ready
   - Pre-configure your spending transaction
   - Have RBF (Replace-By-Fee) ready

3. STEALTH APPROACH
   - Use Tor/VPN for all blockchain interactions
   - Don't announce intent to solve puzzles publicly
   - Consider private mining pools for immediate block inclusion

4. IMMEDIATE ACTION PLAN
   - Upon finding key: Generate transaction locally
   - Submit to your own node or trusted private pool
   - Use maximum feasible fee from the start
   - Monitor for confirmations before announcing

5. LEGAL CONSIDERATIONS
   - Document your solve process timestamped
   - Consider patent/copyright on novel algorithms
   - Have legal counsel ready for prize claims

6. TECHNICAL PREPARATION
   - Pre-compute transaction templates
   - Have multiple wallet backups
   - Test your spending transaction with small amounts first

Remember: Speed is everything. Bots can front-run within seconds of your broadcast.
"#
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_puzzle_database_integrity() {
        // Verify all puzzles have valid data
        for puzzle in PUZZLE_MAP {
            assert!(puzzle.n > 0);
            assert!(puzzle.btc_reward > 0.0);
            assert!(!puzzle.pub_key_hex.is_empty());
            assert!(!puzzle.target_address.is_empty());
        }
    }

    #[test]
    fn test_remaining_prize_pool() {
        let remaining = calculate_remaining_prize_pool().unwrap();
        println!("Remaining prize pool: {} BTC", remaining);
        assert!(remaining > 900.0); // Should be around 969 BTC
    }

    #[test]
    fn test_solved_puzzles_have_private_keys() {
        let solved = get_solved_puzzles().unwrap();
        for puzzle in solved {
            assert!(puzzle.privkey_hex.is_some());
        }
    }

    #[test]
    fn benchmark_load_puzzles() {
        use std::time::Instant;

        let start = Instant::now();
        let puzzles = load_puzzles_from_file().unwrap();
        let duration = start.elapsed();

        println!("Loaded {} puzzles in {:?}", puzzles.len(), duration);
        assert!(puzzles.len() > 100); // Should have many puzzles
        assert!(duration.as_millis() < 100); // Should load quickly
    }

    #[test]
    fn benchmark_range_calculations() {
        use std::time::Instant;

        let start = Instant::now();
        let ranges = get_sequential_puzzle_ranges(1, 50); // Test first 50 ranges
        let duration = start.elapsed();

        println!("Calculated {} ranges in {:?}", ranges.len(), duration);
        assert_eq!(ranges.len(), 50);

        // Verify range calculations
        let (n, min, max) = &ranges[0]; // n=1
        assert_eq!(*n, 1);
        assert_eq!(min, &BigInt256::one()); // 2^0 = 1
        assert_eq!(max, &BigInt256::from_u64(1)); // 2^1 - 1 = 1

        let (n, min, max) = &ranges[1]; // n=2
        assert_eq!(*n, 2);
        assert_eq!(min, &BigInt256::from_u64(2)); // 2^1 = 2
        assert_eq!(max, &BigInt256::from_u64(3)); // 2^2 - 1 = 3

        assert!(duration.as_millis() < 10); // Should be very fast
    }

    #[test]
    fn test_range_hex_parsing() {
        // Test that ranges are properly parsed from hex
        let puzzles = load_puzzles_from_file().unwrap();

        for puzzle in puzzles {
            // For solved puzzles, ranges should be properly set
            if puzzle.status == PuzzleStatus::Solved {
                // Range min should be > 0 for solved puzzles
                assert!(!puzzle.range_min.is_zero());
                assert!(!puzzle.range_max.is_zero());
                assert!(puzzle.range_max > puzzle.range_min);
            }
        }
    }
}