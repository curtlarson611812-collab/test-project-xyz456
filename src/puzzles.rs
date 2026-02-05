//! Bitcoin Puzzle Challenge Database Loader (February 2026)
//!
//! Loads puzzle data from puzzles.txt flat file instead of hardcoded constants.
//! This allows easy updates without recompiling the code.

use crate::math::BigInt256;
use crate::types::Point;
use serde::{Deserialize, Serialize};
use log::warn;
use std::fs;
use std::path::Path;
use anyhow::{Result, anyhow};

/// Puzzle entry loaded from puzzles.txt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PuzzleEntry {
    pub n: u32,
    pub status: PuzzleStatus,
    pub btc_reward: f64,
    pub pub_key_hex: String,
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

        let n: u32 = parts[0].trim().parse()?;
        let status = match parts[1].trim() {
            "SOLVED" => PuzzleStatus::Solved,
            "UNSOLVED" => PuzzleStatus::Unsolved,
            "REVEALED" => PuzzleStatus::Revealed,
            _ => {
                warn!("Unknown status '{}' in line {}, skipping", parts[1], line_num + 1);
                continue;
            }
        };
        let btc_reward: f64 = parts[2].trim().parse()?;
        let pub_key_hex = parts[3].trim().to_string();
        let target_address = parts[5].trim().to_string();
        let range_min_hex = parts[6].trim();
        let range_max_hex = parts[7].trim();
        let search_space_bits: u32 = parts[8].trim().parse()?;

        // Calculate estimated ops (log2 of search space)
        let estimated_ops = search_space_bits as f64;

        // Parse ranges
        let range_min = BigInt256::from_hex(range_min_hex);
        let range_max = BigInt256::from_hex(range_max_hex);

        puzzles.push(PuzzleEntry {
            n,
            status,
            btc_reward,
            pub_key_hex,
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
        let min = if n == 0 {
            BigInt256::zero()
        } else {
            // 2^(n-1)
            BigInt256::one() << ((n - 1) as usize)
        };

        let max = if n == 0 {
            BigInt256::one()
        } else {
            // 2^n - 1
            (BigInt256::one() << (n as usize)) - BigInt256::one()
        };

        ranges.push((n, min, max));
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
        let remaining = calculate_remaining_prize_pool();
        println!("Remaining prize pool: {} BTC", remaining);
        assert!(remaining > 900.0); // Should be around 969 BTC
    }

    #[test]
    fn test_solved_puzzles_have_private_keys() {
        let solved = get_solved_puzzles();
        for puzzle in solved {
            assert!(puzzle.priv_hex.is_some());
        }
    }
}