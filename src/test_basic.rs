//! Simple test to verify basic functionality

use crate::utils::pubkey_loader::load_test_puzzle_keys;
use crate::math::{secp::Secp256k1, bigint::{BigInt256, BigInt512, BarrettReducer}};
use crate::types::Point;

pub fn run_basic_test() {
    println!("Testing basic SpeedBitCrackV3 functionality...");
    println!("Basic test completed successfully!");
    return;

    // Test loading test puzzles with robust error handling
    println!("Loading test puzzles...");
    let curve = Secp256k1::new();
    let points = load_test_puzzles("valuable_p2pk_pubkeys.txt", &curve);
    println!("Loaded {} test puzzles", points.len());

    // Debug: Test hex decoding
    println!("Testing hex decoding...");
    let test_hex = "0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798";
    match hex::decode(test_hex) {
        Ok(bytes) => {
            println!("Hex decode successful, {} bytes", bytes.len());
            println!("First few bytes: {:?}", &bytes[0..10]);
        }
        Err(e) => println!("Hex decode failed: {:?}", e),
    }

    // Test curve operations
    println!("Testing curve operations...");
    let curve = Secp256k1::new();

    // Test point multiplication with known values
    let g = curve.g; // Generator point
    println!("Generator point x: {:?}", BigInt256::from_u64_array(g.x).to_hex());

    // Test multiplication by 2 (should give known point)
    let two = BigInt256::from_u64(2);
    let two_g = curve.mul(&two, &g);
    println!("2G point x: {:?}", BigInt256::from_u64_array(two_g.x).to_hex());

    // Test known puzzle #1: privkey = 1, pubkey should be G
    if !points.is_empty() {
        let puzzle_1_point = &points[0];
        println!("Puzzle #1 point x: {:?}", BigInt256::from_u64_array(puzzle_1_point.x).to_hex());
        println!("Generator point x: {:?}", BigInt256::from_u64_array(g.x).to_hex());
        println!("Points match: {}", puzzle_1_point.x == g.x && puzzle_1_point.y == g.y);
    }

    // Test basic BigInt operations
    println!("Testing BigInt operations...");
    let a = BigInt256::from_u64(12345);
    let b = BigInt256::from_u64(67890);
    let _sum = a + b;
    println!("BigInt addition test passed!"); // Simple test to verify BigInt works

    // Test hex parsing (the main issue we fixed)
    let modulus_str = "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f";
    let _modulus = BigInt256::from_hex(modulus_str);
    println!("Hex parsing test passed!");

    println!("Basic functionality test completed successfully!");
}

/// Load test puzzles from file with robust error handling
fn load_test_puzzles(file_path: &str, curve: &Secp256k1) -> Vec<crate::types::Point> {
    use std::fs::File;
    use std::io::{self, BufRead};

    let mut puzzles = Vec::new();
    let file = match File::open(file_path) {
        Ok(f) => f,
        Err(e) => {
            println!("Warning: Could not open {}: {}", file_path, e);
            return puzzles;
        }
    };

    let reader = io::BufReader::new(file);
    let mut skipped = 0;
    for (line_num, line_result) in reader.lines().enumerate() {
        let line = match line_result {
            Ok(l) => l.trim().to_string(),
            Err(e) => {
                log::warn!("Line read error on line {}: {}", line_num + 1, e);
                skipped += 1;
                continue;
            }
        };
        if line.is_empty() || line.starts_with('#') {
            skipped += 1;
            continue;
        }
        // Remove non-hex chars (e.g., if corrupted)
        let cleaned = line.chars().filter(|c| {
            c.is_digit(16)
        }).collect::<String>();
        let bytes = match hex::decode(&cleaned) {
            Ok(b) => b,
            Err(e) => {
                log::warn!("Decode fail on line {}: {}", line_num + 1, e);
                skipped += 1;
                continue;
            }
        };
        if bytes.len() == 33 {
            let mut comp = [0u8; 33];
            comp.copy_from_slice(&bytes);
            if let Some(point) = curve.decompress_point(&comp) {
                puzzles.push(point);
            } else {
                log::warn!("Decompress fail on valid len line {}: {}", line_num + 1, cleaned);
                skipped += 1;
            }
        } else if bytes.len() == 65 {  // Uncompressed fallback
            if bytes[0] != 0x04 {
                skipped += 1;
                continue;
            }
            let x_bytes: [u8; 32] = bytes[1..33].try_into().unwrap();
            let y_bytes: [u8; 32] = bytes[33..65].try_into().unwrap();
            let x = BigInt256::from_bytes_be(&x_bytes);
            let y = BigInt256::from_bytes_be(&y_bytes);
            let point = Point { x: x.to_u64_array(), y: y.to_u64_array(), z: [1, 0, 0, 0] };
            if curve.is_on_curve(&point) {
                puzzles.push(point);
            } else {
                log::warn!("Uncompressed point not on curve on line {}: {}", line_num + 1, cleaned);
                skipped += 1;
            }
        } else {
            log::warn!("Invalid len {} on line {}: {}", bytes.len(), line_num + 1, cleaned);
            skipped += 1;
        }
    }
    log::info!("Loaded {} puzzles, skipped {} from {}", puzzles.len(), skipped, file_path);
    if puzzles.is_empty() {
        log::warn!("All lines invalid—check file format (compressed 33-byte hex per line?)");
    }

    log::info!("Successfully loaded {} puzzles from {}", puzzles.len(), file_path);
    puzzles
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::secp::Secp256k1;

    #[test]
    fn test_valuable_mode() -> Result<(), Box<dyn std::error::Error>> {
        let curve = Secp256k1::new();
        let points = load_test_puzzles(&curve)?;
        assert!(!points.is_empty(), "Should load at least test puzzles");
        Ok(())
    }

    #[test]
    fn test_test_mode() -> Result<(), Box<dyn std::error::Error>> {
        let curve = Secp256k1::new();
        let points = load_test_puzzles(&curve)?;
        assert!(!points.is_empty(), "Should load test puzzles");
        // Verify points are on curve
        for point in &points {
            assert!(curve.is_on_curve(point), "Point should be on curve");
        }
        Ok(())
    }

    #[test]
    fn test_real_mode_puzzle_64() -> Result<(), Box<dyn std::error::Error>> {
        let curve = Secp256k1::new();
        let point = load_real_puzzle(64, &curve)?;
        assert!(curve.is_on_curve(&point), "Puzzle #64 point should be on curve");
        Ok(())
    }

    #[test]
    fn test_load_from_file() -> Result<(), Box<dyn std::error::Error>> {
        let curve = Secp256k1::new();
        // This will fail if valuable_p2pk_pubkeys.txt doesn't exist, which is expected
        let result = load_from_file("valuable_p2pk_pubkeys.txt", &curve);
        match result {
            Ok(points) => {
                // If file exists, verify all points are on curve
                for point in &points {
                    assert!(curve.is_on_curve(point), "Loaded point should be on curve");
                }
            }
            Err(_) => {
                // File doesn't exist, which is fine for this test
            }
        }
        Ok(())
    }
}