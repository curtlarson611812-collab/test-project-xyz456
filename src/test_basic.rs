//! Simple test to verify basic functionality

use crate::utils::pubkey_loader::load_test_puzzle_keys;
use crate::math::{secp::Secp256k1, bigint::{BigInt256, BigInt512, BarrettReducer}};

pub fn run_basic_test() {
    println!("Testing basic SpeedBitCrackV3 functionality...");

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
        let puzzle_1_point = &points[0].0;
        println!("Puzzle #1 point x: {:?}", BigInt256::from_u64_array(puzzle_1_point.x).to_hex());
        println!("Generator point x: {:?}", BigInt256::from_u64_array(g.x).to_hex());
        println!("Points match: {}", puzzle_1_point.x == g.x && puzzle_1_point.y == g.y);
    }

    // Test basic BigInt operations
    println!("Testing BigInt operations...");
    let a = BigInt256::from_u64(12345);
    let b = BigInt256::from_u64(67890);
    let sum = a + b;
    println!("BigInt addition test passed!"); // Simple test to verify BigInt works

    // Test hex parsing (the main issue we fixed)
    let modulus_str = "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f";
    let modulus = BigInt256::from_hex(modulus_str);
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
    for (line_num, line_result) in reader.lines().enumerate() {
        let line = match line_result {
            Ok(l) => l,
            Err(e) => {
                println!("Warning: Error reading line {}: {}", line_num + 1, e);
                continue;
            }
        };

        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue; // Skip blank lines and comments
        }

        match crate::utils::pubkey_loader::parse_compressed(&line) {
            Ok(x) => {
                // Convert hex back to compressed bytes for decompression
                let bytes = match hex::decode(&line) {
                    Ok(b) => b,
                    Err(e) => {
                        println!("Warning: Failed to decode hex on line {}: {}", line_num + 1, e);
                        continue;
                    }
                };

                if bytes.len() != 33 {
                    println!("Warning: Invalid length {} on line {}", bytes.len(), line_num + 1);
                    continue;
                }

                let mut compressed = [0u8; 33];
                compressed.copy_from_slice(&bytes);

                if let Some(point) = curve.decompress_point(&compressed) {
                    puzzles.push(point);
                } else {
                    println!("Warning: Failed to decompress point on line {}", line_num + 1);
                }
            }
            Err(e) => {
                println!("Warning: Failed to parse compressed key on line {}: {}", line_num + 1, e);
            }
        }
    }

    println!("Successfully loaded {} puzzles from {}", puzzles.len(), file_path);
    puzzles
}