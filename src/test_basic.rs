//! Simple test to verify basic functionality

use crate::utils::pubkey_loader::load_test_puzzle_keys;
use crate::math::{secp::Secp256k1, bigint::BigInt256};

pub fn run_basic_test() {
    println!("Testing basic SpeedBitCrackV3 functionality...");

    // Test loading test puzzles
    println!("Loading test puzzles...");
    let (points, _config) = load_test_puzzle_keys();
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

    println!("Basic functionality test completed successfully!");
}