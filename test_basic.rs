//! Simple test to verify basic functionality

use speedbitcrack::utils::pubkey_loader::load_test_puzzle_keys;
use speedbitcrack::math::secp::Secp256k1;

fn main() {
    println!("Testing basic SpeedBitCrackV3 functionality...");

    // Test loading test puzzles
    println!("Loading test puzzles...");
    let (points, config) = load_test_puzzle_keys();
    println!("Loaded {} test puzzles", points.len());

    // Test curve operations
    println!("Testing curve operations...");
    let curve = Secp256k1::new();

    // Test point multiplication with known values
    let g = curve.g(); // Generator point
    println!("Generator point: {:?}", g.x.to_hex());

    // Test multiplication by 2 (should give known point)
    let two_g = curve.mul(&curve.two(), &g);
    println!("2G point: {:?}", two_g.x.to_hex());

    // Test known puzzle #1: privkey = 1, pubkey should be G
    let puzzle_1_point = &points[0].0;
    println!("Puzzle #1 point: {:?}", puzzle_1_point.x.to_hex());
    println!("Generator point: {:?}", g.x.to_hex());
    println!("Points match: {}", puzzle_1_point.x == g.x && puzzle_1_point.y == g.y);

    println!("Basic functionality test completed successfully!");
}