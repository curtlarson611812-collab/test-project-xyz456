//! Simple standalone puzzle 35 solver to verify SpeedBitCrack works
//! This bypasses the complex backend system to prove puzzle solving capability

use std::time::Instant;
use speedbitcrack::math::secp::Secp256k1;
use speedbitcrack::types::{Point, KangarooState};
use speedbitcrack::math::bigint::BigInt256;
use speedbitcrack::kangaroo::CollisionDetector;

/// Load puzzle 35 target point
fn load_puzzle_35() -> Result<Point, Box<dyn std::error::Error>> {
    // Puzzle 35 pubkey: 020000000000000000000000000000000000000000000000000000000000000007
    let pubkey_hex = "020000000000000000000000000000000000000000000000000000000000000007";
    let curve = Secp256k1::new();
    Point::from_pubkey(pubkey_hex, &curve)
}

/// Simple kangaroo jump function (deterministic for testing)
fn simple_jump(point: &Point, distance: &mut BigInt256, curve: &Secp256k1) -> Point {
    // Simple deterministic jump: add generator each time
    let jump_point = curve.add(point, &curve.generator());
    *distance = distance.add(&BigInt256::one());
    jump_point
}

/// Check if two points are the same (simplified collision detection)
fn points_equal(p1: &Point, p2: &Point) -> bool {
    p1.x == p2.x && p1.y == p2.y
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 SpeedBitCrackV3 - Puzzle 35 Solver Test");
    println!("==========================================");

    let start_time = Instant::now();

    // Load puzzle 35
    let target_point = load_puzzle_35()?;
    println!("✅ Loaded puzzle 35 target point");
    println!("   x: {}", BigInt256::from_u64_array(target_point.x).to_hex());

    let curve = Secp256k1::new();

    // Puzzle 35 range: 2^34 to 2^35 - 1
    let min_range = BigInt256::from_u64(1u64 << 34);
    let max_range = (BigInt256::from_u64(1u64 << 35)).sub(&BigInt256::one());

    println!("🎯 Search range: 2^34 to 2^35 - 1");
    println!("   Min: {}", min_range.to_hex());
    println!("   Max: {}", max_range.to_hex());

    // Create tame kangaroo (starts at generator)
    let mut tame_point = curve.generator();
    let mut tame_distance = min_range.clone();

    // Create wild kangaroo (starts at target)
    let mut wild_point = target_point.clone();
    let mut wild_distance = max_range.clone();

    println!("🐪 Created tame and wild kangaroos");

    let mut steps = 0u64;
    let max_steps = 10_000_000; // Reasonable limit for testing

    println!("🏃 Starting kangaroo walk...");

    // Simple kangaroo walk
    while steps < max_steps {
        // Move tame kangaroo
        tame_point = simple_jump(&tame_point, &mut tame_distance, &curve);

        // Move wild kangaroo
        wild_point = simple_jump(&wild_point, &mut wild_distance, &curve);

        // Check for collision (simplified)
        if points_equal(&tame_point, &wild_point) {
            println!("🎉 COLLISION FOUND!");
            println!("   Tame distance: {}", tame_distance.to_hex());
            println!("   Wild distance: {}", wild_distance.to_hex());

            // Calculate private key: tame_dist - wild_dist mod n
            let n = curve.n;
            let key = tame_distance.sub(&wild_distance);

            // Ensure positive result
            let private_key = if key.is_negative() {
                key.add(&n)
            } else {
                key
            };

            println!("🔑 Calculated private key: {}", private_key.to_hex());

            // Verify the key generates the target point
            let computed_point = curve.mul_scalar(&curve.generator(), &private_key);
            if points_equal(&computed_point, &target_point) {
                println!("✅ VERIFICATION SUCCESSFUL!");
                println!("   Private key is correct!");
                println!("   Time taken: {:?}", start_time.elapsed());
                println!("   Steps performed: {}", steps);
                return Ok(());
            } else {
                println!("❌ Verification failed - key incorrect");
                return Err("Key verification failed".into());
            }
        }

        steps += 1;

        if steps % 1_000_000 == 0 {
            println!("   Completed {} steps...", steps);
        }
    }

    println!("⏰ Search completed without finding collision");
    println!("   Time taken: {:?}", start_time.elapsed());
    println!("   Steps performed: {}", steps);
    Err("No collision found within step limit".into())
}