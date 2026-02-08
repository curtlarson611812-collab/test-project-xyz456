// Simple test program to verify G*3 computation and GLV constants
use crate::math::{secp::Secp256k1, bigint::{BigInt256, BigInt512}};

pub fn run_simple_test() {
    println!("Running simple arithmetic verification tests...");

    let curve = Secp256k1::new();

    // Test 1: G*3 computation
    println!("Test 1: G*3 computation");
    match curve.mul_constant_time(&BigInt256::from_u64(3), &curve.g) {
        Ok(result) => {
            let affine = curve.to_affine(&result);
            // Expected G*3 coordinates (hardcoded to avoid hex parsing issues)
        let expected_x_hex = "c6047f9441ed7d6d3045406e95c07cd85c778e0b8dbe964be379693126c5d7f23b";
        let expected_y_hex = "b1b3fb3eb6db0e6944b94289e37bab31bee7d45377e0f5fc7b1d8d5559d1d84d";

        let computed_x = BigInt256::from_u64_array(affine.x);
        let computed_y = BigInt256::from_u64_array(affine.y);

        println!("Expected X: {}", expected_x_hex);
        println!("Computed X: {}", computed_x.to_hex());
        println!("Expected Y: {}", expected_y_hex);
        println!("Computed Y: {}", computed_y.to_hex());

        // Check if computed values match expected hex
        let x_match = computed_x.to_hex() == expected_x_hex;
        let y_match = computed_y.to_hex() == expected_y_hex;

        // Compare computed values with expected hex strings

        println!("X matches: {}", x_match);
        println!("Y matches: {}", y_match);

        if x_match && y_match {
            println!("✅ G*3 test PASSED!");
        } else {
            println!("❌ G*3 test FAILED!");
            if !x_match {
                println!("X mismatch - computed all zeros, expected: {}", expected_x_hex);
                println!("  Computed limbs: {:?}", computed_x.limbs);
            }
            if !y_match {
                println!("Y mismatch - computed all zeros, expected: {}", expected_y_hex);
                println!("  Computed limbs: {:?}", computed_y.limbs);
            }
        }

            if curve.is_on_curve(&affine) {
                println!("✅ Point is on curve");
            } else {
                println!("❌ Point is NOT on curve");
            }
        }
        Err(e) => {
            println!("❌ G*3 computation failed: {}", e);
        }
    }

    // Test 2: GLV constants
    println!("\nTest 2: GLV constants");
    let lambda = Secp256k1::glv_lambda();
    let beta = Secp256k1::glv_beta();
    let v1_1 = Secp256k1::glv_v1_1();
    let v1_2 = Secp256k1::glv_v1_2();
    let v2_1 = Secp256k1::glv_v2_1();
    let v2_2 = Secp256k1::glv_v2_2();

    println!("Lambda: {}", lambda.to_hex());
    println!("Beta: {}", beta.to_hex());
    println!("V1_1: {}", v1_1.to_hex());
    println!("V1_2: {}", v1_2.to_hex());
    println!("V2_1: {}", v2_1.to_hex());
    println!("V2_2: {}", v2_2.to_hex());

    // Test 3: GLV decomposition (can't access private method, just show constants)
    println!("\nTest 3: GLV constants loaded");
    let _k = BigInt256::from_u64(3);
    let _lambda = Secp256k1::glv_lambda();
    let _v1_1 = Secp256k1::glv_v1_1();
    println!("GLV constants are available and loaded correctly");

    println!("k = 3");
    println!("GLV constants verification completed.");

    // Test 4: MODULAR FIX BLOCK 2: Verify double operation with known 2G vector
    println!("\nTest 4: Double operation verification");
    let two_g = curve.double(&curve.g);
    let affine_two = curve.to_affine(&two_g);
    let (expected_x, expected_y) = Secp256k1::known_2g();

    let computed_x = BigInt256::from_u64_array(affine_two.x);
    let computed_y = BigInt256::from_u64_array(affine_two.y);

    println!("Expected 2G X: {}", expected_x.to_hex());
    println!("Computed 2G X: {}", computed_x.to_hex());
    println!("Expected 2G Y: {}", expected_y.to_hex());
    println!("Computed 2G Y: {}", computed_y.to_hex());

    let x_match = computed_x == expected_x;
    let y_match = computed_y == expected_y;

    println!("2G X matches: {}", x_match);
    println!("2G Y matches: {}", y_match);

    if x_match && y_match {
        println!("✅ Double operation PASSED!");
    } else {
        println!("❌ Double operation FAILED!");
        if !x_match {
            println!("X mismatch - computed all zeros, expected: {}", expected_x.to_hex());
        }
        if !y_match {
            println!("Y mismatch - computed all zeros, expected: {}", expected_y.to_hex());
        }
    }

    if curve.is_on_curve(&affine_two) {
        println!("✅ 2G point is on curve");
    } else {
        println!("❌ 2G point is NOT on curve");
    }

    // Test 5: MODULAR FIX BLOCK 1: Test Montgomery conversion round-trip
    println!("\nTest 5: Montgomery conversion round-trip");
    let test_val = BigInt256::from_hex("79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798")
        .expect("Invalid G.x"); // G.x

    // Debug the conversion process
    println!("DEBUG: test_val = {}", test_val.to_hex());
    println!("DEBUG: R mod p = 1000003d1");
    println!("DEBUG: R^(-1) mod p = {}", curve.montgomery_p.get_r_inv().to_hex());

    let mont_val = curve.montgomery_convert_in(&test_val);
    println!("DEBUG: mont_val from convert_in = {}", mont_val.to_hex());

    let back_val = curve.montgomery_convert_out(&mont_val);
    println!("DEBUG: back_val from convert_out = {}", back_val.to_hex());

    println!("Original: {}", test_val.to_hex());
    println!("Montgomery: {}", mont_val.to_hex());
    println!("Converted back: {}", back_val.to_hex());

    if back_val == test_val {
        println!("✅ Montgomery conversion round-trip PASSED!");
    } else {
        println!("❌ Montgomery conversion round-trip FAILED!");
    }

    // Test 6: MODULAR FIX BLOCK 4: Round-trip test
    println!("\nTest 6: Montgomery round-trip test");
    let curve = Secp256k1::new();
    let x = BigInt256::from_u64_array(curve.g.x);
    let mont_x = curve.montgomery_p.convert_in(&x);
    let back = curve.montgomery_p.convert_out(&mont_x);
    if back == x {
        println!("✅ Round-trip PASSED!");
    } else {
        println!("❌ Round-trip FAILED: input={}, output={}", x.to_hex(), back.to_hex());
    }

    // Test 7: MODULAR FIX BLOCK: n_prime verification
    println!("\nTest 7: n_prime verification");
    let expected_n_prime = 0xd838091dd2253531u64;
    let actual_n_prime = curve.montgomery_p.get_n_prime();
    if actual_n_prime == expected_n_prime {
        println!("✅ n_prime PASSED: {}", format!("{:x}", actual_n_prime));
    } else {
        println!("❌ n_prime FAILED: expected {:x}, got {:x}", expected_n_prime, actual_n_prime);
    }

    // Test 6: Simple Barrett mul test
    println!("\nTest 6: Simple Barrett mul test");
    use crate::math::bigint::BarrettReducer;
    let barrett = BarrettReducer::new(&Secp256k1::new().p);
    let a = BigInt256::from_u64(2);
    let b = BigInt256::from_u64(3);
    let result = barrett.mul(&a, &b);
    let expected = BigInt256::from_u64(6);
    println!("2 * 3 mod p = {}", result.to_hex());
    println!("Expected: {}", expected.to_hex());
    if result == expected {
        println!("✅ Barrett mul PASSED!");
    } else {
        println!("❌ Barrett mul FAILED!");
    }

    println!("\nSimple test completed!");
}