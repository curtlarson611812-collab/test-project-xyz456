    if let Some(sol) = solution {
        let private_key_bigint = BigInt256::from_u64_array(sol.private_key);
        println!("[VICTORY] Solution found! Private key: {}", private_key_bigint.to_hex());
    } else {
        println!("[VICTORY] Full range hunt completed - no solution found.");
    }