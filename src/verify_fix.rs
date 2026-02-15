        let is_valid = computed_affine.x == target_affine.x && computed_affine.y == target_affine.y;

        if is_valid {
            info!("✅ Solution verified: private key {:032x}{:032x}{:032x}{:032x}",
                  solution.private_key[3], solution.private_key[2],
                  solution.private_key[1], solution.private_key[0]);
        } else {
            warn!("❌ Solution verification failed");
        }

        Ok(is_valid)