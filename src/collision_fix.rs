            // Handle different collision result types
            let collision_solution = match collision_result {
                CollisionResult::Full(solution) => Some(solution),
                CollisionResult::Near(_near_states) => {
                    info!("🎯 Near collision detected - boosters available via config flags");
                    None
                }
                CollisionResult::None => None,
            };

            // Check collision result
            if let Some(solution) = collision_solution {
                info!("COLLISION DETECTED!");
                if self.verify_solution(&solution)? {
                    return Ok(Some(solution));
                }
            }