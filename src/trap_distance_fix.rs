            let trap_distance_bytes = trap.dist.to_bytes_le();
            let trap_distance = if trap_distance_bytes.len() >= 8 {
                ((trap_distance_bytes[7] as u64) << 56) |
                ((trap_distance_bytes[6] as u64) << 48) |
                ((trap_distance_bytes[5] as u64) << 40) |
                ((trap_distance_bytes[4] as u64) << 32) |
                ((trap_distance_bytes[3] as u64) << 24) |
                ((trap_distance_bytes[2] as u64) << 16) |
                ((trap_distance_bytes[1] as u64) << 8) |
                (trap_distance_bytes[0] as u64)
            } else {
                0
            };