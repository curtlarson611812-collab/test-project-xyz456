                let distance = ((dist[1] as u64) << 32) | dist[0] as u64;

                KangarooState::new(
                    position,
                    BigInt256::from_u64(distance), // distance as BigInt256
                    kangaroos[i].alpha,
                    kangaroos[i].beta,
                    kangaroos[i].is_tame,
                    kangaroos[i].is_dp,
                    kangaroos[i].id,
                    kangaroos[i].step,
                    if kangaroos[i].is_tame { 1 } else { 0 }, // kangaroo_type
                )
            })
            .collect();