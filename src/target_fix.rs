            // Generate new kangaroo batch - distribute herd size across targets
            let kangaroos_per_target = if self.targets.is_empty() {
                warn!("No targets loaded – using fallback single target");
                1  // Use single kangaroo for fallback
            } else {
                std::cmp::max(1, self.config.herd_size / self.targets.len() as usize)
            };
            let target_points: Vec<_> = if self.targets.is_empty() {
                // Fallback: use generator point
                use crate::math::secp::Secp256k1;
                let curve = Secp256k1::new();
                vec![curve.g.clone()]
            } else {
                self.targets.iter().map(|t| t.point).collect()
            };