        // Load targets with priority support
        let targets = if let Some(priority_path) = &config.priority_list {
            info!("Loading high-priority targets from {}", priority_path.display());
            match load_pubkeys_from_file(priority_path) {
                Ok(priority_points) => {
                    info!("Loaded {} high-priority points", priority_points.len());
                    if priority_points.len() < loaded_targets.len() / 10 {
                        warn!("Priority list very small ({}), may not be optimal", priority_points.len());
                    }
                    // Convert points to targets
                    priority_points.into_iter().enumerate().map(|(i, point)| {
                        Target::new(point, i as u64)
                    }).collect()
                }
                Err(e) => {
                    warn!("Failed to load priority list: {}, falling back to full list", e);
                    loaded_targets
                }
            }
        } else {
            loaded_targets
        };