    /// Create new KangarooManager
    pub fn new(config: Config) -> Result<Self> {
        println!("DEBUG: Entered KangarooManager::new");
        println!("DEBUG: Starting target loading...");

        // Load targets properly
        use crate::targets::TargetLoader;
        println!("DEBUG: Creating TargetLoader");
        let target_loader = TargetLoader::new();
        println!("DEBUG: Calling load_targets");
        let targets = target_loader.load_targets(&config)?;
        println!("DEBUG: load_targets completed, got {} targets", targets.len());

        info!("Loaded {} targets", targets.len());

        // Load targets with priority support
        let targets = if let Some(priority_path) = &config.priority_list {
            info!("Loading high-priority targets from {}", priority_path.display());
            match load_pubkeys_from_file(priority_path) {
                Ok(priority_points) => {
                    info!("Loaded {} high-priority points", priority_points.len());
                    if priority_points.len() < targets.len() / 10 {
                        warn!("Priority list very small ({}), may not be optimal", priority_points.len());
                    }
                    // Convert points to targets
                    priority_points.into_iter().enumerate().map(|(i, point)| {
                        Target::new(point, i as u64)
                    }).collect()
                }
                Err(e) => {
                    warn!("Failed to load priority list: {}, falling back to full list", e);
                    targets
                }
            }
        } else {
            targets
        };

        // Initialize POS pre-seed baseline (always active per rules)
        info!("Initializing POS pre-seed baseline for unsolved puzzles...");
        let preseed_pos = if let Some(first_target) = targets.first() {
            // Use first target range as representative
            let range_min = k256::Scalar::ZERO;
            let range_width = k256::Scalar::from(2u64).pow_vartime(&[20]); // 2^20
            crate::utils::bias::generate_preseed_pos(&range_min, &range_width)
        } else {
            vec![] // Empty if no targets
        };

        let empirical_pos = config.bias_log.as_ref()
            .map(|log_path| crate::utils::bias::load_empirical_pos(log_path))
            .unwrap_or(None);
        let blended_pos = crate::utils::bias::blend_proxy_preseed(
            preseed_pos,
            1000, // num_random
            empirical_pos,
            (0.5, 0.25, 0.25), // weights
            config.enable_noise
        );
        info!("Blended proxy positions: {} total", blended_pos.len());

        // Generate cascade histogram for POS filter tuning
        let cascades = crate::utils::bias::analyze_preseed_cascade(&blended_pos, 10);
        let max_bias = cascades.iter().map(|(_, bias)| *bias).fold(0.0, f64::max);
        info!("POS cascade analysis: max bias factor {:.2}x", max_bias);

        // Initialize components
        let dp_table = Arc::new(Mutex::new(DpTable::new(config.dp_bits)));

        // Initialize bloom filter if enabled
        let bloom = if config.use_bloom {
            let expected_dps = (config.herd_size as f64 * config.max_ops as f64) / 2f64.powi(config.dp_bits as i32);
            Some(Bloom::new_for_fp_rate(expected_dps as usize, 0.01))
        } else {
            None
        };