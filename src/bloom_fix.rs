        let bloom = if config.use_bloom {
            let expected_dps = (config.herd_size as f64 * config.max_ops as f64) / 2f64.powi(config.dp_bits as i32);
            Some(Bloom::new_for_fp_rate(expected_dps as usize, 0.01))
        } else {
            None
        };