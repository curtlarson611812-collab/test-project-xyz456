        if dp_utilization > 0.8 {
            let mut dp_table = self.dp_table.lock().await;
            if let Err(e) = dp_table.prune_entries_async().await {
                warn!("DP pruning failed: {}", e);
            } else {
                debug!("DP table pruned successfully");
            }
        }