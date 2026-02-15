        // Serialize and save to sled database
        let serialized = bincode::serialize(&checkpoint)?;
        {
            let dp_table = self.dp_table.lock().await;
            if let Some(db) = dp_table.sled_db() {
                db.insert("checkpoint", serialized)?;
                db.flush()?;
                info!("Checkpoint saved at {} ops with {} DP entries", self.total_ops, dp_entries_count);
            } else {
                warn!("Checkpoint not saved - disk storage not enabled");
            }
        }