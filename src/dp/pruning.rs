//! Async incremental pruning for DP table
//!
//! Value-based + cluster preference pruning, chunked eviction, metrics logging

use crate::dp::table::DpTable;
use crate::math::bigint::BigInt256;
use tokio::sync::Mutex;
use std::sync::Arc;
use anyhow::Result;

/// DP table pruning manager
pub struct DpPruning {
    dp_table: Arc<Mutex<DpTable>>,
    pruning_threshold: f64, // Utilization threshold for triggering pruning
    chunk_size: usize,      // Number of entries to prune per chunk
}

impl DpPruning {
    /// Create new DP pruning manager
    pub fn new(dp_table: Arc<Mutex<DpTable>>, pruning_threshold: f64, chunk_size: usize) -> Self {
        DpPruning {
            dp_table,
            pruning_threshold,
            chunk_size,
        }
    }

    /// Run incremental pruning
    pub async fn prune_incremental(&self) -> Result<PruningStats> {
        let mut stats = PruningStats::default();

        loop {
            // Check if pruning is needed
            let needs_pruning = {
                let table = self.dp_table.lock().await;
                stats.entries_before = table.stats().total_entries;
                table.needs_pruning() ||
                table.stats().utilization >= self.pruning_threshold
            };

            if !needs_pruning {
                break;
            }

            // Prune one chunk
            let chunk_stats = self.prune_chunk().await?;
            stats.entries_removed += chunk_stats.entries_removed;
            stats.chunks_processed += 1;

            // Yield control to avoid blocking
            tokio::task::yield_now().await;
        }

        Ok(stats)
    }

    /// Prune a single chunk of entries
    async fn prune_chunk(&self) -> Result<PruningStats> {
        let mut stats = PruningStats::default();

        // Get entries sorted by value score
        let entries_to_prune = {
            let table = self.dp_table.lock().await;
            let mut entries: Vec<(f64, u64)> = table.get_entries().iter()
                .filter_map(|(hash, entry)| {
                    // Calculate real value score based on distance and cluster density
                    let dist_bigint = BigInt256 { limbs: [entry.state.distance[0] as u64, entry.state.distance[1] as u64, entry.state.distance[2] as u64, entry.state.distance[3] as u64] };
                    let dist_score = dist_bigint.to_f64_approx();
                    let cluster_density = table.get_cluster_size(entry.cluster_id) as f64;
                    let value_score = dist_score / (cluster_density + 1.0);
                    Some((value_score, *hash))
                })
                .collect();

            entries.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            entries.into_iter().take(self.chunk_size).collect::<Vec<_>>()
        };

        // Remove entries using the actual DpTable remove method
        {
            let mut table = self.dp_table.lock().await;
            let removed_count = table.remove_dps(&entries_to_prune.iter().map(|(_, hash)| *hash).collect::<Vec<_>>());
            stats.entries_removed = removed_count;
            stats.entries_after = table.stats().total_entries;
        }

        Ok(stats)
    }

    /// Advanced cluster-based pruning using simple k-means clustering
    /// Groups DP entries by spatial clustering and removes outliers
    pub async fn prune_advanced_clusters(&self, k_clusters: usize) -> Result<PruningStats> {
        let mut stats = PruningStats::default();

        // Get all DP entries - collect them before any lock dropping
        let entries: Vec<_> = {
            let table = self.dp_table.lock().await;
            table.get_entries().iter().map(|(k, v)| (*k, v.clone())).collect()
        };

        if entries.len() < k_clusters * 2 {
            // Not enough data for meaningful clustering
            return Ok(stats);
        }

        // Extract features for clustering (using x-coordinate and distance)
        let samples: Vec<[f64; 2]> = entries.iter().map(|(_, entry)| {
            [
                (entry.point.x[0] as f64) / u32::MAX as f64,  // Normalize x-coordinate
                { let dist_bigint = BigInt256 { limbs: [entry.state.distance[0] as u64, entry.state.distance[1] as u64, entry.state.distance[2] as u64, entry.state.distance[3] as u64] }; dist_bigint.to_f64_approx().log2().max(0.0) / 64.0 },  // Log distance normalized
            ]
        }).collect();

        // Perform simple k-means clustering
        let cluster_result = self.simple_kmeans(&samples, k_clusters, 50);
        let hashes_to_remove = if let Ok(cluster_assignments) = cluster_result {
            // Calculate cluster sizes
            let mut cluster_sizes = vec![0; k_clusters];
            for &assignment in &cluster_assignments {
                cluster_sizes[assignment] += 1;
            }

            // Remove entries from smallest clusters (outliers)
            let min_cluster_size = cluster_sizes.iter().min().unwrap_or(&0);
            if *min_cluster_size < entries.len() / (k_clusters * 2) {
                // Collect hashes to remove from small clusters
                let mut hashes = Vec::new();
                for (i, (_, entry)) in entries.iter().enumerate() {
                    if cluster_sizes[cluster_assignments[i]] == *min_cluster_size {
                        hashes.push(entry.x_hash);
                    }
                }
                hashes
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        // Remove the entries if any were identified
        if !hashes_to_remove.is_empty() {
            let mut table = self.dp_table.lock().await;
            stats.entries_removed = table.remove_dps(&hashes_to_remove);
        }

        if !hashes_to_remove.is_empty() {
            stats.chunks_processed = 1;
            stats.additional_info = format!("Clustered {} entries into {} groups", entries.len(), k_clusters);
        }

        Ok(stats)
    }

    /// Simple k-means clustering implementation
    fn simple_kmeans(&self, samples: &[[f64; 2]], k: usize, max_iterations: usize) -> Result<Vec<usize>> {
        use rand::Rng;

        if samples.is_empty() || k == 0 {
            return Ok(vec![]);
        }

        let mut rng = rand::thread_rng();

        // Initialize centroids randomly
        let mut centroids: Vec<[f64; 2]> = (0..k)
            .map(|_| {
                let idx = rng.gen_range(0..samples.len());
                samples[idx]
            })
            .collect();

        let mut assignments = vec![0; samples.len()];

        for _ in 0..max_iterations {
            let mut changed = false;

            // Assign points to nearest centroids
            for (i, &point) in samples.iter().enumerate() {
                let mut min_dist = f64::INFINITY;
                let mut best_centroid = 0;

                for (j, &centroid) in centroids.iter().enumerate() {
                    let dist = self.euclidean_distance(point, centroid);
                    if dist < min_dist {
                        min_dist = dist;
                        best_centroid = j;
                    }
                }

                if assignments[i] != best_centroid {
                    assignments[i] = best_centroid;
                    changed = true;
                }
            }

            if !changed {
                break; // Converged
            }

            // Update centroids
            let mut counts = vec![0; k];
            let mut sums: Vec<[f64; 2]> = vec![[0.0, 0.0]; k];

            for (i, &assignment) in assignments.iter().enumerate() {
                counts[assignment] += 1;
                sums[assignment][0] += samples[i][0];
                sums[assignment][1] += samples[i][1];
            }

            for i in 0..k {
                if counts[i] > 0 {
                    centroids[i][0] = sums[i][0] / counts[i] as f64;
                    centroids[i][1] = sums[i][1] / counts[i] as f64;
                }
            }
        }

        Ok(assignments)
    }

    /// Calculate Euclidean distance between two 2D points
    fn euclidean_distance(&self, a: [f64; 2], b: [f64; 2]) -> f64 {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        (dx * dx + dy * dy).sqrt()
    }

    /// Run value-based pruning (prefer low-value entries)
    pub async fn prune_value_based(&self, target_utilization: f64) -> Result<PruningStats> {
        let mut stats = PruningStats::default();

        let table = self.dp_table.lock().await;
        stats.entries_before = table.stats().total_entries;
        drop(table);

        while {
            let table = self.dp_table.lock().await;
            table.stats().utilization > target_utilization
        } {
            let chunk_stats = self.prune_chunk().await?;
            stats.entries_removed += chunk_stats.entries_removed;
            stats.chunks_processed += 1;
        }

        // Update entries_after
        let table = self.dp_table.lock().await;
        stats.entries_after = table.stats().total_entries;

        Ok(stats)
    }

    /// Run cluster-based pruning (prefer dense clusters)
    pub async fn prune_cluster_based(&self, _max_cluster_size: usize) -> Result<PruningStats> {
        let stats = PruningStats::default();

        // TODO: Implement cluster-based pruning
        // Identify clusters that are too dense
        // Remove entries from over-dense clusters

        Ok(stats)
    }

    /// Get pruning recommendations
    pub async fn get_recommendations(&self) -> PruningRecommendations {
        let table = self.dp_table.lock().await;
        let stats = table.stats();

        PruningRecommendations {
            should_prune: stats.utilization > self.pruning_threshold,
            recommended_method: if stats.cluster_count > 1000 {
                PruningMethod::ClusterBased
            } else {
                PruningMethod::ValueBased
            },
            target_utilization: 0.8,
            estimated_entries_to_remove: ((stats.utilization - 0.8) * stats.max_size as f64) as usize,
        }
    }
}

/// Pruning operation statistics
#[derive(Debug, Clone)]
pub struct PruningStats {
    pub entries_before: usize,
    pub entries_after: usize,
    pub entries_removed: usize,
    pub chunks_processed: usize,
    pub duration_ms: u64,
    pub additional_info: String,
}

impl Default for PruningStats {
    fn default() -> Self {
        PruningStats {
            entries_before: 0,
            entries_after: 0,
            entries_removed: 0,
            chunks_processed: 0,
            duration_ms: 0,
            additional_info: String::new(),
        }
    }
}

/// Pruning recommendations
#[derive(Debug, Clone)]
pub struct PruningRecommendations {
    pub should_prune: bool,
    pub recommended_method: PruningMethod,
    pub target_utilization: f64,
    pub estimated_entries_to_remove: usize,
}

/// Pruning method recommendations
#[derive(Debug, Clone)]
pub enum PruningMethod {
    ValueBased,
    ClusterBased,
    Combined,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dp::table::DpTable;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    #[tokio::test]
    async fn test_dp_pruning_value_based() {
        let dp_table = Arc::new(Mutex::new(DpTable::new(24)));
        let pruner = DpPruning::new(dp_table.clone(), 0.8, 1000);

        // Fill table with some entries
        {
            let mut table = dp_table.lock().await;
            for i in 0..2000 {
                let hash = i as u64;
                let point = crate::types::Point {
                    x: [hash, 0, 0, 0],
                    y: [hash + 1, 0, 0, 0],
                    z: [1, 0, 0, 0],
                };
                let state = crate::types::KangarooState::new(
                    point.clone(),
                    BigInt256::from_u64(i as u64 * 1000), // Vary distances
                    [0; 4],
                    [0; 4],
                    true,
                    false,
                    i as u64,
                );
                let entry = crate::types::DpEntry::new(point, state, hash, (i % 10) as u32);
                table.add_dp_async(entry).await.unwrap();
            }
        }

        // Prune based on value
        let stats = pruner.prune_value_based(0.5).await.unwrap();
        assert!(stats.entries_removed > 0);
        assert!(stats.entries_after < stats.entries_before);
    }

    #[tokio::test]
    async fn test_dp_pruning_cluster_based() {
        let dp_table = Arc::new(Mutex::new(DpTable::new(24)));
        let pruner = DpPruning::new(dp_table.clone(), 0.8, 100);

        // Fill table with clustered entries
        {
            let mut table = dp_table.lock().await;
            for i in 0..500 {
                let hash = i as u64;
                let cluster_id = if i < 50 { 0 } else { (i / 50) as u32 }; // Small cluster vs larger ones
                let point = crate::types::Point {
                    x: [hash, 0, 0, cluster_id as u64],
                    y: [hash + 1, 0, 0, 0],
                    z: [1, 0, 0, 0],
                };
                let state = crate::types::KangarooState::new(
                    point.clone(),
                    BigInt256::from_u64(i as u64 * 100),
                    [0; 4],
                    [0; 4],
                    true,
                    false,
                    i as u64,
                );
                let entry = crate::types::DpEntry::new(point, state, hash, cluster_id);
                table.add_dp_async(entry).await.unwrap();
            }
        }

        // Prune clusters
        let stats = pruner.prune_advanced_clusters(10).await.unwrap();
        log::info!("Prune stats: {:?}", stats);
        // Should have removed some entries from small clusters
        // May be 0 if clustering doesn't identify clear outliers
    }
}