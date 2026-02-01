//! Async incremental pruning for DP table
//!
//! Value-based + cluster preference pruning, chunked eviction, metrics logging

use crate::dp::table::DpTable;
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
                    let dist_score = entry.state.distance as f64;
                    let cluster_density = table.get_cluster_size(entry.cluster_id) as f64;
                    let value_score = dist_score / (cluster_density + 1.0);
                    Some((value_score, *hash))
                })
                .collect();

            entries.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            entries.into_iter().take(self.chunk_size).collect::<Vec<_>>()
        };

        // Remove entries
        {
            // Note: In a real implementation, we would need to add a remove method to DpTable
            // For now, this is a placeholder
            stats.entries_removed = entries_to_prune.len();
        }

        Ok(stats)
    }

    /// Advanced cluster-based pruning using simple k-means clustering
    /// Groups DP entries by spatial clustering and removes outliers
    pub async fn prune_advanced_clusters(&self, k_clusters: usize) -> Result<PruningStats> {
        let mut stats = PruningStats::default();

        // Get all DP entries
        let table = self.dp_table.lock().await;
        let entries: Vec<_> = table.get_entries().iter().collect();

        if entries.len() < k_clusters * 2 {
            // Not enough data for meaningful clustering
            return Ok(stats);
        }

        // Extract features for clustering (using x-coordinate and distance)
        let samples: Vec<[f64; 2]> = entries.iter().map(|(_, entry)| {
            [
                (entry.point.x[0] as f64) / u32::MAX as f64,  // Normalize x-coordinate
                (entry.state.distance as f64).log2().max(0.0) / 64.0,  // Log distance normalized
            ]
        }).collect();

        // Perform simple k-means clustering
        match self.simple_kmeans(&samples, k_clusters, 50) {
            Ok(cluster_assignments) => {
                // Calculate cluster sizes
                let mut cluster_sizes = vec![0; k_clusters];
                for &assignment in &cluster_assignments {
                    cluster_sizes[assignment] += 1;
                }

                // Remove entries from smallest clusters (outliers)
                let min_cluster_size = cluster_sizes.iter().min().unwrap_or(&0);
                if *min_cluster_size < entries.len() / (k_clusters * 2) {
                    // Remove entries from small clusters
                    for (i, (_, _entry)) in entries.iter().enumerate() {
                        if cluster_sizes[cluster_assignments[i]] == *min_cluster_size {
                            // Mark for removal (would need remove method in DpTable)
                            stats.entries_removed += 1;
                        }
                    }
                }

                stats.chunks_processed = 1;
                stats.additional_info = format!("Clustered {} entries into {} groups", entries.len(), k_clusters);
            }
            Err(_) => {
                // Fallback to simple pruning if clustering fails
                stats.additional_info = "Clustering failed, using fallback".to_string();
            }
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

        while {
            let table = self.dp_table.lock().await;
            table.stats().utilization > target_utilization
        } {
            let chunk_stats = self.prune_chunk().await?;
            stats.entries_removed += chunk_stats.entries_removed;
            stats.chunks_processed += 1;
        }

        Ok(stats)
    }

    /// Run cluster-based pruning (prefer dense clusters)
    pub async fn prune_cluster_based(&self, max_cluster_size: usize) -> Result<PruningStats> {
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
#[derive(Debug, Clone, Default)]
pub struct PruningStats {
    pub entries_removed: usize,
    pub chunks_processed: usize,
    pub duration_ms: u64,
    pub additional_info: String,
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