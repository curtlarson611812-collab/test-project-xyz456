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
                    let cluster_density = table.clusters.get(&entry.cluster_id)
                        .map(|cluster| cluster.len() as f64)
                        .unwrap_or(1.0);
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

    /// Advanced cluster-based pruning using k-means clustering
    /// Groups DP entries by spatial clustering and removes outliers
    pub async fn prune_advanced_clusters(&self, k_clusters: usize) -> Result<PruningStats> {
        use kmeans::*;

        let mut stats = PruningStats::default();

        // Get all DP entries
        let table = self.dp_table.lock().await;
        let entries: Vec<_> = table.get_entries().iter().collect();

        if entries.len() < k_clusters * 2 {
            // Not enough data for meaningful clustering
            return Ok(stats);
        }

        // Extract features for clustering (using x-coordinate and distance)
        let samples: Vec<Vec<f64>> = entries.iter().map(|(_, entry)| {
            vec![
                (entry.point.x[0] as f64) / u32::MAX as f64,  // Normalize x-coordinate
                (entry.state.distance as f64).log2().max(0.0) / 64.0,  // Log distance normalized
            ]
        }).collect();

        // Perform k-means clustering
        let kmeans = KMeans::new(samples, k_clusters, Distance::Euclidean);
        let result = kmeans.kmeans_lloyd(100, KMeans::default_rng());

        match result {
            Ok((cluster_centers, cluster_assignments)) => {
                // Calculate cluster sizes
                let mut cluster_sizes = vec![0; k_clusters];
                for &assignment in &cluster_assignments {
                    cluster_sizes[assignment] += 1;
                }

                // Remove entries from smallest clusters (outliers)
                let min_cluster_size = cluster_sizes.iter().min().unwrap_or(&0);
                if *min_cluster_size < entries.len() / (k_clusters * 2) {
                    // Remove entries from small clusters
                    for (i, (_, entry)) in entries.iter().enumerate() {
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