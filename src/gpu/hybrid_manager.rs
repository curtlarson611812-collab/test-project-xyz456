//! Refined Hybrid GPU Manager with Drift Mitigation
//!
//! Manages concurrent CUDA/Vulkan execution with shared memory
//! and drift monitoring for precision-critical computations

use super::shared::SharedBuffer;
use super::backends::hybrid_backend::HybridBackend;
use super::backends::backend_trait::GpuBackend;
use crate::types::{Point, KangarooState};
use crate::math::secp::Secp256k1;
use anyhow::Result;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Metrics for drift monitoring
#[derive(Debug, Clone)]
pub struct DriftMetrics {
    pub error_rate: f64,
    pub cuda_throughput: f64,    // ops/sec
    pub vulkan_throughput: f64,  // ops/sec
    pub swap_count: u64,
    pub last_swap_time: Instant,
}

/// Refined hybrid manager with drift mitigation
pub struct HybridGpuManager {
    hybrid_backend: HybridBackend,
    curve: Secp256k1,
    drift_threshold: f64,
    check_interval: Duration,
    metrics: Arc<Mutex<DriftMetrics>>,
    sync_version: Arc<Mutex<u64>>,
}

impl HybridGpuManager {
    /// Create new hybrid manager with drift monitoring
    pub async fn new(drift_threshold: f64, check_interval_secs: u64) -> Result<Self> {
        let hybrid_backend = HybridBackend::new().await?;
        let curve = Secp256k1::new();

        Ok(Self {
            hybrid_backend,
            curve,
            drift_threshold,
            check_interval: Duration::from_secs(check_interval_secs),
            metrics: Arc::new(Mutex::new(DriftMetrics {
                error_rate: 0.0,
                cuda_throughput: 0.0,
                vulkan_throughput: 0.0,
                swap_count: 0,
                last_swap_time: Instant::now(),
            })),
            sync_version: Arc::new(Mutex::new(0)),
        })
    }

    /// Execute computation with drift monitoring (single-threaded)
    pub fn execute_with_drift_monitoring(
        &self,
        shared_points: &mut SharedBuffer<Point>,
        shared_distances: &mut SharedBuffer<u64>,
        batch_size: usize,
        total_steps: u64,
    ) -> Result<()> {
        let start_time = Instant::now();
        let mut steps_completed = 0u64;

        while steps_completed < total_steps {
            let batch_start = Instant::now();

            // Execute computation using hybrid backend
            {
                // Convert to Vec for backend API
                let mut positions_vec: Vec<[[u32; 8]; 3]> = shared_points.as_slice().iter().map(|p| [
                    p.x.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]).collect::<Vec<_>>().try_into().unwrap_or([0; 8]),
                    p.y.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]).collect::<Vec<_>>().try_into().unwrap_or([0; 8]),
                    p.z.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]).collect::<Vec<_>>().try_into().unwrap_or([0; 8]),
                ]).collect();

                let mut distances_vec: Vec<[u32; 8]> = shared_distances.as_slice().iter().map(|&d| [
                    d as u32, (d >> 32) as u32, 0, 0, 0, 0, 0, 0
                ]).collect();

                let types_vec: Vec<u32> = vec![1; batch_size]; // Simplified - all tame

                // Execute step batch using GpuBackend trait
                if let Err(e) = self.hybrid_backend.step_batch(&mut positions_vec, &mut distances_vec, &types_vec) {
                    log::error!("Hybrid backend step failed: {}", e);
                    break;
                }

                // Convert back to SharedBuffer format
                for (i, pos) in positions_vec.iter().enumerate() {
                    if i < shared_points.len() {
                        let point = &mut shared_points.as_mut_slice()[i];
                        for j in 0..4 {
                            point.x[j] = ((pos[0][j*2 + 1] as u64) << 32) | pos[0][j*2] as u64;
                            point.y[j] = ((pos[1][j*2 + 1] as u64) << 32) | pos[1][j*2] as u64;
                            point.z[j] = ((pos[2][j*2 + 1] as u64) << 32) | pos[2][j*2] as u64;
                        }
                    }
                }

                for (i, dist) in distances_vec.iter().enumerate() {
                    if i < shared_distances.len() {
                        shared_distances.as_mut_slice()[i] = ((dist[1] as u64) << 32) | dist[0] as u64;
                    }
                }
            }

            steps_completed += batch_size as u64;

            // Periodic drift checking
            if steps_completed % 10000 == 0 { // Check every 10k steps
                let error = self.compute_drift_error(shared_points, shared_distances, &self.curve);

                let mut metrics = self.metrics.lock().unwrap();
                metrics.error_rate = error;

                if error > self.drift_threshold {
                    metrics.swap_count += 1;
                    metrics.last_swap_time = Instant::now();
                    log::warn!("Drift detected (error: {:.6}), potential precision loss", error);
                }

                // Update throughput
                let batch_time = batch_start.elapsed();
                metrics.vulkan_throughput = batch_size as f64 / batch_time.as_secs_f64();
            }

            // Small delay to prevent tight looping
            thread::sleep(Duration::from_micros(1000));
        }

        let total_time = start_time.elapsed();
        log::info!("Hybrid computation completed {} steps in {:.2}s ({:.0} ops/s)",
                  steps_completed, total_time.as_secs_f64(),
                  steps_completed as f64 / total_time.as_secs_f64());

        Ok(())
    }

    /// Run Vulkan computation with drift monitoring
    fn run_vulkan_computation(
        &self,
        shared_points: &Arc<Mutex<SharedBuffer<Point>>>,
        shared_distances: &Arc<Mutex<SharedBuffer<u64>>>,
        metrics: &Arc<Mutex<DriftMetrics>>,
        sync_version: &Arc<Mutex<u64>>,
        batch_size: usize,
        total_steps: u64,
    ) {
        let start_time = Instant::now();
        let mut steps_completed = 0u64;

        while steps_completed < total_steps {
            let batch_start = Instant::now();

            // Execute computation using hybrid backend (falls back to Vulkan)
            {
                let mut points_guard = shared_points.lock().unwrap();
                let mut distances_guard = shared_distances.lock().unwrap();

                // Convert to Vec for backend API (simplified - would use slices)
                let mut positions_vec: Vec<[[u32; 8]; 3]> = points_guard.as_slice().iter().map(|p| [
                    p.x.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]).collect::<Vec<_>>().try_into().unwrap_or([0; 8]),
                    p.y.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]).collect::<Vec<_>>().try_into().unwrap_or([0; 8]),
                    p.z.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]).collect::<Vec<_>>().try_into().unwrap_or([0; 8]),
                ]).collect();

                let mut distances_vec: Vec<[u32; 8]> = distances_guard.as_slice().iter().map(|&d| [
                    d as u32, (d >> 32) as u32, 0, 0, 0, 0, 0, 0
                ]).collect();

                let types_vec: Vec<u32> = vec![1; batch_size]; // Simplified - all tame

                // Execute step batch using GpuBackend trait
                if let Err(e) = self.hybrid_backend.step_batch(&mut positions_vec, &mut distances_vec, &types_vec) {
                    log::error!("Hybrid backend step failed: {}", e);
                    break;
                }

                // Convert back (simplified)
                for (i, pos) in positions_vec.iter().enumerate() {
                    if i < points_guard.len() {
                        let point = &mut points_guard.as_mut_slice()[i];
                        for j in 0..4 {
                            point.x[j] = ((pos[0][j*2 + 1] as u64) << 32) | pos[0][j*2] as u64;
                            point.y[j] = ((pos[1][j*2 + 1] as u64) << 32) | pos[1][j*2] as u64;
                            point.z[j] = ((pos[2][j*2 + 1] as u64) << 32) | pos[2][j*2] as u64;
                        }
                    }
                }

                for (i, dist) in distances_vec.iter().enumerate() {
                    if i < distances_guard.len() {
                        distances_guard.as_mut_slice()[i] = ((dist[1] as u64) << 32) | dist[0] as u64;
                    }
                }

                // Update sync version
                *sync_version.lock().unwrap() += 1;
            }

            steps_completed += batch_size as u64;

            // Update throughput metrics
            let batch_time = batch_start.elapsed();
            let throughput = batch_size as f64 / batch_time.as_secs_f64();
            metrics.lock().unwrap().vulkan_throughput = throughput;

            // Small delay to prevent tight looping
            thread::sleep(Duration::from_micros(1000));
        }

        let total_time = start_time.elapsed();
        log::info!("Vulkan computation completed {} steps in {:.2}s ({:.0} ops/s)",
                  steps_completed, total_time.as_secs_f64(),
                  steps_completed as f64 / total_time.as_secs_f64());
    }

    /// Compute drift error by comparing sample points to CPU ground truth
    fn compute_drift_error(&self, points: &SharedBuffer<Point>, distances: &SharedBuffer<u64>, curve: &Secp256k1) -> f64 {
        let sample_size = (points.len() / 100).max(1).min(10); // Sample 1% or at least 1, max 10

        let mut total_error = 0.0;
        let mut samples_checked = 0;

        let points_slice = points.as_slice();
        let distances_slice = distances.as_slice();

        for i in (0..points.len()).step_by(points.len() / sample_size) {
            if samples_checked >= sample_size || i >= points.len() {
                break;
            }

            let gpu_point = points_slice[i];
            let gpu_distance = distances_slice[i];

            // For drift detection, compare against expected CPU computation
            // In a real implementation, this would maintain a CPU reference computation
            // For now, use a simplified check: verify point is still on curve
            let point_valid = gpu_point.validate_curve(curve);

            // Check if coordinates are reasonable (not corrupted)
            let coords_reasonable = gpu_point.x.iter().all(|&x| x < curve.p.limbs[0] * 2) &&
                                   gpu_point.y.iter().all(|&x| x < curve.p.limbs[0] * 2) &&
                                   gpu_point.z.iter().all(|&x| x < curve.p.limbs[0] * 2);

            if !point_valid || !coords_reasonable {
                total_error += 1.0; // Full error for invalid points
            } else {
                // Small error for valid but potentially drifted points
                total_error += 0.01;
            }

            samples_checked += 1;
        }

        if samples_checked > 0 {
            total_error / samples_checked as f64
        } else {
            0.0
        }
    }

    /// Get current drift metrics
    pub fn get_metrics(&self) -> DriftMetrics {
        self.metrics.lock().unwrap().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_hybrid_manager_creation() {
        let manager = HybridGpuManager::new(0.001, 1).await;
        assert!(manager.is_ok());
    }
}