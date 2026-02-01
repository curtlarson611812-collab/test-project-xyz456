use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use speedbitcrack::gpu::backend::GpuBackend;
use speedbitcrack::types::{KangarooState, Point};
use speedbitcrack::kangaroo::collision::Trap;
#[cfg(feature = "cudarc")]
use std::process::Command;

fn bench_step_batch(c: &mut Criterion) {
    // Only run CUDA benchmarks if CUDA feature is enabled
    #[cfg(feature = "cudarc")]
    {
        let backend = speedbitcrack::gpu::backend::CudaBackend::new().unwrap();

        // Create test data - 1024 kangaroos
        let batch_size = 1024;
        let mut positions = vec![[[0u32; 8]; 3]; batch_size];
        let mut distances = vec![[0u32; 8]; batch_size];
        let types = vec![0u32; batch_size];

        // Initialize with some test data
        for i in 0..batch_size {
            // Set some basic point data
            positions[i][0] = [1, 0, 0, 0, 0, 0, 0, 0]; // X coordinate
            positions[i][1] = [2, 0, 0, 0, 0, 0, 0, 0]; // Y coordinate
            positions[i][2] = [1, 0, 0, 0, 0, 0, 0, 0]; // Z coordinate
            distances[i] = [i as u32, 0, 0, 0, 0, 0, 0, 0];
        }

        c.bench_function("cuda_step_batch_1024", |b| {
            b.iter(|| {
                let _traps = backend.step_batch(&mut positions, &mut distances, &types);
            })
        });

        // Automated Nsight Compute profiling for performance validation
        c.bench_function("cuda_step_batch_with_ncu_profiling", |b| {
            b.iter(|| {
                let _traps = backend.step_batch(&mut positions, &mut distances, &types);
                // Run Nsight Compute metrics collection
                let output = Command::new("ncu")
                    .arg("--metrics")
                    .arg("sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes_read.sum,sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active")
                    .arg("--target-processes")
                    .arg("all")
                    .arg("--replay-mode")
                    .arg("application")
                    .output();
                match output {
                    Ok(result) => {
                        let metrics = String::from_utf8_lossy(&result.stdout);
                        // Log metrics for analysis (would be collected in CI)
                        println!("NCU Metrics: {}", metrics);
                    }
                    Err(_) => {
                        // Nsight not available, continue with benchmark
                    }
                }
            })
        });
    }

    // CPU fallback benchmark
    #[cfg(not(feature = "cudarc"))]
    {
        c.bench_function("cpu_step_batch_1024", |b| {
            b.iter(|| {
                // CPU implementation would go here
                // For now, just a placeholder
                let _result = 42;
            })
        });
    }
}

fn bench_occupancy_check(c: &mut Criterion) {
    // Occupancy benchmark for Nsight profiling (target >90% on RTX 5090)
    #[cfg(feature = "cudarc")]
    {
        let backend = speedbitcrack::gpu::backend::CudaBackend::new().unwrap();

        let mut group = c.benchmark_group("occupancy");

        // Test different batch sizes for optimal occupancy
        for batch_size in [256, 512, 1024, 2048, 4096].iter() {
            let mut positions = vec![[[0u32; 8]; 3]; *batch_size];
            let mut distances = vec![[0u32; 8]; *batch_size];
            let types = vec![0u32; *batch_size];

            // Initialize test data
            for i in 0..*batch_size {
                positions[i][0] = [1, 0, 0, 0, 0, 0, 0, 0]; // X
                positions[i][1] = [2, 0, 0, 0, 0, 0, 0, 0]; // Y
                positions[i][2] = [1, 0, 0, 0, 0, 0, 0, 0]; // Z
                distances[i] = [i as u32, 0, 0, 0, 0, 0, 0, 0];
            }

            group.bench_with_input(
                BenchmarkId::from_parameter(batch_size),
                batch_size,
                |b, _batch_size| {
                    b.iter(|| {
                        let _traps = backend.step_batch(&mut positions, &mut distances, &types);
                    })
                }
            );
        }
        group.finish();
    }
}

criterion_group!(benches, bench_step_batch, bench_occupancy_check);
criterion_main!(benches);