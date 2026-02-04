use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use speedbitcrack::gpu::backend::GpuBackend;
use speedbitcrack::types::{KangarooState, Point};
use speedbitcrack::kangaroo::collision::Trap;
#[cfg(feature = "cudarc")]
use std::process::Command;
use num_bigint::{BigInt, BigUint};
use std::collections::HashMap;

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

fn bench_pos_slicing(c: &mut Criterion) {
    let full_range = (BigInt::from(0u64), BigInt::from(1u64) << 66);
    let mod_biases: HashMap<u32, f64> = [(9, 1.25), (27, 1.35), (81, 1.42)].into_iter().collect();

    let mut group = c.benchmark_group("pos_slicing");
    for iters in [0, 1, 2, 3] {  // 0 = no slicing (baseline)
        group.bench_with_input(
            BenchmarkId::new("refine_iters", iters),
            &iters,
            |b, &iters| {
                let mut slice = speedbitcrack::kangaroo::generator::PosSlice::new(full_range.clone(), 0);
                b.iter(|| {
                    let mut s = slice.clone();
                    for _ in 0..iters {
                        speedbitcrack::kangaroo::generator::refine_pos_slice(&mut s, &mod_biases, 5);
                    }
                    // simulate 10k kangaroo starts in the slice
                    let _starts: Vec<BigInt> = (0..10_000).map(|_| {
                        speedbitcrack::kangaroo::generator::random_in_slice(&s)
                    }).collect();
                });
            },
        );
    }
    group.finish();
}

// Chunk: #66 Range and Key Validation (benches/kangaroo.rs)
fn bench_puzzle66(c: &mut Criterion) {
    let low = BigInt::from(1u64 << 65);
    let high = (BigInt::from(1u64 << 66) - 1);
    let known_key = BigInt::from_str_radix("2832ed74f2b5e35ee", 16).unwrap();
    let target = speedbitcrack::targets::loader::load_puzzle_keys().get(65).unwrap().0;  // Pubkey
    let mut group = c.benchmark_group("puzzle66_crack");
    group.bench_function(BenchmarkId::new("full_crack", 4096), |b| {
        b.iter(|| {
            let key = speedbitcrack::kangaroo::pollard_lambda_parallel_pos(target, (low.clone(), high.clone()));
            #[cfg(feature = "smoke")]
            if let Some(k) = key { assert_eq!(k, known_key); }
            black_box(key);  // Prevent opt
        });
    });
    group.finish();
}

// Chunk: Laptop-Opt #66 Bench (benches/kangaroo.rs)
fn bench_puzzle66_laptop(c: &mut Criterion) {
    let low = BigInt::from(1u64 << 65);
    let high = (BigInt::from(1u64 << 66) - BigInt::from(1));
    let known_key = BigInt::from_str_radix("2832ed74f2b5e35ee", 16).unwrap();
    let target = speedbitcrack::targets::loader::load_puzzle_keys().get(65).unwrap().0;
    let config = speedbitcrack::config::laptop_3070_config();  // From config.rs
    let mut group = c.benchmark_group("puzzle66_laptop");
    group.bench_function("partial_crack", |b| b.iter(|| {
        let key = speedbitcrack::kangaroo::pollard_lambda_parallel_pos(target, (low.clone(), high.clone()), config.max_kangaroos, 81, 2);  // iters=2
        black_box(key);
    }));
    group.finish();
}

// Chunk: #64/#65 Benches (benches/kangaroo.rs)
fn bench_solved_puzzles(c: &mut Criterion) {
    for n in [64, 65] {
        let (low, high, known) = speedbitcrack::puzzles::load_solved(n);
        let target = speedbitcrack::targets::loader::load_puzzle_keys().get((n - 1) as usize).unwrap().0;
        let mut group = c.benchmark_group(format!("puzzle{}_crack", n));
        group.bench_function("full_crack", |b| b.iter(|| {
            let key = speedbitcrack::kangaroo::pollard_lambda_parallel_pos(target, (low.clone(), high.clone()), 4096);
            #[cfg(feature = "smoke")]
            if let Some(k) = key { assert_eq!(k, known); }
            black_box(key);
        }));
    }
}

criterion_group!(benches, bench_step_batch, bench_occupancy_check, bench_pos_slicing, bench_puzzle66, bench_puzzle66_laptop, bench_solved_puzzles);
criterion_main!(benches);