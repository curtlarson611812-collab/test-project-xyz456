# SpeedBitCrackV3 - High-Performance ECDLP Solver

A cutting-edge implementation of Pollard's rho/kangaroo algorithm for secp256k1 elliptic curve discrete logarithm problems, optimized for Bitcoin puzzle solving.

## 🚀 Features

- **Hybrid GPU Backend**: Vulkan + CUDA acceleration with dynamic load balancing
- **Advanced Bias Exploitation**: Mod9, Mod27, Mod81 residue analysis
- **Comprehensive Testing**: Unit tests, benchmarks, and validation framework
- **Nsight Compute Integration**: Automated GPU kernel profiling and optimization
- **Thermal Management**: Dynamic scaling based on GPU temperature
- **Checkpointing**: Resume interrupted solves with binary state persistence

## 🔧 Quick Start

```bash
# Build the project
cargo build --release

# Run basic tests
cargo test --lib --features smoke

# Solve a small puzzle with GPU acceleration
cargo run -- --puzzle=32 --gpu

# Profile and optimize GPU performance
./scripts/profile_and_analyze.sh
```

## 📊 Performance Profiling

SpeedBitCrackV3 includes comprehensive GPU profiling capabilities using NVIDIA Nsight Compute:

### Quick Profiling
```bash
# Profile GPU kernels and get optimization recommendations
./scripts/profile_and_analyze.sh
```

### Manual Profiling
```bash
# Occupancy and utilization metrics
ncu --metrics sm_efficiency,achieved_occupancy,warp_execution_efficiency ./target/release/speedbitcrack --puzzle=32

# Memory hierarchy analysis
ncu --metrics l1tex__t_bytes_hit_rate,l2tex__t_bytes_hit_rate,dram__bytes_read.sum.pct_of_peak_sustained_active ./target/release/speedbitcrack --puzzle=66

# Compute throughput analysis
ncu --metrics sm__pipe_alu_cycles_active.average.pct_of_peak_sustained_active,sm__inst_executed.avg.pct_of_peak_sustained_active ./target/release/speedbitcrack --puzzle=32
```

### Automated Optimization

The solver automatically applies optimizations based on profiling results:

- **Memory-bound**: Reduces kangaroo count when DRAM utilization >80%
- **Occupancy-bound**: Adjusts block sizes when SM efficiency <70%
- **Compute-bound**: Increases parallelism when ALU utilization >90%
- **Thermal throttling**: Scales back when GPU temperature >75°C

## 🎯 Usage Examples

### Basic Puzzle Solving
```bash
# Solve puzzle #32 (quick validation)
cargo run -- --puzzle=32 --gpu

# Solve with laptop optimizations
cargo run -- --puzzle=66 --gpu --laptop

# Bias analysis for unsolved puzzles
cargo run -- --bias-analysis
```

### Performance Benchmarking
```bash
# Run benchmarks with Nsight profiling
NVIDIA_COMPUTE=1 ./scripts/setup_profiling.sh bench_puzzle66

# View comprehensive metrics
cat ci_metrics.json | jq '.rho_kernel'
```

### Advanced Configuration
```bash
# Custom kangaroo count and bias modulus
cargo run -- --puzzle=67 --gpu --num-kangaroos=1024 --bias-mod=81

# Enable thermal logging
cargo run -- --puzzle=67 --gpu --laptop  # Auto-starts nvidia-smi logging
```

## 📈 Key Metrics Monitored

### GPU Performance Indicators
- **SM Efficiency**: Target >70% (GPU utilization)
- **L2 Cache Hit Rate**: Target >70% (memory efficiency)
- **DRAM Utilization**: Target <80% (avoid bandwidth saturation)
- **ALU Utilization**: Target >80% (compute efficiency)
- **Occupancy**: Target 60-75% (parallelism efficiency)

### Optimization Triggers
- Memory-bound: DRAM >80% → reduce kangaroo count
- Compute-bound: ALU >90% → increase parallelism
- Thermal: Temp >75°C → reduce load
- Occupancy: SM eff <70% → adjust kernel parameters

## 🏗️ Architecture

### Core Components
- **Math Library**: Custom BigInt256 with Barrett reduction
- **Kangaroo Engine**: Parallel state management with DP tables
- **GPU Backends**: Vulkan (bulk), CUDA (precision), Hybrid (coordination)
- **Bias Analysis**: Statistical residue pattern detection
- **Checkpointing**: Binary state persistence for long runs

### Hybrid Backend Architecture
```
┌─────────────────┐    ┌─────────────────┐
│   Vulkan Bulk   │    │  CUDA Precision │
│   Operations    │    │   Operations    │
│                 │    │                 │
│ • Kangaroo      │    │ • Modular inv   │
│   stepping      │    │ • BigInt mul    │
│ • State update  │    │ • Collision     │
│ • Memory mgmt   │    │   solving       │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────────────────────┘
               Hybrid Manager
          • Load balancing
          • Nsight metrics
          • Thermal scaling
          • Dynamic optimization
```

## 🔬 Nsight Compute Integration

### Metric Categories

1. **Occupancy Metrics**: SM efficiency, achieved occupancy, warp execution efficiency
2. **Memory Metrics**: L1/L2 cache hit rates, DRAM utilization, memory coalescing
3. **Compute Metrics**: ALU utilization, instruction throughput, warp efficiency
4. **Launch Metrics**: Block/thread counts, register usage, grid configuration

### Automated Optimization Rules

```rust
// Example: Memory-bound detection and correction
if metrics.dram_utilization > 0.8 {
    config.max_kangaroos /= 2;  // Reduce memory pressure
    log::info!("Applied memory optimization: {} kangaroos", config.max_kangaroos);
}
```

### Profiling Workflow

1. **Collect Metrics**: Run Nsight profiling on representative workloads
2. **Analyze Results**: Parse CSV outputs and generate recommendations
3. **Apply Optimizations**: Automatically adjust configuration parameters
4. **Verify Improvements**: Re-profile to confirm optimization effectiveness

## 📚 Documentation

- `docs/nsight_profiling.md`: Detailed profiling guide and optimization strategies
- `scripts/setup_profiling.sh`: Automated profiling setup and data collection
- `scripts/profile_and_analyze.sh`: One-click performance analysis

## 🧪 Testing

```bash
# Run unit tests
cargo test --lib

# Run benchmarks
cargo bench

# Run smoke tests with GPU
cargo test --lib --features smoke

# Profile GPU performance
./scripts/profile_and_analyze.sh
```

## 🤝 Contributing

This project uses advanced GPU profiling techniques. When contributing:

1. Run profiling before/after changes: `./scripts/profile_and_analyze.sh`
2. Ensure SM efficiency remains >70%
3. Test on target hardware (RTX 30-series recommended)
4. Document any new metrics or optimization rules

## ⚠️ Disclaimer

This software is for research and educational purposes in computational cryptography. Use responsibly and in compliance with applicable laws and regulations.