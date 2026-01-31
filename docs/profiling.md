# GPU Profiling Guide for SpeedBitCrackV3

This guide covers profiling tools and techniques for optimizing SpeedBitCrackV3's hybrid Vulkan+CUDA performance to achieve 800M-1.5B ops/sec on RTX 5090 (realistic ceiling for kangaroo ECDLP).

## CUDA Profiling with Nsight Compute

### Installation
```bash
# Ubuntu/Debian
sudo apt install nvidia-cuda-toolkit nvidia-nsight-compute-2025.1

# Verify installation
ncu --version
```

### Profiling Commands

#### Basic Occupancy Analysis
```bash
# Profile kangaroo stepping kernel
ncu --set full --target-processes all ./target/release/speedbitcrack --features cudarc

# Profile specific kernel
ncu --kernel-name "kangaroo_step_batch" --set full ./target/release/speedbitcrack
```

#### Memory Access Analysis
```bash
# Check coalesced memory access
ncu --section MemoryWorkloadAnalysis --target-processes all ./target/release/speedbitcrack

# Profile L1/L2 cache efficiency
ncu --section MemoryWorkloadAnalysis --target-processes all ./target/release/speedbitcrack
```

#### Performance Metrics
```bash
# Get occupancy and throughput metrics
ncu --section Occupancy --section SpeedOfLight --target-processes all ./target/release/speedbitcrack

# Profile fused multiplication kernel
ncu --kernel-name "fused_mul_reduce_kernel" --set full ./target/release/speedbitcrack
```

### Nsight Compute Reports

#### Key Metrics to Monitor
- **Occupancy**: Target >90% for RTX 5090
- **Memory Throughput**: Coalesced access patterns
- **Shared Memory Efficiency**: Bank conflicts <5%
- **Instruction Throughput**: Max utilization of CUDA cores

#### Common Optimizations
```cuda
// Shared memory alignment (128B for optimal performance)
__shared__ __align__(128) uint32_t shared_data[4096];

// Coalesced global memory access
uint32_t* global_ptr = global_base + (threadIdx.x * stride);
```

## Vulkan Profiling with RenderDoc

### Installation
```bash
# Ubuntu/Debian
sudo apt install renderdoc

# Verify installation
renderdoccmd --version
```

### Capture Commands

#### Vulkan Frame Capture
```bash
# Capture Vulkan compute dispatch
renderdoccmd capture -- ./target/release/speedbitcrack --features vulkan

# Capture with specific frame range
renderdoccmd capture --frame-range 1-10 -- ./target/release/speedbitcrack
```

#### Shader Performance Analysis
```bash
# Analyze compute shader occupancy
renderdoccmd analysis -- ./capture.rdc

# Check pipeline state validation
renderdoccmd validation -- ./target/release/speedbitcrack
```

### RenderDoc Analysis

#### Key Metrics to Monitor
- **Workgroup Occupancy**: Target 100% for compute shaders
- **Memory Access Patterns**: Coalesced storage buffer access
- **Pipeline Efficiency**: Minimize pipeline stalls
- **Shader Resource Usage**: Optimize descriptor sets

#### WGSL Optimizations
```wgsl
// Shared memory usage in compute shaders
var<workgroup> shared_jump_table: array<vec4u, 32>;

// Coalesced storage buffer access
@group(0) @binding(0) var<storage, read> positions: array<array<array<u32, 8>, 3>>;
```

## Performance Benchmarking

### Criterion Benchmarks
```bash
# Run all benchmarks
cargo bench --features cudarc

# Run specific benchmark
cargo bench --bench kangaroo -- step_batch

# Generate HTML reports
cargo bench --bench kangaroo -- --save-baseline --plotting
```

### Custom Profiling Integration
```rust
// Add profiling markers
#[cfg(feature = "profiling")]
{
    // Nsight range markers
    cudaProfilerStart();
    // ... GPU operations ...
    cudaProfilerStop();
}
```

## Optimization Checklist

### CUDA Kernel Optimizations
- [ ] Occupancy > 90% (check with Nsight)
- [ ] Coalesced memory access patterns
- [ ] Shared memory bank conflicts < 5%
- [ ] Minimal control flow divergence
- [ ] Optimal thread block sizes (256 threads)

### Vulkan Shader Optimizations
- [ ] Workgroup size optimization (256 threads)
- [ ] Shared memory utilization
- [ ] Coalesced storage buffer access
- [ ] Minimize subgroup size variance
- [ ] Optimal descriptor set layouts

### System-Level Optimizations
- [ ] PCI Express bandwidth utilization
- [ ] CPU-GPU synchronization overhead
- [ ] Memory allocation/deallocation patterns
- [ ] Async operation overlap efficiency

## Troubleshooting Common Issues

### Low Occupancy
```bash
# Check block size and shared memory usage
ncu --section Occupancy ./target/release/speedbitcrack

# Solutions:
# - Reduce shared memory per thread
# - Increase thread block size
# - Optimize register usage
```

### Memory Bottlenecks
```bash
# Analyze memory access patterns
ncu --section MemoryWorkloadAnalysis ./target/release/speedbitcrack

# Solutions:
# - Use shared memory for frequently accessed data
# - Implement coalesced access patterns
# - Consider memory layout optimizations
```

### Vulkan Validation Errors
```bash
# Check for Vulkan API misuse
renderdoccmd validation -- ./target/release/speedbitcrack

# Solutions:
# - Fix descriptor set bindings
# - Correct pipeline state objects
# - Validate shader interfaces
```

## Performance Targets Verification

### RTX 5090 Expected Performance
- **Kangaroo Stepping**: 800M-1.5B operations/second (realistic ceiling)
- **Modular Arithmetic**: 2.8B operations/second
- **Memory Bandwidth**: 80%+ utilization
- **GPU Occupancy**: 90%+ for compute kernels

### Benchmark Commands
```bash
# Full performance suite
cargo bench --features cudarc -- --save-baseline

# Continuous monitoring
cargo bench --features cudarc -- --bench

# Profile-guided optimization
ncu --set full --target-processes all ./target/release/speedbitcrack
```

This profiling setup enables systematic optimization to achieve SpeedBitCrackV3's 800M-1.5B ops/sec performance target on RTX 5090 hardware (realistic ceiling for kangaroo ECDLP).