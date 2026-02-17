# SpeedBitCrackV3 Test Suite

## Overview

SpeedBitCrackV3 includes a comprehensive, multi-layered test suite designed to validate the entire cryptographic pipeline from basic math operations to full GPU-accelerated kangaroo hunts. With ~28,000 lines of code, thorough testing is critical for maintaining mathematical correctness and performance.

## Test Architecture

### 1. **Bash Test Runner** (`scripts/test_runner.sh`)
- **Purpose**: CI/CD-ready test orchestration
- **Features**:
  - Phased execution (compilation → core → GPU → crypto → integration)
  - Colored output with timing
  - Graceful handling of missing GPU features
  - Optional performance benchmarks
- **Usage**:
  ```bash
  # Run all tests
  ./scripts/test_runner.sh

  # Fast tests only
  ./scripts/test_runner.sh --fast

  # Sanity check only
  ./scripts/test_runner.sh --sanity
  ```

### 2. **Python Test Runner** (`scripts/test_runner.py`)
- **Purpose**: Advanced reporting and analytics
- **Features**:
  - JSON result archiving
  - Performance regression detection
  - Parallel test execution planning
  - CI/CD integration support
- **Usage**:
  ```bash
  # Run with detailed reporting
  python3 scripts/test_runner.py --verbose

  # CI mode (quiet output)
  python3 scripts/test_runner.py --ci

  # Fast tests only
  python3 scripts/test_runner.py --fast
  ```

### 3. **Rust Test Orchestrator** (`src/test_orchestrator.rs`)
- **Purpose**: Integrated Rust test suite
- **Features**:
  - Native Rust performance
  - Structured test phases
  - Panic-safe test execution
  - Comprehensive result collection
- **Usage**:
  ```bash
  # Run orchestrator tests
  cargo test test_orchestrator

  # Run comprehensive suite
  cargo test run_comprehensive_test_suite
  ```

## Test Categories

### **Phase 1: Core Tests** (Always Run)
- **Compilation Check**: Ensure code compiles without errors
- **Library Tests**: Unit tests for all core modules
- **Math Core**: BigInt256, Secp256k1, GLV operations

### **Phase 2: GPU Backend Tests** (GPU Optional)
- **GPU Initialization**: Backend creation and feature detection
- **GPU EC Math**: Elliptic curve operations on GPU
- **Backend Compatibility**: Vulkan/CUDA interoperability

### **Phase 3: Hybrid GPU Tests** (GPU Required)
- **GPU Hybrid Suite**: Complete hybrid test suite (16 test functions)
- **Parity Validation**: CPU ↔ GPU result equivalence (52 parity checks)
- **Performance Validation**: Operation timing and throughput

### **Phase 4: Cryptographic Validation** (Always Run)
- **GLV Tests**: Gallant-Lambert-Vanstone optimization correctness
- **Puzzle Validation**: Bitcoin puzzle solving (end-to-end)
- **Bias Analysis**: Statistical distribution validation

### **Phase 5: Integration Tests** (Always Run)
- **Kangaroo Algorithm**: Core search algorithm validation
- **Multi-Target Support**: Parallel target processing
- **Configuration System**: CLI parsing and validation

### **Phase 6: Performance Benchmarks** (Optional)
- **GLV Optimization**: Stall reduction measurements
- **Throughput Tests**: Operations per second
- **Memory Benchmarks**: GPU memory utilization

## Key Test Files

| File | Purpose | Test Count | GPU Required |
|------|---------|------------|--------------|
| `src/tests/gpu_hybrid.rs` | Complete hybrid GPU validation | 16 functions | Yes |
| `src/gpu/tests.rs` | GPU backend math tests | 3 categories | Optional |
| `tests/glv.rs` | GLV optimization validation | 13 tests | No |
| `tests/puzzle.rs` | Bitcoin puzzle solving | 8 tests | Optional |
| `tests/bias_validation.rs` | Statistical analysis | 5 tests | No |
| `src/test_orchestrator.rs` | Integrated test runner | Meta-tests | No |

## GPU Hybrid Test Suite Details

### **16 Comprehensive Test Functions:**

1. **`gpu_hybrid_suite`** - Main test orchestrator
2. **`test_parity_barrett`** - Barrett reduction CPU↔GPU
3. **`test_parity_bigint_mul`** - BigInt multiplication parity
4. **`test_gpu_mod_bias`** - Modular bias operations
5. **`test_gpu_collision_solve`** - Collision solving validation
6. **`test_gpu_rho_walk`** - Near-collision detection
7. **`test_gpu_multi_target`** - Multi-target processing
8. **`test_hybrid_fallback`** - CPU fallback mechanisms
9. **`test_gpu_hybrid_puzzle66`** - End-to-end puzzle #66
10. **`test_10m_step_parity_hybrid`** - 10M step parity validation
11. **`test_preseed_pos_generation`** - Pre-seed position generation
12. **`test_preseed_blend_proxy`** - Pre-seed blending
13. **`test_preseed_cascade_analysis`** - Cascade analysis
14. **`test_scalar_operations_parity`** - Scalar math parity
15. **`test_point_operations_parity`** - Point math parity
16. **`test_memory_layout_parity`** - Memory layout validation

### **52 Parity Validations:**
- Mathematical operations (addition, multiplication, modular reduction)
- Elliptic curve operations (point addition, doubling, scalar multiplication)
- Jump table computations and precomputation
- Memory layout and data structure consistency
- Statistical bias and distribution analysis

## Running Tests

### **Quick Start**
```bash
# Run all tests with the bash runner
./scripts/test_runner.sh

# Run fast tests only (skip GPU/slow tests)
./scripts/test_runner.sh --fast
```

### **Development Workflow**
```bash
# During development - fast feedback
cargo test --lib

# Before commits - comprehensive validation
./scripts/test_runner.sh

# CI/CD - automated with reports
python3 scripts/test_runner.py --ci
```

### **GPU Testing**
```bash
# Test with Vulkan (default)
cargo test --features wgpu

# Test with CUDA
cargo test --features cuda

# Test hybrid (both CUDA + Vulkan)
cargo test --features hybrid
```

### **Performance Testing**
```bash
# Run benchmarks
cargo bench

# Profile specific operations
cargo bench --bench kangaroo bench_glv_opt_stalls
```

## Test Results Interpretation

### **Success Criteria**
- ✅ **Phase 1-5**: All tests pass (100% success rate)
- ✅ **Phase 6**: Performance within acceptable ranges
- ✅ **GPU Tests**: Pass when hardware/features available
- ✅ **Parity Tests**: CPU and GPU results identical

### **Expected Results by Environment**

#### **CPU-Only Environment:**
- Core tests: ✅ PASS
- GPU tests: ⏭️ SKIP (expected)
- Hybrid tests: ⏭️ SKIP (expected)
- Crypto validation: ✅ PASS
- Integration: ✅ PASS

#### **GPU Environment:**
- All tests: ✅ PASS
- Parity validations: ✅ 52/52 passing
- Performance: ✅ Within targets

### **Failure Analysis**

#### **Common Issues:**
1. **Compilation Failures**: Check `cargo check` output
2. **GPU Initialization**: Verify Vulkan/CUDA drivers
3. **Mathematical Errors**: Check GLV constants and curve parameters
4. **Performance Regressions**: Compare against baseline benchmarks

#### **Debugging Steps:**
```bash
# Isolate failing tests
cargo test specific_test_name

# Verbose output
RUST_LOG=debug cargo test

# GPU debugging
cargo test --features hybrid -- --nocapture
```

## CI/CD Integration

### **GitHub Actions Example:**
```yaml
- name: Run Test Suite
  run: |
    python3 scripts/test_runner.py --ci
    # Check results in test_reports/

- name: Performance Regression Check
  run: |
    # Compare current benchmarks against baseline
    cargo bench > current_bench.txt
    # Compare with expected performance
```

### **Test Reports**
- **JSON Format**: `test_reports/test_run_YYYYMMDD_HHMMSS.json`
- **Summary**: Pass/fail counts, timing, success rates
- **Details**: Individual test results with error messages

## Maintenance

### **Adding New Tests**
1. Add test function to appropriate file
2. Update test orchestrators if needed
3. Add to appropriate phase in runners
4. Update documentation

### **Test Coverage Goals**
- **Core Math**: 100% coverage
- **GPU Operations**: 95% coverage
- **Integration Paths**: 90% coverage
- **Error Conditions**: 80% coverage

### **Performance Baselines**
- **Compilation**: < 30 seconds
- **Core Tests**: < 2 minutes
- **GPU Tests**: < 5 minutes
- **Full Suite**: < 10 minutes

---

## **Ready for Production Testing!** 🚀

The SpeedBitCrackV3 test suite provides comprehensive validation of all ~28,000 lines of code, ensuring mathematical correctness, GPU acceleration, and production readiness. Use the appropriate runner based on your needs and environment.