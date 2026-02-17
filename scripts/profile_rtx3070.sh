#!/bin/bash

# SpeedBitCrackV3 RTX3070MaxQ Performance Profiling Script
# Focus: Hybrid CUDA/Vulkan optimization for laptop deployment

set -e

echo "🚀 SpeedBitCrackV3 RTX3070MaxQ Performance Profiling"
echo "=================================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration for RTX3070MaxQ
GPU_MEMORY_MB=8192
GPU_NAME="RTX3070MaxQ"
TARGET_OPS_PER_SEC=2500000000  # 2.5B ops/sec target

# Function to check GPU status
check_gpu_status() {
    echo -e "${BLUE}GPU Status Check:${NC}"

    # Check CUDA availability
    if command -v nvidia-smi &> /dev/null; then
        echo "CUDA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total,memory.used,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits
        echo ""
    else
        echo -e "${YELLOW}Warning: nvidia-smi not found${NC}"
    fi

    # Check Vulkan availability
    if command -v vulkaninfo &> /dev/null; then
        echo "Vulkan GPU detected"
    else
        echo -e "${YELLOW}Warning: vulkaninfo not found${NC}"
    fi
}

# Function to run basic compilation check
check_compilation() {
    echo -e "${BLUE}Compilation Check:${NC}"

    # Build with CUDA
    echo "Building with CUDA support..."
    if cargo build --release --features cuda --quiet; then
        echo -e "${GREEN}✅ CUDA build successful${NC}"
        CUDA_AVAILABLE=true
    else
        echo -e "${RED}❌ CUDA build failed${NC}"
        CUDA_AVAILABLE=false
    fi

    # Build with Vulkan
    echo "Building with Vulkan support..."
    if cargo build --release --features wgpu --quiet; then
        echo -e "${GREEN}✅ Vulkan build successful${NC}"
        VULKAN_AVAILABLE=true
    else
        echo -e "${RED}❌ Vulkan build failed${NC}"
        VULKAN_AVAILABLE=false
    fi

    # Build hybrid
    echo "Building hybrid CUDA+Vulkan..."
    if cargo build --release --features hybrid --quiet; then
        echo -e "${GREEN}✅ Hybrid build successful${NC}"
        HYBRID_AVAILABLE=true
    else
        echo -e "${RED}❌ Hybrid build failed${NC}"
        HYBRID_AVAILABLE=false
    fi
}

# Function to run basic functionality test
run_basic_test() {
    echo -e "${BLUE}Basic Functionality Test:${NC}"

    if [ "$HYBRID_AVAILABLE" = true ]; then
        echo "Running basic hybrid test..."
        timeout 30s cargo run --release --features hybrid -- --help > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Basic hybrid execution successful${NC}"
        else
            echo -e "${RED}❌ Basic hybrid execution failed${NC}"
        fi
    fi
}

# Function to run performance benchmark
run_performance_benchmark() {
    echo -e "${BLUE}Performance Benchmark:${NC}"

    # Create test config for benchmarking
    cat > benchmark_config.json << EOF
{
    "puzzle_mode": true,
    "target_puzzle": 66,
    "herd_size": 1024,
    "dp_bits": 24,
    "steps_per_batch": 10000,
    "max_steps": 100000,
    "bias_mode": "magic9",
    "enable_gpu": true,
    "enable_cuda": true,
    "enable_vulkan": true,
    "benchmark_mode": true
}
EOF

    echo "Running performance benchmark (30 second sample)..."
    START_TIME=$(date +%s)

    # Run benchmark with timeout
    timeout 30s cargo run --release --features hybrid -- --config benchmark_config.json > benchmark_output.log 2>&1 &
    BENCH_PID=$!

    # Monitor GPU usage during benchmark
    if command -v nvidia-smi &> /dev/null; then
        echo "Monitoring GPU usage..."
        nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.free --format=csv,noheader,nounits -l 5 > gpu_usage.log &
        GPU_MONITOR_PID=$!
    fi

    # Wait for benchmark to complete
    wait $BENCH_PID 2>/dev/null
    EXIT_CODE=$?

    # Stop GPU monitoring
    if [ ! -z "$GPU_MONITOR_PID" ]; then
        kill $GPU_MONITOR_PID 2>/dev/null
    fi

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    # Analyze results
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}✅ Benchmark completed successfully in ${DURATION}s${NC}"

        # Extract performance metrics
        if [ -f benchmark_output.log ]; then
            OPS_PER_SEC=$(grep -o "ops/sec: [0-9,]*" benchmark_output.log | tail -1 | grep -o "[0-9,]*" | tr -d ',')
            if [ ! -z "$OPS_PER_SEC" ]; then
                echo "Performance: ${OPS_PER_SEC} ops/sec"
                EFFICIENCY=$(echo "scale=2; $OPS_PER_SEC * 100 / $TARGET_OPS_PER_SEC" | bc 2>/dev/null || echo "0")
                echo "Target efficiency: ${EFFICIENCY}% of RTX5090 target"

                if (( $(echo "$EFFICIENCY > 80" | bc -l 2>/dev/null || echo "0") )); then
                    echo -e "${GREEN}🎯 Excellent performance!${NC}"
                elif (( $(echo "$EFFICIENCY > 50" | bc -l 2>/dev/null || echo "0") )); then
                    echo -e "${YELLOW}⚡ Good performance, optimization opportunities available${NC}"
                else
                    echo -e "${RED}📉 Performance needs optimization${NC}"
                fi
            fi
        fi

        # GPU usage analysis
        if [ -f gpu_usage.log ]; then
            echo "GPU Usage Summary:"
            tail -5 gpu_usage.log | while IFS=, read timestamp name gpu_util mem_util mem_used mem_free; do
                echo "  GPU: ${gpu_util}%  Memory: ${mem_util}% (${mem_used}MB used)"
            done
        fi

    else
        echo -e "${RED}❌ Benchmark failed${NC}"
        if [ -f benchmark_output.log ]; then
            echo "Last 10 lines of output:"
            tail -10 benchmark_output.log
        fi
    fi

    # Cleanup
    rm -f benchmark_config.json benchmark_output.log gpu_usage.log
}

# Function to run parity validation
run_parity_validation() {
    echo -e "${BLUE}Parity Validation:${NC}"

    echo "Running CPU vs GPU parity tests..."
    if cargo test --release --features hybrid gpu_parity --quiet; then
        echo -e "${GREEN}✅ Parity tests passed${NC}"
    else
        echo -e "${RED}❌ Parity tests failed${NC}"
        echo "Running verbose parity test..."
        cargo test --release --features hybrid gpu_parity -- --nocapture | tail -20
    fi
}

# Function to generate optimization report
generate_report() {
    echo -e "${BLUE}Optimization Report:${NC}"

    REPORT_FILE="rtx3070_optimization_$(date +%Y%m%d_%H%M%S).txt"

    cat > "$REPORT_FILE" << EOF
SpeedBitCrackV3 RTX3070MaxQ Optimization Report
===============================================

Generated: $(date)
GPU: $GPU_NAME
Memory: ${GPU_MEMORY_MB}MB

System Information:
$(uname -a)

CUDA Version:
$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null || echo "N/A")

Vulkan Version:
$(vulkaninfo --summary 2>/dev/null | grep "Vulkan Instance" || echo "N/A")

Build Status:
CUDA Available: $CUDA_AVAILABLE
Vulkan Available: $VULKAN_AVAILABLE
Hybrid Available: $HYBRID_AVAILABLE

Performance Target: ${TARGET_OPS_PER_SEC} ops/sec
EOF

    echo "Report saved to: $REPORT_FILE"

    # Recommendations based on current status
    echo ""
    echo "Optimization Recommendations:"
    if [ "$HYBRID_AVAILABLE" = true ]; then
        echo "✅ Hybrid mode available - focus on memory sharing optimization"
        echo "✅ Implement zero-copy buffers between CUDA and Vulkan"
        echo "✅ Profile kernel launch overhead and batch sizing"
    else
        echo "❌ Fix compilation issues before performance optimization"
    fi

    if [ "$CUDA_AVAILABLE" = true ]; then
        echo "✅ CUDA backend functional - optimize kernel occupancy"
        echo "✅ Focus on shared memory usage and warp divergence"
    fi

    if [ "$VULKAN_AVAILABLE" = true ]; then
        echo "✅ Vulkan backend functional - optimize compute shader workgroups"
        echo "✅ Implement proper WGSL shader variants for different operations"
    fi
}

# Main execution
main() {
    check_gpu_status
    echo ""

    check_compilation
    echo ""

    run_basic_test
    echo ""

    if [ "$HYBRID_AVAILABLE" = true ]; then
        run_performance_benchmark
        echo ""

        run_parity_validation
        echo ""
    fi

    generate_report

    echo ""
    echo -e "${GREEN}🚀 RTX3070MaxQ profiling complete!${NC}"
    echo "Next steps:"
    echo "1. Review optimization report"
    echo "2. Focus on identified performance bottlenecks"
    echo "3. Implement zero-copy memory sharing"
    echo "4. Scale to vast.ai RTX5090 when ready"
}

main "$@"