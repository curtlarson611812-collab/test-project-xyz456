#!/bin/bash
# Profile and analyze SpeedBitCrackV3 performance with Nsight Compute

set -e

echo "🚀 SpeedBitCrackV3 Nsight Compute Performance Analysis"
echo "======================================================"

# Check if Nsight Compute is available
if ! command -v ncu &> /dev/null; then
    echo "❌ Nsight Compute (ncu) not found. Please install NVIDIA Nsight Compute."
    echo "   Ubuntu/Debian: sudo apt install nvidia-nsight-compute-2025.1"
    exit 1
fi

# Build release binary
echo "📦 Building release binary..."
cargo build --release --quiet

# Run comprehensive profiling
echo "🔬 Running comprehensive profiling..."
NVIDIA_COMPUTE=1 ./scripts/setup_profiling.sh --puzzle=32 > /dev/null 2>&1

# Check if profiling succeeded
if [ ! -f "ci_metrics.json" ]; then
    echo "❌ Profiling failed - no ci_metrics.json found"
    exit 1
fi

# Display results
echo ""
echo "📊 PERFORMANCE ANALYSIS RESULTS"
echo "==============================="

# Check for CUDA-specific optimizations
echo ""
echo "🎯 CUDA MEMORY OPTIMIZATION ANALYSIS:"
echo "-------------------------------------"

if command -v jq &> /dev/null && [ -f "ci_metrics.json" ]; then
    echo "🎯 GPU KERNEL PERFORMANCE:"
    echo "-------------------------"

    # Rho kernel metrics
    if jq -e '.rho_kernel' ci_metrics.json > /dev/null 2>&1; then
        echo "Rho Kernel Metrics:"
        jq -r '.rho_kernel | to_entries[] | select(.key | startswith("occ_")) | "  \(.key): \(.value)"' ci_metrics.json 2>/dev/null || true
        jq -r '.rho_kernel | to_entries[] | select(.key | startswith("mem_")) | "  \(.key): \(.value)"' ci_metrics.json 2>/dev/null || true
        jq -r '.rho_kernel | to_entries[] | select(.key | startswith("compute_")) | "  \(.key): \(.value)"' ci_metrics.json 2>/dev/null || true
    else
        echo "  No rho_kernel metrics found"
    fi

    echo ""
    echo "💡 OPTIMIZATION RECOMMENDATIONS:"
    echo "--------------------------------"

    if jq -e '.optimization_recommendations' ci_metrics.json > /dev/null 2>&1; then
        jq -r '.optimization_recommendations[] | "  • \(. )"' ci_metrics.json 2>/dev/null || echo "  No recommendations generated"
    else
        echo "  No optimization recommendations available"
    fi

    echo ""
    echo "📈 KEY PERFORMANCE INDICATORS:"
    echo "------------------------------"

    # Calculate performance score
    sm_eff=$(jq -r '.rho_kernel."occ_sm_efficiency" | sub("%"; "") | tonumber / 100' ci_metrics.json 2>/dev/null || echo "0")
    l2_hit=$(jq -r '.rho_kernel."mem_l2tex__t_bytes_hit_rate" | sub("%"; "") | tonumber / 100' ci_metrics.json 2>/dev/null || echo "0")
    dram_util=$(jq -r '.rho_kernel."mem_dram__bytes_read.sum.pct_of_peak_sustained_active" | sub("%"; "") | tonumber / 100' ci_metrics.json 2>/dev/null || echo "0")

    if (( $(echo "$sm_eff > 0" | bc -l) )); then
        echo "  SM Efficiency: $(printf "%.1f" $(echo "$sm_eff * 100" | bc -l))%"
        echo "  L2 Cache Hit Rate: $(printf "%.1f" $(echo "$l2_hit * 100" | bc -l))%"
        echo "  DRAM Utilization: $(printf "%.1f" $(echo "$dram_util * 100" | bc -l))%"

        # Performance assessment
        score=$(echo "($sm_eff + $l2_hit + (1 - $dram_util)) / 3" | bc -l)
        if (( $(echo "$score > 0.8" | bc -l) )); then
            echo "  🟢 Overall Performance: EXCELLENT (Score: $(printf "%.2f" $score))"
        elif (( $(echo "$score > 0.6" | bc -l) )); then
            echo "  🟡 Overall Performance: GOOD (Score: $(printf "%.2f" $score))"
        else
            echo "  🔴 Overall Performance: NEEDS OPTIMIZATION (Score: $(printf "%.2f" $score))"
        fi
    fi

    # CUDA Memory Optimization Assessment
    echo ""
    echo "🧠 CUDA MEMORY OPTIMIZATION ASSESSMENT:"
    echo "----------------------------------------"

    # Check memory coalescing
    sector_eff=$(jq -r '.rho_kernel."mem_sm__sass_average_data_bytes_per_sector_mem_global_op_ld" // 0' ci_metrics.json 2>/dev/null || echo "0")
    if (( $(echo "$sector_eff < 3.5" | bc -l 2>/dev/null || echo "1") )); then
        echo "  🚨 MEMORY COALESCING: Poor (avg $sector_eff bytes/sector)"
        echo "     → Implement SoA layout in rho_kernel.cu"
        echo "     → Separate x_limbs[], y_limbs[], dist_limbs[] arrays"
    else
        echo "  ✅ MEMORY COALESCING: Good (avg $sector_eff bytes/sector)"
    fi

    # Check shared memory utilization
    bank_conflicts=$(jq -r '.rho_kernel."mem_sm__sass_average_bank_conflicts_pipe_lsu_mem_shared_op_ld" // 0' ci_metrics.json 2>/dev/null || echo "0")
    if (( $(echo "$bank_conflicts > 0" | bc -l 2>/dev/null || echo "0") )); then
        echo "  🚨 SHARED MEMORY: Bank conflicts detected ($bank_conflicts avg)"
        echo "     → Optimize bias_table access pattern in bias_check_kernel.cu"
        echo "     → Use stride-1 access or padding"
    else
        echo "  ✅ SHARED MEMORY: No bank conflicts"
    fi

    # Check L1 cache utilization
    l1_hit=$(jq -r '.rho_kernel."mem_l1tex__t_bytes_hit_rate" // 0' ci_metrics.json 2>/dev/null || echo "0")
    if (( $(echo "$l1_hit < 80" | bc -l 2>/dev/null || echo "1") )); then
        echo "  🚨 L1 CACHE: Low hit rate ($(printf "%.1f" $l1_hit)%)"
        echo "     → Set CUDA cache config to PreferL1"
        echo "     → cudaDeviceSetCacheConfig(cudaFuncCachePreferL1)"
    else
        echo "  ✅ L1 CACHE: Good hit rate ($(printf "%.1f" $l1_hit)%)"
    fi

    # CUDA Memory Optimization Analysis
    echo ""
    echo "🧠 CUDA MEMORY OPTIMIZATION ANALYSIS:"
    echo "----------------------------------------"

    # Check for CUDA-specific optimizations
    sector_eff=$(jq -r '.rho_kernel."mem_sm__sass_average_data_bytes_per_sector_mem_global_op_ld" // 0' ci_metrics.json 2>/dev/null || echo "0")
    if (( $(echo "$sector_eff < 3.5" | bc -l 2>/dev/null || echo "1") )); then
        echo "  🚨 MEMORY COALESCING: Poor (avg $sector_eff bytes/sector)"
        echo "     → Implement SoA layout in rho_kernel.cu"
        echo "     → Use shared memory for Barrett constants"
        echo "     → Consider texture memory for jump tables"
    else
        echo "  ✅ MEMORY COALESCING: Good (avg $sector_eff bytes/sector)"
    fi

    # Bank conflict analysis
    bank_conflicts=$(jq -r '.rho_kernel."mem_sm__sass_average_bank_conflicts_pipe_lsu_mem_shared_op_ld" // 0' ci_metrics.json 2>/dev/null || echo "0")
    if (( $(echo "$bank_conflicts > 0" | bc -l 2>/dev/null || echo "0") )); then
        echo "  🚨 SHARED MEMORY: Bank conflicts detected ($bank_conflicts avg)"
        echo "     → Use padded shared memory access pattern"
        echo "     → Implement swizzled indexing in bias_check_kernel.cu"
    else
        echo "  ✅ SHARED MEMORY: No bank conflicts"
    fi

    # L1 cache optimization
    l1_hit=$(jq -r '.rho_kernel."mem_l1tex__t_bytes_hit_rate" // 0' ci_metrics.json 2>/dev/null || echo "0")
    if (( $(echo "$l1_hit < 80" | bc -l 2>/dev/null || echo "1") )); then
        echo "  🚨 L1 CACHE: Low hit rate ($(printf "%.1f" $l1_hit)%)"
        echo "     → Set CUDA cache config to PreferL1"
        echo "     → Optimize local variable usage in kernels"
    else
        echo "  ✅ L1 CACHE: Good hit rate ($(printf "%.1f" $l1_hit)%)"
    fi

    # Nsight Rules Analysis
    echo ""
    echo "📋 NSIGHT RULES ANALYSIS:"
    echo "-------------------------"

    rules_count=$(jq -r '.rules_analysis | length' ci_metrics.json 2>/dev/null || echo "0")
    if [ "$rules_count" -gt 0 ]; then
        jq -r '.rules_analysis | to_entries[] | "  \(.key): \(.value.status) - \(.value.suggestion // "No suggestion")"' ci_metrics.json 2>/dev/null || echo "  No rules analysis available"
    else
        echo "  No Nsight rules analysis found (run with --rules all)"
    fi

else
    echo "❌ jq not found. Install jq for detailed analysis:"
    echo "   sudo apt install jq"
    echo ""
    echo "Raw results available in ci_metrics.json"
fi

echo ""
echo "📁 Profiling artifacts saved:"
echo "  - ci_metrics.json: Comprehensive metrics and recommendations"
echo "  - *profile_*.csv: Raw Nsight CSV data"
echo "  - rules_report.ncu-rep: Nsight rules analysis"
echo ""
echo "📖 For detailed analysis, see docs/nsight_profiling.md"
echo ""
echo "✅ Analysis complete!"