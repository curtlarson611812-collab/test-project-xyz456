#!/bin/bash
# GPU Profiling and Analysis Script
# Profiles SpeedBitCrackV3 performance and provides optimization suggestions

set -e

# Default values
OUTPUT_DIR="profiling_output"
KERNEL_NAME="kangaroo_step"
MIN_EFFICIENCY=70

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=== SpeedBitCrackV3 GPU Profiling ==="
echo "Output directory: $OUTPUT_DIR"

# Run Nsight Compute profiling
echo "Running Nsight Compute profiling..."
ncu --metrics sm_efficiency,shared_efficiency,mem_efficiency \
    --target-processes all \
    --print-summary per-kernel \
    --csv \
    --output "$OUTPUT_DIR/ncu_profile.csv" \
    cargo run --release -- --test-mode 2>&1 | tee "$OUTPUT_DIR/run_output.log"

# Generate detailed kernel analysis
echo "Generating kernel analysis..."
ncu --kernel-name "$KERNEL_NAME" \
    --metrics all \
    --print-summary per-kernel \
    --output "$OUTPUT_DIR/ncu_detailed.csv" \
    cargo run --release -- --test-mode 2>&1 > /dev/null

# Analyze results and provide recommendations
echo "=== Performance Analysis ===" > "$OUTPUT_DIR/analysis_report.txt"

# Check SM efficiency
SM_EFF=$(grep "sm_efficiency" "$OUTPUT_DIR/ncu_profile.csv" | tail -1 | cut -d',' -f2 | sed 's/%//')
if (( $(echo "$SM_EFF < $MIN_EFFICIENCY" | bc -l) )); then
    echo "WARNING: SM efficiency is $SM_EFF% (below $MIN_EFFICIENCY% threshold)" >> "$OUTPUT_DIR/analysis_report.txt"
    echo "RECOMMENDATION: Increase workgroup size in kangaroo.wgsl or block size in step.cu" >> "$OUTPUT_DIR/analysis_report.txt"
else
    echo "SM efficiency: $SM_EFF% (acceptable)" >> "$OUTPUT_DIR/analysis_report.txt"
fi

# Check memory efficiency
MEM_EFF=$(grep "mem_efficiency" "$OUTPUT_DIR/ncu_profile.csv" | tail -1 | cut -d',' -f2 | sed 's/%//')
if (( $(echo "$MEM_EFF < 80" | bc -l) )); then
    echo "WARNING: Memory efficiency is $MEM_EFF% (below 80% threshold)" >> "$OUTPUT_DIR/analysis_report.txt"
    echo "RECOMMENDATION: Optimize buffer access patterns, consider SOA layout" >> "$OUTPUT_DIR/analysis_report.txt"
else
    echo "Memory efficiency: $MEM_EFF% (good)" >> "$OUTPUT_DIR/analysis_report.txt"
fi

# Check shared memory efficiency
SHARED_EFF=$(grep "shared_efficiency" "$OUTPUT_DIR/ncu_profile.csv" | tail -1 | cut -d',' -f2 | sed 's/%//')
if (( $(echo "$SHARED_EFF < 50" | bc -l) )); then
    echo "WARNING: Shared memory efficiency is $SHARED_EFF% (below 50% threshold)" >> "$OUTPUT_DIR/analysis_report.txt"
    echo "RECOMMENDATION: Improve shared memory usage in kernels" >> "$OUTPUT_DIR/analysis_report.txt"
else
    echo "Shared memory efficiency: $SHARED_EFF% (acceptable)" >> "$OUTPUT_DIR/analysis_report.txt"
fi

# Extract kernel execution time
KERNEL_TIME=$(grep "$KERNEL_NAME" "$OUTPUT_DIR/ncu_profile.csv" | head -1 | cut -d',' -f4)
echo "Kernel execution time: $KERNEL_TIME ms" >> "$OUTPUT_DIR/analysis_report.txt"

# Calculate ops/sec estimate
if [[ -n "$KERNEL_TIME" ]] && (( $(echo "$KERNEL_TIME > 0" | bc -l) )); then
    OPS_PER_SEC=$(echo "scale=2; 1000 / $KERNEL_TIME * 1000000" | bc -l)
    echo "Estimated ops/sec: $OPS_PER_SEC Mops/sec" >> "$OUTPUT_DIR/analysis_report.txt"
fi

# SmallOddPrime-specific optimizations
echo "" >> "$OUTPUT_DIR/analysis_report.txt"
echo "=== SmallOddPrime Optimization Analysis ===" >> "$OUTPUT_DIR/analysis_report.txt"

# Check for SmallOddPrime kernel efficiency
if grep -q "sm_efficiency" "$OUTPUT_DIR/ncu_detailed.csv"; then
    SOP_EFF=$(grep "sm_efficiency" "$OUTPUT_DIR/ncu_detailed.csv" | grep -i "sop\|bucket\|jump" | tail -1 | cut -d',' -f2 | sed 's/%//')
    if [[ -n "$SOP_EFF" ]] && (( $(echo "$SOP_EFF < $MIN_EFFICIENCY" | bc -l) )); then
        echo "WARNING: SmallOddPrime bucket/jump efficiency is $SOP_EFF% (below threshold)" >> "$OUTPUT_DIR/analysis_report.txt"
        echo "RECOMMENDATION: Precompute PRIME_MULTIPLIERS in shared memory" >> "$OUTPUT_DIR/analysis_report.txt"
        echo "RECOMMENDATION: Optimize select_bucket mixing for better divergence" >> "$OUTPUT_DIR/analysis_report.txt"
    else
        echo "SmallOddPrime efficiency: $SOP_EFF% (acceptable)" >> "$OUTPUT_DIR/analysis_report.txt"
    fi
fi

# Check for herd size optimization
echo "RECOMMENDATION: Monitor herd convergence - if stuck on attractors, adjust bias_mod" >> "$OUTPUT_DIR/analysis_report.txt"
echo "RECOMMENDATION: Profile tame vs wild herd performance separately" >> "$OUTPUT_DIR/analysis_report.txt"

echo "Analysis complete. Check $OUTPUT_DIR/analysis_report.txt for recommendations."