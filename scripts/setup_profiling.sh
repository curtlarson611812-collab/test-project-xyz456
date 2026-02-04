#!/bin/bash
# Setup script for GPU profiling tools
# Installs Nsight Compute (CUDA) and RenderDoc (Vulkan) on Ubuntu/Debian

set -e

echo "Setting up GPU profiling tools for SpeedBitCrackV3..."

# Update package lists
sudo apt update

# Install CUDA toolkit and Nsight Compute
echo "Installing NVIDIA CUDA toolkit and Nsight Compute..."
sudo apt install -y nvidia-cuda-toolkit nvidia-nsight-compute-2025.1

# Install RenderDoc for Vulkan profiling
echo "Installing RenderDoc for Vulkan profiling..."
sudo apt install -y renderdoc

# Verify installations
echo "Verifying installations..."
ncu --version
renderdoccmd --version

echo "GPU profiling tools setup complete!"
echo ""
# Chunk: Comprehensive Nsight Compute Metrics Collection (setup_profiling.sh)
if [ "$NVIDIA_COMPUTE" = "1" ]; then
    echo "Running comprehensive Nsight Compute profiling..."

    # Category 1: Occupancy and Utilization Metrics
    echo "Collecting occupancy metrics..."
    ncu --metrics sm_efficiency,achieved_occupancy,warp_execution_efficiency \
        --csv -o occ_profile_$(date +%s).csv --target-processes all \
        cargo run -- --puzzle=32 --gpu > /dev/null 2>&1

    # Category 2: Memory Hierarchy Metrics
    echo "Collecting memory metrics..."
    ncu --metrics l1tex__t_bytes_hit_rate,l2tex__t_bytes_hit_rate,dram__bytes_read.sum.pct_of_peak_sustained_active,sm__sass_average_data_bytes_per_sector_mem_global_op_ld \
        --csv -o mem_profile_$(date +%s).csv --target-processes all \
        cargo run -- --puzzle=66 --gpu > /dev/null 2>&1

    # Category 3: Compute Throughput Metrics
    echo "Collecting compute metrics..."
    ncu --metrics sm__pipe_alu_cycles_active.average.pct_of_peak_sustained_active,sm__inst_executed.avg.pct_of_peak_sustained_active,warp_nonpred_execution_efficiency \
        --csv -o compute_profile_$(date +%s).csv --target-processes all \
        cargo run -- --puzzle=32 --gpu > /dev/null 2>&1

    # Category 4: Launch Configuration Metrics
    echo "Collecting launch config metrics..."
    ncu --metrics launched_blocks,launched_threads,register_usage \
        --csv -o launch_profile_$(date +%s).csv --target-processes all \
        cargo run -- --puzzle=66 --gpu > /dev/null 2>&1

    # Category 5: Targeted Nsight Rule Sets for ECDLP Optimization
    echo "Collecting targeted rule sets and advanced metrics..."

    # Run specific rule categories most relevant to ECDLP workloads
    ncu --rules LaunchConfig,MemoryWorkloadAnalysis,Scheduler,InstructionMix \
        --set full \
        --csv -o rules_profile_$(date +%s).csv --target-processes all \
        --print-rule-details \
        cargo run -- --puzzle=32 --gpu > rules_report.ncu-rep 2>/dev/null

    # Additional targeted profiling for specific bottlenecks
    echo "Collecting additional targeted metrics..."
    ncu --rules Occupancy,SassAnalysis \
        --csv -o occupancy_profile_$(date +%s).csv --target-processes all \
        cargo run -- --puzzle=32 --gpu > occupancy_report.ncu-rep 2>/dev/null

    # Parse all metrics into comprehensive JSON
    python3 - <<EOF
import csv, json, glob, re, os

def parse_csv_profile(filename, prefix=""):
    metrics = {}
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                current_kernel = None
                for row in reader:
                    if row and 'Kernel Name' in row[0]:
                        current_kernel = row[1]
                        if current_kernel not in metrics:
                            metrics[current_kernel] = {}
                    elif current_kernel and len(row) >= 2:
                        metric_name = row[0].strip()
                        metric_value = row[1].strip()
                        if prefix:
                            metric_name = f"{prefix}_{metric_name}"
                        metrics[current_kernel][metric_name] = metric_value
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
    return metrics

# Collect all metrics
all_metrics = {}

# Parse individual CSV files
csv_files = glob.glob("*profile_*.csv")
for csv_file in csv_files:
    if "occ_profile" in csv_file:
        all_metrics.update(parse_csv_profile(csv_file, "occ"))
    elif "mem_profile" in csv_file:
        all_metrics.update(parse_csv_profile(csv_file, "mem"))
    elif "compute_profile" in csv_file:
        all_metrics.update(parse_csv_profile(csv_file, "compute"))
    elif "launch_profile" in csv_file:
        all_metrics.update(parse_csv_profile(csv_file, "launch"))
    else:
        all_metrics.update(parse_csv_profile(csv_file))

    # Parse rules report with detailed analysis and color scoring
    rules_analysis = {}
    if os.path.exists('rules_report.ncu-rep'):
        with open('rules_report.ncu-rep', 'r') as f:
            content = f.read()

            # Extract rule results with performance scores
            rule_results = re.findall(r'Rule "(.*?)" (Passed|Failed|Warning)(?:\s+\((.*?)\))?', content)
            for rule_name, status, score_info in rule_results:
                # Extract numerical scores from parentheses
                score = 0.0
                if score_info:
                    score_match = re.search(r'(\d+\.?\d*)%', score_info)
                    if score_match:
                        score = float(score_match.group(1))

                # Color-code based on performance
                if status == "Passed" or score > 80:
                    color_indicator = "🟢"  # Green for good performance
                elif status == "Warning" or (score > 60 and score <= 80):
                    color_indicator = "🟡"  # Yellow for moderate issues
                else:
                    color_indicator = "🔴"  # Red for significant issues

                rules_analysis[rule_name] = {
                    "status": status,
                    "score": score,
                    "color": color_indicator
                }

            # Extract detailed suggestions
            suggestions = re.findall(r'Rule "(.*?)".*?Suggestion: (.*?)(?:\n\n|\n\S)', content, re.DOTALL)
            for rule_name, suggestion in suggestions:
                if rule_name in rules_analysis:
                    rules_analysis[rule_name]["suggestion"] = suggestion.strip()

    all_metrics['rules_analysis'] = rules_analysis

# Write comprehensive metrics
with open('ci_metrics.json', 'w') as j:
    json.dump(all_metrics, j, indent=2)

# Generate optimization recommendations
recommendations = []

# Occupancy recommendations
rho_kernel_metrics = all_metrics.get('rho_kernel', {})
if 'occ_sm_efficiency' in rho_kernel_metrics:
    sm_eff = float(rho_kernel_metrics['occ_sm_efficiency'].rstrip('%')) / 100
    if sm_eff < 0.7:
        recommendations.append("Low SM efficiency (<70%) - consider reducing register usage or unrolling in rho_kernel")

if 'occ_achieved_occupancy' in rho_kernel_metrics:
    occ = float(rho_kernel_metrics['occ_achieved_occupancy'].rstrip('%')) / 100
    if occ < 0.6:
        recommendations.append("Low occupancy (<60%) - reduce block size or increase parallelism")

# Memory recommendations with CUDA optimization suggestions
if 'mem_l2tex__t_bytes_hit_rate' in rho_kernel_metrics:
    l2_hit = float(rho_kernel_metrics['mem_l2tex__t_bytes_hit_rate'].rstrip('%')) / 100
    if l2_hit < 0.7:
        recommendations.append("Low L2 cache hit rate (<70%) - convert AoS to SoA layout in rho_kernel.cu for better coalescing")

if 'mem_dram__bytes_read.sum.pct_of_peak_sustained_active' in rho_kernel_metrics:
    dram_pct = float(rho_kernel_metrics['mem_dram__bytes_read.sum.pct_of_peak_sustained_active'].rstrip('%')) / 100
    if dram_pct > 0.8:
        recommendations.append("High DRAM utilization (>80%) - memory bandwidth bound, implement SoA coalescing and shared memory for bias tables")

# Compute recommendations
if 'compute_sm__pipe_alu_cycles_active.average.pct_of_peak_sustained_active' in rho_kernel_metrics:
    alu_util = float(rho_kernel_metrics['compute_sm__pipe_alu_cycles_active.average.pct_of_peak_sustained_active'].rstrip('%')) / 100
    if alu_util < 0.8:
        recommendations.append("Low ALU utilization (<80%) - fuse Barrett reduction operations in bias_check_kernel.cu")

if 'compute_warp_nonpred_execution_efficiency' in rho_kernel_metrics:
    warp_eff = float(rho_kernel_metrics['compute_warp_nonpred_execution_efficiency'].rstrip('%')) / 100
    if warp_eff < 0.9:
        recommendations.append("Low warp execution efficiency (<90%) - use subgroup operations for bias residue calculations")

# Launch config recommendations
if 'launch_register_usage' in rho_kernel_metrics:
    reg_usage = int(rho_kernel_metrics['launch_register_usage'])
    if reg_usage > 64:
        recommendations.append(f"High register usage ({reg_usage} > 64) - reduce local variables and use shared memory for constants")

# Rule-based recommendations
rules_analysis = all_metrics.get('rules_analysis', {})
if 'Low Coalescing' in rules_analysis and rules_analysis['Low Coalescing'].get('status') == 'Failed':
    recommendations.append("Low Coalescing detected - implement SoA layout: separate arrays for x_limbs, y_limbs, dist_limbs")

if 'Bank Conflicts' in rules_analysis and rules_analysis['Bank Conflicts'].get('status') == 'Failed':
    recommendations.append("Shared memory bank conflicts - optimize bias_table access pattern in bias_check_kernel.cu")

if 'Low Compute Utilization' in rules_analysis and rules_analysis['Low Compute Utilization'].get('status') == 'Failed':
    recommendations.append("Low compute utilization - fuse multiple operations per thread in rho_kernel.cu")

# CUDA-specific memory optimization recommendations
if 'mem_l1tex__t_bytes_hit_rate' in rho_kernel_metrics:
    l1_hit = float(rho_kernel_metrics['mem_l1tex__t_bytes_hit_rate'].rstrip('%')) / 100
    if l1_hit < 0.8:
        recommendations.append("Low L1 cache utilization (<80%) - set CUDA cache config to prefer L1 for local variables")

# Add rule suggestions to recommendations
for rule_name, rule_data in rules_analysis.items():
    if 'suggestion' in rule_data and rule_data.get('status') in ['Failed', 'Warning']:
        recommendations.append(f"{rule_name}: {rule_data['suggestion']}")

all_metrics['optimization_recommendations'] = recommendations

# Rewrite with recommendations
with open('ci_metrics.json', 'w') as j:
    json.dump(all_metrics, j, indent=2)

print(f"Collected metrics for {len(all_metrics)-2} kernels")  # -2 for rules_suggestions and optimization_recommendations
print(f"Generated {len(recommendations)} optimization recommendations")

EOF

    echo "Nsight Compute profiling complete. Results saved to ci_metrics.json"
    echo "Run 'cat ci_metrics.json' to view detailed metrics and recommendations"
else
    cargo criterion "$@"
fi
# Usage: NVIDIA_COMPUTE=1 ./setup_profiling.sh puzzle66_crack

echo "Usage examples:"
echo "  CUDA profiling: ncu --set full ./target/release/speedbitcrack --features cudarc"
echo "  Vulkan capture: renderdoccmd capture -- ./target/release/speedbitcrack --features vulkan"
echo "  Benchmarks: cargo bench --features cudarc"
echo "  Automated profiling: NVIDIA_COMPUTE=1 ./setup_profiling.sh bench_name"
echo ""
echo "For detailed profiling guide, see docs/profiling.md"