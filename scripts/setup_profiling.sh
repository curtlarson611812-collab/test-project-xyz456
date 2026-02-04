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
# Chunk: Python Nsight Parse (setup_profiling.sh)
if [ "$NVIDIA_COMPUTE" = "1" ]; then
    ncu --set full --csv -o profile_$(date +%s).csv --target-processes all cargo criterion "$@"
    python3 - <<EOF
import csv, json
metrics = {}
current_kernel = None
try:
    with open('profile.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and 'Kernel Name' in row[0]:
                current_kernel = row[1]
                metrics[current_kernel] = {}
            elif current_kernel and 'sm_efficiency' in row[0]:
                metrics[current_kernel][row[0]] = row[1]
except Exception as e:
    print(f"Error: {e}")
with open('ci_metrics.json', 'w') as j:
    json.dump(metrics, j)
EOF
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