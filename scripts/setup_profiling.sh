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
# Chunk: Nsight Auto Profile (setup_profiling.sh)
# Dependencies: ncu installed, CUDA_HOME set
if [ "$NVIDIA_COMPUTE" = "1" ]; then
    ncu --set full --csv -o profile_$(date +%s).csv --target-processes all cargo criterion "$@"
    # Parse CSV: grep sm_efficiency profile*.csv > metrics.log
    python -c "import csv; with open('profile.csv') as f: r=csv.reader(f); print({row[0]:row[1] for row in r if 'sm_efficiency' in row[0]})" >> ci_metrics.json
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