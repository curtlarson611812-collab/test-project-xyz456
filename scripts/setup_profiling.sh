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
echo "Usage examples:"
echo "  CUDA profiling: ncu --set full ./target/release/speedbitcrack --features cudarc"
echo "  Vulkan capture: renderdoccmd capture -- ./target/release/speedbitcrack --features vulkan"
echo "  Benchmarks: cargo bench --features cudarc"
echo ""
echo "For detailed profiling guide, see docs/profiling.md"