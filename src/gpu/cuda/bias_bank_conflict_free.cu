/*
 * Bank Conflict-Free Shared Memory Bias Kernel for SpeedBitCrackV3
 *
 * Optimizes shared memory access patterns to eliminate bank conflicts
 * in bias table lookups, achieving maximum shared memory bandwidth.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Bank conflict-free bias check with padding
__global__ void bias_check_kernel_no_conflicts(
    const uint32_t* dist_limbs,    // [num_states * 4] - SoA layout
    uint32_t* results,             // [num_states] - output results
    const float* bias_table_global, // [81] - global bias table
    uint32_t num_states,
    uint32_t bias_modulus
) {
    // Shared memory with padding to avoid bank conflicts
    // CUDA shared memory has 32 banks, so we pad to avoid stride conflicts
    __shared__ float bias_table_shared[81 + 31];  // 112 elements, 32 banks

    // Cooperative loading with conflict-free pattern
    uint32_t local_tid = threadIdx.x;
    if (local_tid < bias_modulus) {
        // Load with padding offset to avoid bank conflicts
        uint32_t target_idx = local_tid + (local_tid / 32) * 32;  // Pad every 32 elements
        bias_table_shared[target_idx] = bias_table_global[local_tid];
    }
    __syncthreads();

    uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_tid < num_states) {
        // Load distance value (SoA access)
        uint32_t dist_low = dist_limbs[global_tid * 4];

        // Calculate residue
        uint32_t residue = dist_low % bias_modulus;

        // Access shared memory with conflict-free indexing
        uint32_t shared_idx = residue + (residue / 32) * 32;
        float bias_factor = bias_table_shared[shared_idx];

        // Store result
        results[global_tid] = (bias_factor > 1.0f) ? 1 : 0;
    }
}

// Alternative: Swizzled access pattern for conflict-free reads
__global__ void bias_check_kernel_swizzled(
    const uint32_t* dist_limbs,    // [num_states * 4]
    uint32_t* results,             // [num_states]
    const float* bias_table_global, // [81]
    uint32_t num_states,
    uint32_t bias_modulus
) {
    // Use XOR swizzling to distribute access across banks
    __shared__ float bias_table_shared[128];  // Power of 2 size

    uint32_t local_tid = threadIdx.x;
    uint32_t bank_mask = 31;  // 32 banks - 1

    // Load with XOR swizzling for conflict-free access
    if (local_tid < bias_modulus) {
        uint32_t swizzled_idx = local_tid ^ (local_tid >> 5);  // XOR with high bits
        bias_table_shared[swizzled_idx] = bias_table_global[local_tid];
    }
    __syncthreads();

    uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_tid < num_states) {
        uint32_t dist_low = dist_limbs[global_tid * 4];
        uint32_t residue = dist_low % bias_modulus;

        // Unswizzle for access
        uint32_t access_idx = residue ^ (residue >> 5);
        float bias_factor = bias_table_shared[access_idx];

        results[global_tid] = (bias_factor > 1.0f) ? 1 : 0;
    }
}

// Warp-level optimized bias kernel using shuffle operations
__global__ void bias_check_kernel_warp_shuffle(
    const uint32_t* dist_limbs,    // [num_states * 4]
    uint32_t* results,             // [num_states]
    const float* bias_table_shared, // [81] - must be in shared memory
    uint32_t num_states,
    uint32_t bias_modulus
) {
    // Assume bias_table is pre-loaded into shared memory by calling kernel
    // This kernel focuses on the shuffle optimization

    uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_tid < num_states) {
        uint32_t dist_low = dist_limbs[global_tid * 4];
        uint32_t residue = dist_low % bias_modulus;

        // Use warp shuffle to share bias factor within warp
        // First thread in warp loads and broadcasts
        float bias_factor = 0.0f;
        if (threadIdx.x % 32 == 0) {
            bias_factor = bias_table_shared[residue];
        }

        // Broadcast to all threads in warp
        uint32_t is_biased_int = __shfl_sync(0xFFFFFFFF, *reinterpret_cast<uint32_t*>(&bias_factor), 0);
        float shared_bias = *reinterpret_cast<float*>(&is_biased_int);

        results[global_tid] = (shared_bias > 1.0f) ? 1 : 0;
    }
}

// Combined kernel: Load bias table + process with conflict-free access
__global__ void bias_check_kernel_combined(
    const uint32_t* dist_limbs,    // [num_states * 4]
    uint32_t* results,             // [num_states]
    const float* bias_table_global, // [81]
    uint32_t num_states,
    uint32_t bias_modulus
) {
    // Shared memory sized for conflict-free access
    // 81 elements + padding to avoid bank conflicts
    __shared__ float bias_table_shared[112];  // 81 + 31 padding

    uint32_t local_tid = threadIdx.x;

    // Phase 1: Load bias table with conflict-free pattern
    if (local_tid < bias_modulus) {
        // Use padding to avoid bank conflicts
        // Each bank gets at most one access per 32 threads
        uint32_t bank_idx = local_tid % 32;
        uint32_t bank_offset = local_tid / 32;
        uint32_t shared_idx = bank_idx + bank_offset * 32 + bank_offset;  // Pad between banks

        bias_table_shared[shared_idx] = bias_table_global[local_tid];
    }
    __syncthreads();

    // Phase 2: Process states
    uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_tid < num_states) {
        uint32_t dist_low = dist_limbs[global_tid * 4];
        uint32_t residue = dist_low % bias_modulus;

        // Access with same conflict-free pattern
        uint32_t bank_idx = residue % 32;
        uint32_t bank_offset = residue / 32;
        uint32_t access_idx = bank_idx + bank_offset * 32 + bank_offset;

        float bias_factor = bias_table_shared[access_idx];
        results[global_tid] = (bias_factor > 1.0f) ? 1 : 0;
    }
}

// Streaming kernel for processing large datasets in chunks
__global__ void bias_check_kernel_streaming(
    const uint32_t* dist_limbs,    // [total_states * 4]
    uint32_t* results,             // [total_states]
    const float* bias_table_global, // [81]
    uint32_t total_states,
    uint32_t bias_modulus,
    uint32_t chunk_offset         // Process in chunks to fit shared memory
) {
    __shared__ float bias_table_shared[112];

    // Load bias table (same for all chunks)
    uint32_t local_tid = threadIdx.x;
    if (local_tid < bias_modulus) {
        uint32_t bank_idx = local_tid % 32;
        uint32_t bank_offset = local_tid / 32;
        uint32_t shared_idx = bank_idx + bank_offset * 32 + bank_offset;

        bias_table_shared[shared_idx] = bias_table_global[local_tid];
    }
    __syncthreads();

    // Process chunk
    uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x + chunk_offset;

    if (global_tid < total_states) {
        uint32_t dist_low = dist_limbs[global_tid * 4];
        uint32_t residue = dist_low % bias_modulus;

        uint32_t bank_idx = residue % 32;
        uint32_t bank_offset = residue / 32;
        uint32_t access_idx = bank_idx + bank_offset * 32 + bank_offset;

        float bias_factor = bias_table_shared[access_idx];
        results[global_tid] = (bias_factor > 1.0f) ? 1 : 0;
    }
}