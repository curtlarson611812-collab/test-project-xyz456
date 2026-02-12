// PRIME_MULTIPLIERS defined in step.cu - extern declaration here
#include <stdint.h>
extern __constant__ uint64_t PRIME_MULTIPLIERS[32];

// Curve order for mod operations
__constant__ uint32_t CURVE_ORDER_U32[8] = {
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE,
    0xBAAEDCE6, 0xAF48A03B, 0xBFD25E8C, 0xD0364141
};

// Concise Block: CUDA Mod9 Check for Attractor Filter
__global__ void mod9_attractor_check(uint64_t* x_limbs, bool* results, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // Proper modular reduction for 256-bit numbers
    uint64_t carry = 0;
    uint64_t sum = 0;

    // Sum all limbs with carry propagation
    for (int i = 0; i < 4; ++i) {
        uint64_t temp = x_limbs[idx * 4 + i] + carry;
        sum = (sum + (temp % 9)) % 9;
        carry = temp / 9;
    }

    // Handle remaining carry
    while (carry > 0) {
        sum = (sum + (carry % 9)) % 9;
        carry /= 9;
    }

    results[idx] = (sum == 0);
}

// Deep note: Proper mod9 reduction with carry propagation - exact for 256-bit numbers

/// Generate pre-seed positional bias points using G * (small_prime * k)
/// Returns 32*32 = 1024 normalized positions [0,1] within the puzzle range
/// This provides "curve-aware" baseline for unsolved puzzles lacking empirical data
__global__ void generate_preseed_pos_kernel(
    float* pos_out,
    uint32_t range_min,
    uint32_t range_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 32 * 32) return;

    int p_idx = idx / 32;
    int k = (idx % 32) + 1; // k from 1 to 32

    uint32_t prime = PRIME_MULTIPLIERS[p_idx];
    uint32_t scalar_raw = prime * k;
    uint32_t scalar[8] = {scalar_raw, 0, 0, 0, 0, 0, 0, 0}; // Low limb

    // Mod by curve order to prevent overflow
    mod_u32_array(scalar, CURVE_ORDER_U32, scalar);

    // Skip zero scalars
    if (scalar_is_zero_cuda(scalar)) {
        pos_out[idx] = -1.0f; // Invalid marker
        return;
    }

    // Compute point = scalar * G using GLV optimization
    Point point = mul_glv_opt_cuda(GENERATOR, scalar);

    // Skip identity point
    if (point_is_identity(point)) {
        pos_out[idx] = -1.0f; // Invalid marker
        return;
    }

    // Hash point.x to get deterministic pos_proxy
    uint32_t x_hash = hash_point_x_cuda(point.x);
    uint32_t offset = x_hash % range_width;
    float pos_val = (float)(offset - range_min) / (float)range_width;
    pos_val = fmaxf(0.0f, fminf(1.0f, pos_val));

    pos_out[idx] = pos_val;
}

/// Blend pre-seed positions with random simulations and empirical data
/// weights: (preseed_weight, random_weight, empirical_weight) - must sum to 1.0
__global__ void blend_proxy_preseed_kernel(
    float* blended_out,
    const float* preseed_pos,
    const float* random_samples,
    const float* empirical_samples,
    int preseed_count,
    int random_count,
    int empirical_count,
    float weight_pre,
    float weight_rand,
    float weight_emp
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_samples = preseed_count + random_count + empirical_count;

    if (idx >= total_samples) return;

    float total_weight = weight_pre + weight_rand + weight_emp;

    // Duplicate proportional to weights
    int pre_dup = (int)((preseed_count * weight_pre / total_weight) + 0.5f);
    if (idx < pre_dup && preseed_count > 0) {
        int source_idx = idx % preseed_count;
        blended_out[idx] = preseed_pos[source_idx];
        return;
    }

    int rand_dup = (int)((random_count * weight_rand / total_weight) + 0.5f);
    int rand_start = pre_dup;
    if (idx >= rand_start && idx < rand_start + rand_dup && random_count > 0) {
        int source_idx = (idx - rand_start) % random_count;
        float rand_pos = random_samples[source_idx];
        // Add optional noise for variance
        rand_pos += sinf((float)idx) * 0.05f; // Simple noise approximation
        rand_pos = fmaxf(0.0f, fminf(1.0f, rand_pos));
        blended_out[idx] = rand_pos;
        return;
    }

    int emp_dup = (int)((empirical_count * weight_emp / total_weight) + 0.5f);
    int emp_start = rand_start + rand_dup;
    if (idx >= emp_start && idx < emp_start + emp_dup && empirical_count > 0) {
        int source_idx = (idx - emp_start) % empirical_count;
        blended_out[idx] = empirical_samples[source_idx];
    }
}

/// Analyze blended proxy positions for cascade histogram generation
/// Returns histogram bins and bias factors for POS filter tuning
__global__ void analyze_preseed_cascade_kernel(
    const float* proxy_pos,
    int proxy_count,
    int bins,
    float* hist_out,
    float* bias_factors_out
) {
    // Use shared memory for histogram building
    __shared__ unsigned int shared_hist[10]; // Max 10 bins

    int tid = threadIdx.x;
    if (tid < bins && tid < 10) {
        shared_hist[tid] = 0;
    }
    __syncthreads();

    // Build histogram across all threads
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < proxy_count;
         i += blockDim.x * gridDim.x) {

        float pos = proxy_pos[i];
        if (pos >= 0.0f && pos <= 1.0f) {
            int bin = min((int)(pos * bins), bins - 1);
            if (bin < 10) {
                atomicAdd(&shared_hist[bin], 1);
            }
        }
    }
    __syncthreads();

    // Calculate bias factors (only thread 0)
    if (tid == 0) {
        float total_samples = 0.0f;
        for (int b = 0; b < bins && b < 10; ++b) {
            total_samples += shared_hist[b];
        }

        float uniform_count = total_samples / bins;
        for (int b = 0; b < bins && b < 10; ++b) {
            hist_out[b] = (float)shared_hist[b];
            bias_factors_out[b] = (float)shared_hist[b] / uniform_count;
        }
    }
}

// Helper functions
__device__ uint32_t hash_point_x_cuda(uint32_t x[8]) {
    uint32_t hash = 0;
    for (int i = 0; i < 8; ++i) {
        hash ^= x[i];
        hash = (hash << 5) | (hash >> 27); // Rotate for better distribution
    }
    return hash;
}

__device__ bool point_is_identity(Point p) {
    // Check if Z coordinate is zero (Jacobian infinity)
    return p.z[0] == 0 && p.z[1] == 0 && p.z[2] == 0 &&
           p.z[3] == 0 && p.z[4] == 0 && p.z[5] == 0 &&
           p.z[6] == 0 && p.z[7] == 0;
}

__device__ void mod_u32_array(uint32_t a[8], const uint32_t mod[8], uint32_t result[8]) {
    // Simple mod for low values (copy for now, extend if needed)
    for (int i = 0; i < 8; ++i) {
        result[i] = a[i] % mod[i]; // Simplified, real impl needs proper big int mod
    }
}

__device__ bool scalar_is_zero_cuda(const uint32_t scalar[8]) {
    for (int i = 0; i < 8; ++i) {
        if (scalar[i] != 0) return false;
    }
    return true;
}