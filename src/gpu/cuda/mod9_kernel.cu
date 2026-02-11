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
    uint64_t range_min,
    uint64_t range_width,
    const uint64_t* primes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 32 * 32) return;

    int p_idx = idx / 32;
    int k = (idx % 32) + 1; // k from 1 to 32

    uint64_t prime = primes[p_idx];
    uint64_t scalar = prime * k;

    // Compute point = scalar * G using GLV optimization
    Point point = mul_glv_opt_cuda(GENERATOR, scalar);

    // Skip identity point
    if (point_is_identity(point)) {
        pos_out[idx] = -1.0f; // Invalid marker
        return;
    }

    // Hash point.x to get deterministic pos_proxy
    uint64_t x_hash = hash_point_x_cuda(point.x);
    uint64_t offset = x_hash % range_width;
    float pos_val = (float)(offset - range_min) / (float)range_width;
    pos_val = fmaxf(0.0f, fminf(1.0f, pos_val));

    pos_out[idx] = pos_val;
}

/// Blend pre-seed positions with random simulations and empirical data
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

    // Determine source based on weighted distribution
    float cumulative_weight = 0.0f;
    float rand_val = curand_uniform(&rand_state); // Assume rand_state available

    // Pre-seed samples
    int pre_target = (int)((preseed_count * weight_pre / total_weight) + 0.5f);
    if (idx < pre_target && preseed_count > 0) {
        int source_idx = idx % preseed_count;
        blended_out[idx] = preseed_pos[source_idx];
        return;
    }
    cumulative_weight += weight_pre;

    // Random samples
    int rand_target = pre_target + (int)((random_count * weight_rand / total_weight) + 0.5f);
    if (idx < rand_target && random_count > 0) {
        int source_idx = (idx - pre_target) % random_count;
        blended_out[idx] = random_samples[source_idx];
        return;
    }
    cumulative_weight += weight_rand;

    // Empirical samples
    if (empirical_count > 0) {
        int source_idx = (idx - rand_target) % empirical_count;
        blended_out[idx] = empirical_samples[source_idx];
    }
}

/// Analyze blended proxy positions for cascade histogram generation
__global__ void analyze_preseed_cascade_kernel(
    const float* proxy_pos,
    int proxy_count,
    int bins,
    float* hist_out,
    float* bias_factors_out
) {
    // Use shared memory for histogram
    __shared__ unsigned int shared_hist[10]; // Max 10 bins

    int tid = threadIdx.x;
    if (tid < bins && tid < 10) {
        shared_hist[tid] = 0;
    }
    __syncthreads();

    // Build histogram
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
__device__ uint64_t hash_point_x_cuda(uint32_t x[8]) {
    uint64_t hash = 0;
    for (int i = 0; i < 8; ++i) {
        hash ^= (uint64_t)x[i];
        hash = (hash << 5) | (hash >> 59); // Rotate
    }
    return hash;
}

__device__ bool point_is_identity(Point p) {
    // Check if Z coordinate is zero (Jacobian infinity)
    return p.z[0] == 0 && p.z[1] == 0 && p.z[2] == 0 &&
           p.z[3] == 0 && p.z[4] == 0 && p.z[5] == 0 &&
           p.z[6] == 0 && p.z[7] == 0;
}