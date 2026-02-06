// Concise Block: CUDA Multi-Modulus Bias Check for Attractor Filter
__global__ void multi_bias_attractor_check(uint64_t* x_limbs, uint8_t* results, int batch_size, int num_moduli, uint64_t* moduli) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // For each modulus, compute x % modulus == 0
    for (int mod_idx = 0; mod_idx < num_moduli; ++mod_idx) {
        uint64_t modulus = moduli[mod_idx];
        uint64_t carry = 0;
        uint64_t sum = 0;

        // Sum all limbs with carry propagation
        for (int i = 0; i < 4; ++i) {
            uint64_t temp = x_limbs[idx * 4 + i] + carry;
            sum = (sum + (temp % modulus)) % modulus;
            carry = temp / modulus;
        }

        // Handle remaining carry
        while (carry > 0) {
            sum = (sum + (carry % modulus)) % modulus;
            carry /= modulus;
        }

        // Set bit in results if bias condition met
        if (sum == 0) {
            results[idx * num_moduli + mod_idx] = 1;
        } else {
            results[idx * num_moduli + mod_idx] = 0;
        }
    }
}

// Specialized kernel for common bias checks (mod9, mod27, mod81)
__global__ void common_bias_attractor_check(uint64_t* x_limbs, uint8_t* results, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    uint64_t x0 = x_limbs[idx * 4 + 0];
    uint64_t x1 = x_limbs[idx * 4 + 1];
    uint64_t x2 = x_limbs[idx * 4 + 2];
    uint64_t x3 = x_limbs[idx * 4 + 3];

    // Fast mod9 check (digital root)
    uint64_t mod9 = 0;
    uint64_t temp = x0; while (temp > 0) { mod9 += temp % 10; temp /= 10; } mod9 %= 9;
    temp = x1; while (temp > 0) { mod9 += temp % 10; temp /= 10; } mod9 %= 9;
    temp = x2; while (temp > 0) { mod9 += temp % 10; temp /= 10; } mod9 %= 9;
    temp = x3; while (temp > 0) { mod9 += temp % 10; temp /= 10; } mod9 %= 9;
    results[idx * 3 + 0] = (mod9 == 0) ? 1 : 0;

    // Fast mod27 check
    uint64_t mod27 = 0;
    uint64_t carry = 0;
    temp = x0 + carry; mod27 = (mod27 + temp % 27) % 27; carry = temp / 27;
    temp = x1 + carry; mod27 = (mod27 + temp % 27) % 27; carry = temp / 27;
    temp = x2 + carry; mod27 = (mod27 + temp % 27) % 27; carry = temp / 27;
    temp = x3 + carry; mod27 = (mod27 + temp % 27) % 27; carry = temp / 27;
    while (carry > 0) { mod27 = (mod27 + carry % 27) % 27; carry /= 27; }
    results[idx * 3 + 1] = (mod27 == 0) ? 1 : 0;

    // Fast mod81 check
    uint64_t mod81 = 0;
    carry = 0;
    temp = x0 + carry; mod81 = (mod81 + temp % 81) % 81; carry = temp / 81;
    temp = x1 + carry; mod81 = (mod81 + temp % 81) % 81; carry = temp / 81;
    temp = x2 + carry; mod81 = (mod81 + temp % 81) % 81; carry = temp / 81;
    temp = x3 + carry; mod81 = (mod81 + temp % 81) % 81; carry = temp / 81;
    while (carry > 0) { mod81 = (mod81 + carry % 81) % 81; carry /= 81; }
    results[idx * 3 + 2] = (mod81 == 0) ? 1 : 0;
}

// Deep note: Multi-modulus bias checking for efficient attractor filtering - results[idx*3] = mod9, results[idx*3+1] = mod27, results[idx*3+2] = mod81

// Optimized shared memory padding for bank conflict-free bias table access
// Added batch processing for 10^6 scalars per kernel launch
__global__ void bias_check_kernel_padded(uint32_t* dist_limbs, uint8_t* is_biased, uint32_t count, float* bias_global) {
    // Shared memory with padding to avoid bank conflicts (32 banks)
    __shared__ float bias_shared[81 + 31];  // Pad to 112 elements

    // Load bias table into shared memory with conflict-free access pattern
    if (threadIdx.x < 81) {
        uint32_t bank_idx = threadIdx.x % 32;
        uint32_t bank_offset = threadIdx.x / 32;
        uint32_t shared_idx = bank_idx + bank_offset * 32 + bank_offset;
        bias_shared[shared_idx] = bias_global[threadIdx.x];
    }
    __syncthreads();

    // Batch processing: each thread handles multiple scalars
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t batch_size = 16;  // Process 16 scalars per thread for better occupancy

    for (uint32_t batch = 0; batch < batch_size; ++batch) {
        uint32_t scalar_idx = idx * batch_size + batch;
        if (scalar_idx >= count) break;

        // Compute residue using optimized mod operation
        uint32_t res = dist_limbs[scalar_idx] % 81;

        // Access shared memory with conflict-free indexing
        uint32_t bank_idx = res % 32;
        uint32_t bank_offset = res / 32;
        uint32_t access_idx = bank_idx + bank_offset * 32 + bank_offset;

        float bias_factor = bias_shared[access_idx];
        is_biased[scalar_idx] = (bias_factor > 1.0f) ? 1 : 0;
    }
}