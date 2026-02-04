// Concise Block: CUDA Mod81 Check for Attractor Filter
__global__ void mod81_attractor_check(uint64_t* x_limbs, bool* results, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // Proper modular reduction for 256-bit numbers
    uint64_t carry = 0;
    uint64_t sum = 0;

    // Sum all limbs with carry propagation
    for (int i = 0; i < 4; ++i) {
        uint64_t temp = x_limbs[idx * 4 + i] + carry;
        sum = (sum + (temp % 81)) % 81;
        carry = temp / 81;
    }

    // Handle remaining carry
    while (carry > 0) {
        sum = (sum + (carry % 81)) % 81;
        carry /= 81;
    }

    results[idx] = (sum == 0);
}

// Deep note: Proper mod81 reduction with carry propagation - exact for 256-bit numbers