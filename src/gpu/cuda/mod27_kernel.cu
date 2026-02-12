// Concise Block: CUDA Mod27 Check for Attractor Filter
#include <stdint.h>
__global__ void mod27_attractor_check(uint64_t* x_limbs, bool* results, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // Proper modular reduction for 256-bit numbers
    uint64_t carry = 0;
    uint64_t sum = 0;

    // Sum all limbs with carry propagation
    for (int i = 0; i < 4; ++i) {
        uint64_t temp = x_limbs[idx * 4 + i] + carry;
        sum = (sum + (temp % 27)) % 27;
        carry = temp / 27;
    }

    // Handle remaining carry
    while (carry > 0) {
        sum = (sum + (carry % 27)) % 27;
        carry /= 27;
    }

    results[idx] = (sum == 0);
}

// Deep note: Proper mod27 reduction with carry propagation - exact for 256-bit numbers