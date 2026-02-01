// bigint_mul.cu - CUDA kernel for big integer multiplication
// Implements batch 256-bit multiplication: out[batch*16] = a[batch*8] * b[batch*8]
// Uses schoolbook multiplication with carry propagation

#include <cuda_runtime.h>
#include <stdint.h>

// Schoolbook multiplication kernel for 256-bit integers
// Each thread handles one multiplication: a[8] * b[8] -> result[16]
__global__ void bigint_mul_kernel(void *a_void, void *b_void, void *result_void, uint32_t batch) {
    const uint32_t *a = (const uint32_t *)a_void;
    const uint32_t *b = (const uint32_t *)b_void;
    uint32_t *result = (uint32_t *)result_void;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch) return;

    // Get pointers to this batch element's data
    const uint32_t *a_i = a + idx * 8;  // 8 limbs per 256-bit number
    const uint32_t *b_i = b + idx * 8;
    uint32_t *res_i = result + idx * 16; // 16 limbs for 512-bit result

    // Clear result
    for (int i = 0; i < 16; i++) {
        res_i[i] = 0;
    }

    // Schoolbook multiplication with carry propagation
    for (int i = 0; i < 8; i++) {
        uint32_t carry = 0;
        for (int j = 0; j < 8; j++) {
            // Compute a[i] * b[j] + carry + existing result[i+j]
            uint64_t prod = (uint64_t)a_i[i] * b_i[j] + carry + res_i[i + j];
            res_i[i + j] = prod & 0xFFFFFFFFULL;  // Store low 32 bits
            carry = prod >> 32;                   // Carry to next position
        }
        // Propagate remaining carry
        int k = i + 8;
        while (carry && k < 16) {
            uint64_t sum = (uint64_t)res_i[k] + carry;
            res_i[k] = sum & 0xFFFFFFFFULL;
            carry = sum >> 32;
            k++;
        }
    }
}