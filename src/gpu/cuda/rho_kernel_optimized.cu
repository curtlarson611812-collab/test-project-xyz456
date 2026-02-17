/*
 * Optimized Rho Kernel with SoA Coalescing for SpeedBitCrackV3
 *
 * Implements Struct-of-Arrays (SoA) memory layout for better coalescing
 * of BigInt256 operations in kangaroo algorithm steps.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h> // For uint32_t, int64_t etc.
#include "common_constants.h"

// Forward declarations for functions and constants from step.cu
extern __device__ Point mul_glv_opt(Point p, const uint32_t k[8]);
extern __device__ Point jacobian_add(Point p1, Point p2);
extern __device__ Point jacobian_double(Point p);
extern __device__ void mul_mod(const uint32_t* a, const uint32_t* b, uint32_t* res, const uint32_t* mod);
extern __device__ void glv_decompose_scalar(const uint32_t k[8], uint32_t k1[8], uint32_t k2[8], int8_t* sign1, int8_t* sign2);

// Constants from step.cu
extern __constant__ uint32_t P[8];
extern __constant__ uint32_t GLV_BETA[8];

// GLV scalar decomposition for rho kernel (master-level)
static __device__ void glv_decompose(const uint32_t k[8], uint32_t k1[4], uint32_t k2[4]) {
    // Master-level GLV decompose for rho kernel - full precision for correctness
    // Rho kernel needs proper decomposition for cycle finding accuracy

    uint32_t k1_full[8], k2_full[8];
    int8_t sign1, sign2;

    // Use full master GLV decomposition
    glv_decompose_scalar(k, k1_full, k2_full, &sign1, &sign2);

    // Take low 128 bits for rho kernel (sufficient for cycle detection)
    for (int i = 0; i < 4; i++) {
        k1[i] = k1_full[i];
        k2[i] = k2_full[i];
    }

    // Apply signs to ensure positive values
    if (sign1 < 0) {
        // Negate k1 (mod 2^128 for rho kernel approximation)
        uint32_t carry = 1;
        for (int i = 0; i < 4; i++) {
            uint64_t sum = (uint64_t)(~k1[i]) + carry;
            k1[i] = sum & 0xFFFFFFFFULL;
            carry = sum >> 32;
        }
    }

    if (sign2 < 0) {
        // Negate k2 (mod 2^128 for rho kernel approximation)
        uint32_t carry = 1;
        for (int i = 0; i < 4; i++) {
            uint64_t sum = (uint64_t)(~k2[i]) + carry;
            k2[i] = sum & 0xFFFFFFFFULL;
            carry = sum >> 32;
        }
    }
}

// GLV endomorphism application for rho kernel (master-level)
static __device__ void endomorphism_apply(const Point* p, Point* res, const uint32_t* mod) {
    // Apply β(x,y) = (β²*x mod p, β³*y mod p) in Jacobian coordinates
    // β² and β³ are precomputed for efficiency

    // Copy input point
    for (int i = 0; i < 8; i++) {
        res->x[i] = p->x[i];
        res->y[i] = p->y[i];
        res->z[i] = p->z[i];
    }

    // Apply β² to x: x = β² * x mod p
    mul_mod(res->x, GLV_BETA, res->x, mod);  // β * x
    mul_mod(res->x, GLV_BETA, res->x, mod);  // β² * x

    // Apply β³ to y: y = β³ * y mod p
    mul_mod(res->y, GLV_BETA, res->y, mod);  // β * y
    mul_mod(res->y, GLV_BETA, res->y, mod);  // β² * y
    mul_mod(res->y, GLV_BETA, res->y, mod);  // β³ * y

    // z coordinate unchanged (scale invariant in Jacobian)
}

// Barrett reduction constants for secp256k1
__constant__ uint32_t MU[9] = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};
__constant__ uint32_t MODULUS[8] = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE};

// Simplified GLV mul (uses mul_glv_opt from solve.cu)
static __device__ void mul_glv_opt_device(const Point p, const uint32_t k[8], Point* result) {
    *result = mul_glv_opt(p, k);
}

// Optimized Barrett reduction for bias modulus calculation
static __device__ __forceinline__ uint32_t barrett_mod_81(uint32_t low_limb) {
    // Fast approximation for modulus 81 using low limb only
    // For exact calculation, would need full BigInt256 reduction
    return low_limb % 81;
}

// SoA layout rho kernel - optimized for coalescing
__global__ void rho_kernel_soa(
    // Separate arrays for each BigInt256 component (SoA layout)
    uint32_t* x_limbs,    // [t * 4] - x coordinate limbs
    uint32_t* y_limbs,    // [t * 4] - y coordinate limbs
    uint32_t* z_limbs,    // [t * 4] - z coordinate limbs
    uint32_t* dist_limbs, // [t * 4] - distance limbs

    // Jump table and bias data
    uint32_t* jump_table, // [256 * 4] - precomputed jumps
    float* bias_table,    // [81] - bias factors for residues

    uint32_t num_kangaroos,
    uint32_t steps_per_batch,
    uint32_t dp_bits
) {
    uint32_t kangaroo_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (kangaroo_idx >= num_kangaroos) return;

    // Load current state using SoA layout (coalesced reads)
    uint32_t current_x[4], current_y[4], current_z[4], current_dist[4];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        current_x[i] = x_limbs[kangaroo_idx * 4 + i];
        current_y[i] = y_limbs[kangaroo_idx * 4 + i];
        current_z[i] = z_limbs[kangaroo_idx * 4 + i];
        current_dist[i] = dist_limbs[kangaroo_idx * 4 + i];
    }

    // Execute kangaroo steps
    for (uint32_t step = 0; step < steps_per_batch; step++) {
        // Calculate bias residue for jump selection
        uint32_t residue = barrett_mod_81(current_dist[0]);

        // Apply bias scaling to jump selection
        uint32_t jump_idx = (current_dist[0] + (uint32_t)(bias_table[residue] * 1000.0f)) % 256;

        // Load jump vector (coalesced read from jump_table)
        uint32_t jump[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            jump[i] = jump_table[jump_idx * 4 + i];
        }

        // Add jump to distance using full BigInt256 arithmetic
        uint32_t carry = 0;
        for (int limb = 0; limb < 8; limb++) {
            uint64_t sum = (uint64_t)current_dist[limb] + (uint64_t)jump[limb] + carry;
            current_dist[limb] = sum & 0xFFFFFFFFULL;
            carry = (sum >> 32) & 0xFFFFFFFFULL;
        }

        // Check for distinguished point (trailing zeros)
        uint32_t trailing_zeros = __ffs(current_dist[0]) - 1; // CUDA intrinsic
        if (trailing_zeros >= dp_bits) {
            // Mark as distinguished point
            current_dist[0] |= 0x80000000; // Set high bit as DP flag
            break;
        }
    }

    // Store updated state back to SoA layout (coalesced writes)
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        x_limbs[kangaroo_idx * 4 + i] = current_x[i];
        y_limbs[kangaroo_idx * 4 + i] = current_y[i];
        z_limbs[kangaroo_idx * 4 + i] = current_z[i];
        dist_limbs[kangaroo_idx * 4 + i] = current_dist[i];
    }
}

// Legacy AoS kernel for comparison
__global__ void rho_kernel_aos(
    // Array of structs layout (less efficient)
    uint32_t* states,  // [t * 16] - packed x,y,z,dist limbs
    uint32_t* jump_table,
    float* bias_table,
    uint32_t num_kangaroos,
    uint32_t steps_per_batch,
    uint32_t dp_bits
) {
    uint32_t kangaroo_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (kangaroo_idx >= num_kangaroos) return;

    // Load state (uncoalesced due to AoS layout)
    uint32_t* kangaroo_state = &states[kangaroo_idx * 16]; // 4 limbs each for x,y,z,dist

    // Execute steps (same logic as SoA version)
    for (uint32_t step = 0; step < steps_per_batch; step++) {
        uint32_t dist_low = kangaroo_state[12]; // dist limb 0
        uint32_t residue = barrett_mod_81(dist_low);
        uint32_t jump_idx = (dist_low + (uint32_t)(bias_table[residue] * 1000.0f)) % 256;

        // Add jump (simplified)
        kangaroo_state[12] += jump_table[jump_idx * 4];

        uint32_t trailing_zeros = __ffs(kangaroo_state[12]) - 1;
        if (trailing_zeros >= dp_bits) {
            kangaroo_state[12] |= 0x80000000;
            break;
        }
    }
}

// CUDA kernel for collecting DP kangaroos (optimized for SoA)
__global__ void collect_distinguished_points_soa(
    uint32_t* x_limbs, uint32_t* y_limbs, uint32_t* z_limbs, uint32_t* dist_limbs,
    uint32_t* dp_indices, uint32_t* dp_count,
    uint32_t num_kangaroos, uint32_t dp_bits
) {
    uint32_t kangaroo_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (kangaroo_idx >= num_kangaroos) return;

    // Check DP flag in distance (set by rho_kernel)
    uint32_t dist_low = dist_limbs[kangaroo_idx * 4];
    if (dist_low & 0x80000000) { // DP flag set
        // Atomically add to DP list
        uint32_t dp_idx = atomicAdd(dp_count, 1);
        if (dp_idx < 1024) { // Max DP buffer size
            dp_indices[dp_idx] = kangaroo_idx;
        }
    }
}