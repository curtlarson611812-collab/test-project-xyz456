// solve.cu - CUDA kernels for batch collision equation solving and Barrett reduction
// Implements priv = alpha_t - alpha_w + (beta_t - beta_w) * target mod n
// Barrett reduction: q = (x * mu)>>512, r = x - q*mod mod mod
// Hybrid with Montgomery for mul-heavy operations

#include <cuda_runtime.h>
#include <stdint.h>

#define LIMBS 8
#define WIDE_LIMBS 16

// BigInt256 struct for unified CPU/GPU arithmetic (matches CPU BigInt256)
typedef struct {
    uint64_t limbs[4];  // LSB in limbs[0], MSB in limbs[3] - exact match to CPU BigInt256
} bigint256;

// BigInt256 Point structure for elliptic curve operations
typedef struct {
    bigint256 x;
    bigint256 y;
    bigint256 z;
} Point256;

// secp256k1 order (n) and secp256k1 prime (p) constants
__constant__ uint32_t SECP_N[LIMBS] = {
    0xD0364141u, 0xBFD25E8Cu, 0xAF48A03Bu, 0xBAAEDCE6u,
    0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
};

__constant__ uint32_t SECP_P[LIMBS] = {
    0xFFFFFC2Fu, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
};

// Point structure for elliptic curve operations in BSGS
struct Point {
    uint32_t x[LIMBS];
    uint32_t y[LIMBS];
    uint32_t z[LIMBS];
};

// BSGS table entry for baby steps
struct BSGS_Entry {
    uint32_t point_x[LIMBS];  // x-coordinate of g^i
    uint32_t index;          // i
};

// Device function for safe diff mod N (Phase 4 GPU integration)
__device__ void safe_diff_mod_n(const bigint256& tame_dist, const bigint256& wild_dist, const bigint256& n, bigint256& result) {
    bigint256 diff;
    if (bigint_compare(tame_dist, wild_dist) >= 0) {
        bigint_sub(tame_dist, wild_dist, diff);
    } else {
        bigint256 temp;
        bigint_add(tame_dist, n, temp);
        bigint_sub(temp, wild_dist, diff);
    }
    bigint_mod(diff, n, result); // Use barrett if optimized
}

// BSGS table for giant steps (stored in global memory)
struct BSGS_Table {
    BSGS_Entry* baby_steps;
    int m;  // sqrt(order) size
};

// Device function: Point negation (y = -y mod p)
__device__ void point_neg(const Point* p, Point* neg, const uint32_t* mod) {
    for (int i = 0; i < LIMBS; i++) {
        neg->x[i] = p->x[i];
        neg->z[i] = p->z[i];
        neg->y[i] = mod[i] - p->y[i];  // -y = p - y mod p
    }
}

// Device function: Point subtraction (add p1 + neg p2)
__device__ void point_sub(const Point* p1, const Point* p2, Point* result, const uint32_t* mod) {
    Point neg_p2;
    point_neg(p2, &neg_p2, mod);
    point_add_jacobian(p1, &neg_p2, result, mod);
}

// Device function: Point addition in Jacobian coordinates (simplified)
__device__ void point_add_jacobian(const Point* p1, const Point* p2, Point* result, const uint32_t* mod) {
    // Simplified Jacobian point addition - would need full implementation
    // For BSGS, we mainly need scalar multiplication and equality checks
    // This is a placeholder - real implementation would be more complex
    for (int i = 0; i < LIMBS; i++) {
        result->x[i] = (p1->x[i] + p2->x[i]) % mod[i];
        result->y[i] = (p1->y[i] + p2->y[i]) % mod[i];
        result->z[i] = 1; // Simplified
    }
}

// Device function: Scalar multiplication by small integer
__device__ void point_mul_small(const Point* p, uint32_t scalar, Point* result, const uint32_t* mod) {
    // Initialize result to infinity
    for (int i = 0; i < LIMBS; i++) {
        result->x[i] = 0;
        result->y[i] = 0;
        result->z[i] = 0;
    }

    Point current = *p;
    while (scalar > 0) {
        if (scalar & 1) {
            point_add_jacobian(result, &current, result, mod);
        }
        point_add_jacobian(&current, &current, &current, mod); // Double
        scalar >>= 1;
    }
}

// Device function: Modular inverse via extended Euclid (simplified for u64 approximation)
__device__ uint64_t mod_inverse_u64(uint64_t a, uint64_t mod) {
    int64_t m = mod, m0 = m, t, q;
    int64_t x0 = 0, x1 = 1;

    if (m == 1) return 0;

    while (a > 1) {
        q = a / m;
        t = m;
        m = a % m;
        a = t;
        t = x0;
        x0 = x1 - q * x0;
        x1 = t;
    }

    if (x1 < 0) x1 += m0;
    return x1;
}

// Device function: Factor out small primes (simplified u64)
__device__ uint64_t factor_small_primes_u64(uint64_t val) {
    for (int i = 0; i < 32; i++) {
        uint64_t p = PRIME_MULTIPLIERS[i];
        while (val % p == 0 && val >= p) {
            val /= p;
        }
    }
    return val;
}

// Device function: Check if two points are equal (x-coordinate comparison)
__device__ bool point_equal(const Point* p1, const Point* p2) {
    for (int i = 0; i < LIMBS; i++) {
        if (p1->x[i] != p2->x[i]) return false;
    }
    return true;
}

// Device function: Build baby steps table for BSGS
__global__ void build_baby_steps(Point* baby_table, int m, const Point* generator, const uint32_t* mod) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m) return;

    Point result;
    point_mul_small(generator, idx, &result, mod);

    for (int i = 0; i < LIMBS; i++) {
        baby_table[idx].x[i] = result.x[i];
        baby_table[idx].y[i] = result.y[i];
        baby_table[idx].z[i] = result.z[i];
    }
}

// Device function: BSGS solve kernel - find x such that g^x = target
__global__ void bsgs_solve_kernel(
    const Point* target,
    const Point* generator,
    const Point* baby_table,
    uint32_t* results,
    int m,
    int batch_size,
    const uint32_t* mod
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // For each batch item, we would have different targets
    // For now, assume single target per kernel launch
    const Point* current_target = target;

    // Build giant step table in shared memory or use global lookup
    // Simplified: just check baby steps first
    for (int i = 0; i < m; i++) {
        if (point_equal(&baby_table[i], current_target)) {
            results[idx] = i;
            return;
        }
    }

    // If not found in baby steps, would need giant step search
    // For now, mark as not found
    results[idx] = 0xFFFFFFFF;
}

// Device helper: Parallel big integer subtraction with borrow propagation
__device__ void bigint_sub_par(const uint32_t a[LIMBS], const uint32_t b[LIMBS], uint32_t result[LIMBS]) {
    int limb_idx = threadIdx.x % LIMBS;
    uint32_t borrow_in = 0;

    // Get borrow from previous limb via warp shuffle
    if (limb_idx > 0) {
        borrow_in = __shfl_sync(0xFFFFFFFF, (result[limb_idx - 1] >> 31) & 1, limb_idx - 1);
    }

    uint64_t diff = (uint64_t)a[limb_idx] - b[limb_idx] - borrow_in;
    result[limb_idx] = diff & 0xFFFFFFFFULL;

    uint32_t borrow_out = (diff >> 63) & 1;

    // Propagate borrow to next limb
    if (limb_idx < LIMBS - 1) {
        __shfl_sync(0xFFFFFFFF, borrow_out, limb_idx + 1);
    }
    __syncthreads();
}

// Device helper: Parallel big integer addition with carry propagation
__device__ void bigint_add_par(const uint32_t a[LIMBS], const uint32_t b[LIMBS], uint32_t result[LIMBS]) {
    int limb_idx = threadIdx.x % LIMBS;
    uint32_t carry_in = 0;

    // Get carry from previous limb via warp shuffle
    if (limb_idx > 0) {
        carry_in = __shfl_sync(0xFFFFFFFF, result[limb_idx - 1] >> 31, limb_idx - 1);
    }

    uint64_t sum = (uint64_t)a[limb_idx] + b[limb_idx] + carry_in;
    result[limb_idx] = sum & 0xFFFFFFFFULL;

    uint32_t carry_out = (sum >> 32) & 0xFFFFFFFFULL;

    // Propagate carry to next limb
    if (limb_idx < LIMBS - 1) {
        __shfl_sync(0xFFFFFFFF, carry_out, limb_idx + 1);
    }
    __syncthreads();
}

// Device helper: Parallel big integer multiplication (8x8 -> 16 limbs)
__device__ void bigint_mul_par(const uint32_t a[LIMBS], const uint32_t b[LIMBS], uint32_t result[WIDE_LIMBS]) {
    int limb_idx = threadIdx.x % LIMBS;

    // Initialize result
    if (limb_idx < WIDE_LIMBS) result[limb_idx] = 0;
    __syncthreads();

    // Each limb of a multiplies with each limb of b
    for (int i = 0; i < LIMBS; i++) {
        if (limb_idx < LIMBS) {
            uint64_t prod = (uint64_t)a[i] * b[limb_idx];
            uint32_t low = prod & 0xFFFFFFFFULL;
            uint32_t high = (prod >> 32) & 0xFFFFFFFFULL;

            // Atomic add to result limbs
            atomicAdd(&result[i + limb_idx], low);
            if (i + limb_idx + 1 < WIDE_LIMBS) {
                atomicAdd(&result[i + limb_idx + 1], high);
            }
        }
    }
    __syncthreads();

    // Carry propagation for the result
    if (limb_idx < WIDE_LIMBS) {
        uint32_t carry = 0;
        uint64_t sum = result[limb_idx] + carry;
        result[limb_idx] = sum & 0xFFFFFFFFULL;
        carry = (sum >> 32) & 0xFFFFFFFFULL;

        if (limb_idx < WIDE_LIMBS - 1) {
            __shfl_sync(0xFFFFFFFF, carry, limb_idx + 1);
        }
    }
    __syncthreads();
}

// Device helper: Parallel big integer comparison
__device__ int bigint_cmp_par(const uint32_t a[LIMBS], const uint32_t b[LIMBS]) {
    int limb_idx = threadIdx.x % LIMBS;
    int local_cmp = 0;

    if (limb_idx < LIMBS) {
        if (a[limb_idx] > b[limb_idx]) local_cmp = 1;
        else if (a[limb_idx] < b[limb_idx]) local_cmp = -1;
    }

    // Find the most significant difference
    int msb_diff = 0;
    for (int i = LIMBS - 1; i >= 0; i--) {
        int cmp = __shfl_sync(0xFFFFFFFF, local_cmp, i);
        if (cmp != 0) {
            msb_diff = cmp;
            break;
        }
    }

    return msb_diff;
}

// Device helper: Montgomery reduction (simplified)
__device__ void montgomery_redc_par(const uint32_t t[WIDE_LIMBS], const uint32_t mod_[LIMBS], uint32_t n_prime, uint32_t result[LIMBS]) {
    // Simplified REDC implementation - would need full CIOS/FIOS algorithm for production
    uint32_t m, carry;
    uint32_t temp[WIDE_LIMBS];

    // Copy t to temp
    for (int i = 0; i < WIDE_LIMBS; i++) temp[i] = t[i];

    // REDC loop
    for (int i = 0; i < LIMBS; i++) {
        m = ((uint64_t)temp[i] * n_prime) & 0xFFFFFFFFULL;
        carry = 0;

        for (int j = 0; j < LIMBS; j++) {
            uint64_t prod = (uint64_t)m * mod_[j] + temp[i + j] + carry;
            temp[i + j] = prod & 0xFFFFFFFFULL;
            carry = (prod >> 32) & 0xFFFFFFFFULL;
        }

        // Propagate remaining carry
        for (int j = LIMBS; j < WIDE_LIMBS - i; j++) {
            uint64_t sum = temp[i + j] + carry;
            temp[i + j] = sum & 0xFFFFFFFFULL;
            carry = (sum >> 32) & 0xFFFFFFFFULL;
        }
    }

    // Extract upper half
    for (int i = 0; i < LIMBS; i++) result[i] = temp[i + LIMBS];

    // Conditional subtraction
    if (bigint_cmp_par(result, mod_) >= 0) {
        bigint_sub_par(result, mod_, result);
    }
}

// Device helper: Wide multiplication for Barrett (16x9 -> 25 limbs, but we take upper)
__device__ void bigint_wide_mul_par(const uint32_t a[WIDE_LIMBS], const uint32_t b[LIMBS+1], uint32_t result[WIDE_LIMBS + LIMBS + 1]) {
    int limb_idx = threadIdx.x % WIDE_LIMBS;

    // Initialize result
    if (limb_idx < WIDE_LIMBS + LIMBS + 1) result[limb_idx] = 0;
    __syncthreads();

    // Each limb of a multiplies with each limb of b
    for (int i = 0; i < WIDE_LIMBS; i++) {
        if (limb_idx < LIMBS + 1) {
            uint64_t prod = (uint64_t)a[i] * b[limb_idx];
            uint32_t low = prod & 0xFFFFFFFFULL;
            uint32_t high = (prod >> 32) & 0xFFFFFFFFULL;

            atomicAdd(&result[i + limb_idx], low);
            atomicAdd(&result[i + limb_idx + 1], high);
        }
    }
    __syncthreads();

    // Carry propagation
    if (limb_idx < WIDE_LIMBS + LIMBS + 1) {
        uint32_t carry = 0;
        uint64_t sum = result[limb_idx] + carry;
        result[limb_idx] = sum & 0xFFFFFFFFULL;
        carry = (sum >> 32) & 0xFFFFFFFFULL;

        if (limb_idx < WIDE_LIMBS + LIMBS) {
            __shfl_sync(0xFFFFFFFF, carry, limb_idx + 1);
        }
    }
    __syncthreads();
}

// Batch collision equation solving kernel
// priv = alpha_t - alpha_w + (beta_t - beta_w) * target mod n
__global__ void batch_collision_solve(uint32_t *alpha_t, uint32_t *alpha_w, uint32_t *beta_t, uint32_t *beta_w, uint32_t *target, uint32_t *n, uint32_t *priv_out, int batch) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= batch) return;

    // Shared memory for intermediate results (one set per block)
    __shared__ uint32_t alpha_diff[LIMBS];
    __shared__ uint32_t beta_diff[LIMBS];
    __shared__ uint32_t mul_part[WIDE_LIMBS];
    __shared__ uint32_t priv_temp[WIDE_LIMBS];

    int limb_idx = threadIdx.x % LIMBS;

    // Load inputs for this batch item
    uint32_t alpha_t_limb = alpha_t[id * LIMBS + limb_idx];
    uint32_t alpha_w_limb = alpha_w[id * LIMBS + limb_idx];
    uint32_t beta_t_limb = beta_t[id * LIMBS + limb_idx];
    uint32_t beta_w_limb = beta_w[id * LIMBS + limb_idx];
    uint32_t target_limb = target[id * LIMBS + limb_idx];  // Assuming target per batch item
    uint32_t n_limb = n[limb_idx];

    // Step 1: Compute alpha_diff = alpha_t - alpha_w
    bigint_sub_par(&alpha_t[id * LIMBS], &alpha_w[id * LIMBS], alpha_diff);

    // Step 2: Compute beta_diff = beta_t - beta_w
    bigint_sub_par(&beta_t[id * LIMBS], &beta_w[id * LIMBS], beta_diff);

    // Step 3: Compute mul_part = beta_diff * target
    bigint_mul_par(beta_diff, &target[id * LIMBS], mul_part);

    // Step 4: Compute priv_temp = alpha_diff + mul_part
    bigint_add_par(alpha_diff, mul_part, priv_temp);

    // Step 5: Reduce modulo n
    // First, check if priv_temp >= n and subtract if needed
    if (bigint_cmp_par(priv_temp, n) >= 0) {
        bigint_sub_par(priv_temp, n, priv_temp);
    }

    // Handle negative results (if alpha_diff was negative)
    // Check if priv_temp is negative (most significant bit set)
    if ((priv_temp[LIMBS-1] & 0x80000000) != 0) {
        bigint_add_par(priv_temp, n, priv_temp);
    }

    // Store result
    priv_out[id * LIMBS + limb_idx] = priv_temp[limb_idx];
}

// Batch BSGS collision solving for near-collisions
__global__ void batch_bsgs_collision_solve(
    const Point* deltas,        // Delta points (target - current)
    const uint32_t* alphas,     // Alpha coefficients for prime inverse
    const uint32_t* distances,  // Distance arrays for each trap
    uint32_t* solutions,        // Output solutions
    int batch_size,             // Number of collisions to solve
    uint64_t bsgs_threshold,    // Max difference for BSGS
    const uint32_t* mod,        // Modulus (secp256k1 order)
    int gold_bias_combo         // Enable GOLD factoring
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const Point* delta = &deltas[idx];
    const uint32_t* alpha = &alphas[idx * LIMBS];
    const uint32_t* dist = &distances[idx * LIMBS];

    // Diff u64 approx (sum low limbs)
    uint64_t diff = 0;
    for (int i = 0; i < LIMBS; i++) {
        diff += ((uint64_t)dist[i]) << (i * 32);
    }

    // GOLD factoring: reduce diff by factoring small primes
    if (gold_bias_combo) {
        uint64_t reduced = factor_small_primes_u64(diff);
        if (reduced < diff) {
            // Approximate factor count for threshold reduction
            int factor_count = 0;
            uint64_t temp = diff;
            while (temp > reduced) {
                temp = factor_small_primes_u64(temp);
                factor_count++;
            }
            if (factor_count > 1) {
                bsgs_threshold /= factor_count;  // Reduce threshold
            }
        }
    }

    if (diff < bsgs_threshold) {
        // Try prime inverse first if alpha is non-zero
        uint32_t zero[LIMBS] = {0};
        if (bigint_cmp_par(alpha, zero) != 0) {
            // Use simplified u64 inverse for low limb approximation
            uint64_t alpha_low = (uint64_t)alpha[0] | ((uint64_t)alpha[1] << 32);
            uint64_t mod_low = (uint64_t)mod[0] | ((uint64_t)mod[1] << 32);

            if (alpha_low > 0) {
                uint64_t inv = mod_inverse_u64(alpha_low, mod_low);
                if (inv != 0) {
                    // k = inv * (diff + 1) mod order
                    uint64_t k = (inv * (diff + 1)) % mod_low;

                    // Store result (low 64 bits)
                    solutions[idx * LIMBS] = k & 0xFFFFFFFFULL;
                    solutions[idx * LIMBS + 1] = (k >> 32) & 0xFFFFFFFFULL;
                    for (int i = 2; i < LIMBS; i++) {
                        solutions[idx * LIMBS + i] = 0;
                    }
                    return;
                }
            }
        }

        // Fallback: BSGS algorithm
        uint64_t m = (uint64_t)sqrt((double)bsgs_threshold) + 1;

        // Use shared memory for baby steps (limit to reasonable size)
        __shared__ uint32_t baby_x_shared[1024 * LIMBS];  // x coordinates only
        __shared__ uint32_t baby_y_shared[1024 * LIMBS];  // y coordinates only

        // Build baby steps: g^0, g^1, ..., g^{m-1}
        if (threadIdx.x < m) {
            // Generator point (secp256k1 base point)
            Point generator = {
                {0x79BE667E, 0xF9DCBBAC, 0x55A06295, 0xCE870B07,
                 0x029BFCDB, 0x2DCE28D9, 0x59F2815B, 0x16F81798}, // x
                {0x483ADA77, 0x26A3C465, 0x5DA4FBFC, 0x0E1108A8,
                 0xFD17B448, 0xA6855419, 0x9C47D08F, 0xFB10D4B8}, // y
                {1, 0, 0, 0, 0, 0, 0, 0}  // z
            };

            Point baby;
            point_mul_small(&generator, threadIdx.x, &baby, mod);

            // Store in shared memory (x and y only for equality check)
            for (int i = 0; i < LIMBS; i++) {
                baby_x_shared[threadIdx.x * LIMBS + i] = baby.x[i];
                baby_y_shared[threadIdx.x * LIMBS + i] = baby.y[i];
            }
        }
        __syncthreads();

        // Giant step search
        for (uint64_t j = 0; j < m; j++) {
            // Compute giant step: g^{-(j*m)} * delta
            Point generator = {
                {0x79BE667E, 0xF9DCBBAC, 0x55A06295, 0xCE870B07,
                 0x029BFCDB, 0x2DCE28D9, 0x59F2815B, 0x16F81798},
                {0x483ADA77, 0x26A3C465, 0x5DA4FBFC, 0x0E1108A8,
                 0xFD17B448, 0xA6855419, 0x9C47D08F, 0xFB10D4B8},
                {1, 0, 0, 0, 0, 0, 0, 0}
            };

            uint64_t giant_scalar = j * m;
            Point giant_step;
            point_mul_small(&generator, giant_scalar, &giant_step, mod);

            // target = delta - giant_step
            Point target_check;
            point_sub(delta, &giant_step, &target_check, mod);

            // Check if target matches any baby step
            for (uint64_t b = 0; b < m; b++) {
                bool match = true;
                for (int i = 0; i < LIMBS; i++) {
                    if (target_check.x[i] != baby_x_shared[b * LIMBS + i] ||
                        target_check.y[i] != baby_y_shared[b * LIMBS + i]) {
                        match = false;
                        break;
                    }
                }

                if (match) {
                    // Found solution: x = j*m + b
                    uint64_t solution = j * m + b;

                    // Store solution (split into uint32_t array)
                    solutions[idx * LIMBS] = solution & 0xFFFFFFFFULL;
                    solutions[idx * LIMBS + 1] = (solution >> 32) & 0xFFFFFFFFULL;
                    for (int k = 2; k < LIMBS; k++) {
                        solutions[idx * LIMBS + k] = 0;
                    }
                    return;
                }
            }
        }

        // No solution found
        for (int i = 0; i < LIMBS; i++) {
            solutions[idx * LIMBS + i] = 0xFFFFFFFF;
        }
    } else {
        // Difference too large for BSGS
        for (int i = 0; i < LIMBS; i++) {
            solutions[idx * LIMBS + i] = 0xFFFFFFFF;
        }
    }
}

// Batch Barrett reduction kernel with hybrid Montgomery option
__global__ void batch_barrett_reduce(uint32_t *x, uint32_t *mu, uint32_t *mod, uint32_t *out, bool use_mont, uint32_t n_prime, int batch, int limbs) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= batch) return;

    int limb_idx = threadIdx.x % limbs;

    if (use_mont) {
        // Use Montgomery reduction for mul-heavy operations
        montgomery_redc_par(&x[id * WIDE_LIMBS], mod, n_prime, &out[id * limbs]);
    } else {
        // Barrett reduction: q = (x * mu) >> (2*k), r = x - q * mod
        __shared__ uint32_t x_wide[WIDE_LIMBS];
        __shared__ uint32_t q[LIMBS];
        __shared__ uint32_t qp[WIDE_LIMBS];
        __shared__ uint32_t result[LIMBS];

        // Load x (512-bit input)
        if (limb_idx < WIDE_LIMBS) {
            x_wide[limb_idx] = x[id * WIDE_LIMBS + limb_idx];
        }
        __syncthreads();

        // Compute x * mu (wide multiplication)
        uint32_t x_mu[WIDE_LIMBS + LIMBS + 1];
        bigint_wide_mul_par(x_wide, mu, x_mu);

        // Extract q from upper bits (approximation of x * mu >> 512)
        // For 512-bit Barrett, we need bits 512 to 512+k-1
        if (limb_idx < limbs) {
            q[limb_idx] = x_mu[WIDE_LIMBS + limb_idx];  // Upper 256 bits of product
        }
        __syncthreads();

        // Compute q * mod
        bigint_mul_par(q, mod, qp);

        // Compute r = x - q * mod (lower 256 bits)
        if (limb_idx < limbs) {
            uint64_t diff = (uint64_t)x_wide[limb_idx] - qp[limb_idx];
            result[limb_idx] = diff & 0xFFFFFFFFULL;
            uint32_t borrow = (diff >> 63) & 1;

            // Propagate borrow
            if (limb_idx < limbs - 1) {
                __shfl_sync(0xFFFFFFFF, borrow, limb_idx + 1);
            }
        }
        __syncthreads();

        // Handle borrow propagation properly
        if (limb_idx < limbs) {
            uint32_t borrow = 0;
            if (limb_idx > 0) {
                borrow = __shfl_sync(0xFFFFFFFF, (result[limb_idx - 1] >> 31) & 1, limb_idx - 1);
            }
            uint64_t corrected = (uint64_t)result[limb_idx] - borrow;
            result[limb_idx] = corrected & 0xFFFFFFFFULL;
        }
        __syncthreads();

        // Conditional subtraction: if r >= mod, r -= mod
        if (bigint_cmp_par(result, mod) >= 0) {
            bigint_sub_par(result, mod, result);
        }

        // Store final result
        out[id * limbs + limb_idx] = result[limb_idx];
    }
}

// Fused batch collision solve with integrated Montgomery reduction
__global__ void batch_collision_solve_fused(uint32_t *alpha_t, uint32_t *alpha_w, uint32_t *beta_t, uint32_t *beta_w, uint32_t *target, uint32_t *n, uint32_t n_prime, uint32_t *priv_out, int batch) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= batch) return;

    __shared__ uint32_t alpha_diff[LIMBS];
    __shared__ uint32_t beta_diff[LIMBS];
    __shared__ uint32_t mul_part[WIDE_LIMBS];
    __shared__ uint32_t priv_temp[WIDE_LIMBS];

    int limb_idx = threadIdx.x % LIMBS;

    // Load inputs
    uint32_t alpha_t_limb = alpha_t[id * LIMBS + limb_idx];
    uint32_t alpha_w_limb = alpha_w[id * LIMBS + limb_idx];
    uint32_t beta_t_limb = beta_t[id * LIMBS + limb_idx];
    uint32_t beta_w_limb = beta_w[id * LIMBS + limb_idx];
    uint32_t target_limb = target[id * LIMBS + limb_idx];
    uint32_t n_limb = n[limb_idx];

    // Compute alpha_diff = alpha_t - alpha_w
    bigint_sub_par(&alpha_t[id * LIMBS], &alpha_w[id * LIMBS], alpha_diff);

    // Compute beta_diff = beta_t - beta_w
    bigint_sub_par(&beta_t[id * LIMBS], &beta_w[id * LIMBS], beta_diff);

    // Compute mul_part = beta_diff * target
    bigint_mul_par(beta_diff, &target[id * LIMBS], mul_part);

    // Compute priv_temp = alpha_diff + mul_part
    bigint_add_par(alpha_diff, mul_part, priv_temp);

    // Fused Montgomery reduction instead of separate modulo
    montgomery_redc_par(priv_temp, n, n_prime, &priv_out[id * LIMBS]);
}

// Host function for launching BSGS collision solving
extern "C" void launch_batch_bsgs_collision_solve(
    Point* d_deltas,
    uint32_t* d_alphas,
    uint32_t* d_distances,
    uint32_t* d_solutions,
    int batch_size,
    uint64_t bsgs_threshold,
    cudaStream_t stream,
    int gold_bias_combo
) {
    dim3 block(256);
    dim3 grid((batch_size + 255) / 256);

    // Get secp256k1 order
    uint32_t h_mod[LIMBS] = {
        0xD0364141u, 0xBFD25E8Cu, 0xAF48A03Bu, 0xBAAEDCE6u,
        0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
    };

    uint32_t* d_mod;
    cudaMalloc(&d_mod, LIMBS * sizeof(uint32_t));
    cudaMemcpy(d_mod, h_mod, LIMBS * sizeof(uint32_t), cudaMemcpyHostToDevice);

    batch_bsgs_collision_solve<<<grid, block, 0, stream>>>(
        d_deltas, d_alphas, d_distances, d_solutions,
        batch_size, bsgs_threshold, d_mod, gold_bias_combo
    );

    cudaFree(d_mod);
}

// Host function for building baby steps table
extern "C" void launch_build_baby_steps(
    Point* d_baby_table,
    int m,
    Point* d_generator,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((m + 255) / 256);

    // Get secp256k1 modulus
    uint32_t h_mod[LIMBS] = {
        0xFFFFFC2Fu, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
        0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
    };

    uint32_t* d_mod;
    cudaMalloc(&d_mod, LIMBS * sizeof(uint32_t));
    cudaMemcpy(d_mod, h_mod, LIMBS * sizeof(uint32_t), cudaMemcpyHostToDevice);

    build_baby_steps<<<grid, block, 0, stream>>>(
        d_baby_table, m, d_generator, d_mod
    );

    cudaFree(d_mod);
}

// Collision check kernel with safe diff mod N
__global__ void check_collisions(bigint256* tame_dists, bigint256* wild_dists, bigint256* results, int batch_size, bigint256 n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    bigint256 priv_big;
    safe_diff_mod_n(tame_dists[idx], wild_dists[idx], n, priv_big);
    results[idx] = priv_big;
}

// Host function for launching BSGS solve
extern "C" void launch_bsgs_solve(
    Point* d_target,
    Point* d_generator,
    Point* d_baby_table,
    uint32_t* d_results,
    int m,
    int batch_size,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((batch_size + 255) / 256);

    // Get secp256k1 modulus
    uint32_t h_mod[LIMBS] = {
        0xFFFFFC2Fu, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
        0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
    };

    uint32_t* d_mod;
    cudaMalloc(&d_mod, LIMBS * sizeof(uint32_t));
    cudaMemcpy(d_mod, h_mod, LIMBS * sizeof(uint32_t), cudaMemcpyHostToDevice);

    bsgs_solve_kernel<<<grid, block, 0, stream>>>(
        d_target, d_generator, d_baby_table, d_results, m, batch_size, d_mod
    );

    cudaFree(d_mod);
}

// BigInt256 batch scalar multiplication kernel
__global__ void batch_scalar_mul(Point256 *results, Point256 *bases, bigint256 *scalars, int batch_size, bigint256 mod_p, bigint256 mu, bigint256 curve_a) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    Point256 res = {bigint256_zero(), bigint256_one(), bigint256_zero()}; // infinity
    Point256 cur = bases[idx];
    bigint256 k = scalars[idx];
    for (int bit = 0; bit < 256; bit++) {
        uint64_t bit_mask = 1ULL << (bit % 64);
        if ((k.limbs[bit / 64] & bit_mask) != 0) {
            res = ec_add(res, cur, mod_p, mu, curve_a);
        }
        cur = jacobian_double(cur, mod_p, mu, curve_a);
    }
    results[idx] = res;  // To affine later if needed
}

// Forward declarations for BigInt256 functions (defined in other .cu files)
__device__ bigint256 bigint256_zero();
__device__ bigint256 bigint256_one();
__device__ bigint256 bigint256_add(bigint256 a, bigint256 b);
__device__ bigint256 bigint256_sub(bigint256 a, bigint256 b);
__device__ bigint256 bigint256_mul(bigint256 a, bigint256 b);
__device__ bigint256 bigint256_shr(bigint256 x, uint32_t bits);
__device__ int bigint256_cmp(bigint256 a, bigint256 b);
__device__ bool bigint256_ge(bigint256 a, bigint256 b);
__device__ bigint256 barrett_reduce(bigint256 x, bigint256 p, bigint256 mu);
__device__ bigint256 mont_mul(bigint256 a, bigint256 b, bigint256 p, bigint256 inv);
__device__ Point256 jacobian_double(Point256 p, bigint256 mod_p, bigint256 mu, bigint256 curve_a);
__device__ Point256 ec_add(Point256 p1, Point256 p2, bigint256 mod_p, bigint256 mu, bigint256 curve_a);
__device__ bool is_infinity(Point256 p);
__device__ bool is_zero(bigint256 val);
