// Common CUDA types shared across files
// Constants are defined in individual .cu files to avoid linking conflicts

#ifndef COMMON_CONSTANTS_H
#define COMMON_CONSTANTS_H

#include <stdint.h>

// Point structure for elliptic curve points (Jacobian coordinates)
struct Point {
    uint32_t x[8]; // X coordinate (256-bit)
    uint32_t y[8]; // Y coordinate (256-bit)
    uint32_t z[8]; // Z coordinate (256-bit)
};

// Kangaroo state structure (matches Rust definition)
struct KangarooState {
    Point position;
    uint32_t distance[8];    // Distance traveled (steps)
    uint32_t alpha[4];       // Alpha coefficient (64-bit stored as 32-bit pairs)
    uint32_t beta[4];        // Beta coefficient (64-bit stored as 32-bit pairs)
    uint32_t is_tame;        // 0 for wild, 1 for tame
    uint32_t is_dp;          // Whether reached distinguished point
    uint64_t id;             // Kangaroo ID
    uint64_t step;           // Current step count
    uint32_t kangaroo_type;  // 0 = tame, 1 = wild
};

// Constants are declared here but defined in step.cu to avoid duplicates
extern __constant__ uint32_t P[8];
extern __constant__ uint32_t GLV_BETA[8];
extern __constant__ uint32_t GLV_LAMBDA[8];
extern __constant__ uint64_t PRIME_MULTIPLIERS[32];
extern __constant__ uint32_t GLV_V1_1_LIMBS[8];
extern __constant__ uint32_t GLV_V1_2_LIMBS[8];
extern __constant__ uint32_t GLV_V2_1_LIMBS[8];
extern __constant__ uint32_t GLV_V2_2_LIMBS[8];
extern __constant__ uint32_t GLV_R1_LIMBS[8];
extern __constant__ uint32_t GLV_R2_LIMBS[8];
extern __constant__ uint32_t GLV_SQRT_N_LIMBS[8];
extern __constant__ uint32_t GLV_LAMBDA_LIMBS[8];
extern __constant__ uint32_t GLV_BETA_LIMBS[8];
extern __constant__ uint32_t CURVE_N[8];
extern __constant__ uint32_t MU_N[16];

#endif // COMMON_CONSTANTS_H