#!/usr/bin/env python3
"""
Custom Nsight Compute Rules for SpeedBitCrackV3 ECDLP Solver

Implements domain-specific rules for elliptic curve discrete logarithm
problem solving workloads, integrating with GROK Coder's optimization framework.
"""

import re
from nsight import *

class EcdlpBiasEfficiency(Rule):
    """
    ECDLP-specific rule for analyzing bias check kernel efficiency.

    Monitors ALU utilization in Barrett reduction operations and suggests
    fusion optimizations for bias checking workloads.
    """
    id = "EcdlpBiasEff"
    name = "ECDLP Bias Efficiency"
    description = "Analyzes bias check kernel efficiency for ECDLP workloads"
    category = "ECDLP"
    severity = Severity.WARNING

    def get_implementation(self):
        return """
        // Check for high ALU utilization in bias operations
        alu_pct = metrics["sm__pipe_alu_cycles_active.average.pct_of_peak_sustained_active"].value()
        ipc = metrics["sm__inst_executed.avg.pct_of_peak_sustained_active"].value()

        if alu_pct > 80 and ipc < 70:
            return Suggestion(
                "Fuse Barrett reduction in bias_check_kernel.cu",
                "High ALU utilization with low IPC indicates bias check inefficiency. " +
                "Consider fusing Barrett reduction operations to improve throughput.",
                Severity.WARNING
            )
        """

class EcdlpDivergenceAnalysis(Rule):
    """
    Analyzes warp divergence in DP (Distinguished Point) checking operations.

    ECDLP workloads often have divergent paths when checking trailing zeros
    for distinguished points, leading to poor SIMD efficiency.
    """
    id = "EcdlpDivergence"
    name = "ECDLP Divergence Analysis"
    description = "Detects warp divergence in distinguished point checking"
    category = "ECDLP"
    severity = Severity.WARNING

    def get_implementation(self):
        return """
        // Monitor warp efficiency and branch divergence
        warp_eff = metrics["sm__warps_active.avg.pct_of_peak_sustained_active"].value()
        branch_eff = metrics["sm__inst_executed.avg.pct_of_peak_sustained_elapsed"].value()

        if warp_eff < 90 or branch_eff < 0.8:
            return Suggestion(
                "Reduce divergence in DP checking",
                "High warp divergence detected in distinguished point operations. " +
                "Consider using subgroup operations or reorganizing DP checks.",
                Severity.WARNING
            )
        """

class EcdlpMemoryCoalescing(Rule):
    """
    Analyzes memory coalescing efficiency for BigInt256 operations.

    ECDLP requires frequent access to large integer arrays (256-bit numbers).
    Poor coalescing can severely impact performance.
    """
    id = "EcdlpMemoryCoalesce"
    name = "ECDLP Memory Coalescing"
    description = "Analyzes memory access patterns for BigInt operations"
    category = "ECDLP"
    severity = Severity.ERROR

    def get_implementation(self):
        return """
        // Check global memory coalescing
        gld_eff = metrics["sm__inst_executed.avg.pct_of_peak_sustained_active"].value()
        mem_throughput = metrics["dram__bytes.avg.pct_of_peak_sustained_active"].value()

        if gld_eff < 70:
            return Suggestion(
                "Improve memory coalescing",
                f"Low global load efficiency ({gld_eff:.1f}%). " +
                "Consider SoA layout for BigInt256 arrays or adjust access patterns.",
                Severity.ERROR
            )
        """

class EcdlpL1CacheUtilization(Rule):
    """
    Monitors L1 cache utilization for ECDLP constant data.

    Jump tables, curve parameters, and modulus values should be cached
    effectively in L1 for optimal performance.
    """
    id = "EcdlpL1Cache"
    name = "ECDLP L1 Cache Utilization"
    description = "Analyzes L1 cache efficiency for ECDLP constants"
    category = "ECDLP"
    severity = Severity.INFO

    def get_implementation(self):
        return """
        // Monitor L1 cache hit rates
        l1_hit_rate = metrics["l1tex__t_bytes_hit_rate.pct"].value()

        if l1_hit_rate < 80:
            return Suggestion(
                "Optimize L1 cache usage",
                f"L1 cache hit rate is {l1_hit_rate:.1f}%. " +
                "Consider using texture memory for jump tables or adjusting data layout.",
                Severity.INFO
            )
        """

class EcdlpSharedMemoryEfficiency(Rule):
    """
    Analyzes shared memory bank conflicts in ECDLP kernels.

    Bias tables and temporary BigInt storage should minimize bank conflicts
    for optimal shared memory throughput.
    """
    id = "EcdlpSharedMem"
    name = "ECDLP Shared Memory Efficiency"
    description = "Detects bank conflicts in shared memory usage"
    category = "ECDLP"
    severity = Severity.WARNING

    def get_implementation(self):
        return """
        // Monitor shared memory bank conflicts
        bank_conflicts = metrics["l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"].value()

        if bank_conflicts > 1000:  # Threshold for significant conflicts
            return Suggestion(
                "Resolve shared memory bank conflicts",
                f"High bank conflicts detected ({bank_conflicts} total). " +
                "Consider padding shared arrays or using different access patterns.",
                Severity.WARNING
            )
        """

class EcdlpOccupancyOptimization(Rule):
    """
    Analyzes GPU occupancy for ECDLP workloads.

    Optimal occupancy is crucial for hiding latency in compute-bound
    elliptic curve operations.
    """
    id = "EcdlpOccupancy"
    name = "ECDLP Occupancy Optimization"
    description = "Analyzes GPU occupancy for ECDLP kernels"
    category = "ECDLP"
    severity = Severity.INFO

    def get_implementation(self):
        return """
        // Check achieved occupancy
        achieved_occ = metrics["sm__warps_active.avg.pct_of_peak_sustained_active"].value()
        theoretical_max = metrics["sm__maximum_warps_per_active_cycle_pct"].value()

        if achieved_occ < 0.7 * theoretical_max:
            return Suggestion(
                "Increase GPU occupancy",
                f"Achieved occupancy ({achieved_occ:.1f}%) is below optimal. " +
                f"Consider reducing register usage or increasing block size.",
                Severity.INFO
            )
        """

# Register all custom ECDLP rules
def register_ecdlp_rules():
    """Register all ECDLP-specific rules with Nsight Compute."""
    rules = [
        EcdlpBiasEfficiency(),
        EcdlpDivergenceAnalysis(),
        EcdlpMemoryCoalescing(),
        EcdlpL1CacheUtilization(),
        EcdlpSharedMemoryEfficiency(),
        EcdlpOccupancyOptimization(),
    ]

    for rule in rules:
        register_rule(rule)

class EcdlpModularArithmeticEff(Rule):
    """
    Analyzes modular arithmetic efficiency in ECDLP operations.

    Barrett reduction and Montgomery multiplication should dominate ALU time.
    Poor efficiency indicates optimization opportunities.
    """
    id = "EcdlpModularArithmeticEff"
    name = "ECDLP Modular Arithmetic Efficiency"
    description = "Analyzes Barrett/Montgomery multiplication efficiency"
    category = "ECDLP"
    severity = Severity.WARNING

    def get_implementation(self):
        return """
        // Monitor ALU utilization in modular arithmetic
        alu_pct = metrics["sm__pipe_alu_cycles_active.average.pct_of_peak_sustained_active"].value()

        if alu_pct > 10:  # High ALU usage suggests modular arithmetic bottleneck
            return Suggestion(
                "Optimize modular arithmetic",
                f"ALU utilization {alu_pct:.1f}% indicates modular arithmetic bottleneck. " +
                "Consider shared memory for Barrett mu constants or fused operations.",
                Severity.WARNING
            )
        """

class EcdlpEcPointMulBalance(Rule):
    """
    Analyzes elliptic curve point multiplication balance.

    Optimal EC point multiplication requires balanced mul/square operations.
    Imbalanced operations indicate inefficient Jacobian coordinate usage.
    """
    id = "EcdlpEcMulBalance"
    name = "ECDLP EC Point Multiplication Balance"
    description = "Analyzes mul/square operation balance in EC arithmetic"
    category = "ECDLP"
    severity = Severity.INFO

    def get_implementation(self):
        return """
        // Check mul/square ratio for optimal EC arithmetic (target: 12m/4sq ≈ 3:1)
        muls = metrics["sm__inst_executed_pipe_alu_op_mul.count"].value()
        sqs = metrics["sm__inst_executed_pipe_alu_op_sqr.count"].value()

        if sqs > 0:
            ratio = muls / sqs
            if ratio < 2.5 or ratio > 3.5:
                return Suggestion(
                    "Balance EC point operations",
                    f"Mul/square ratio {ratio:.2f} deviates from optimal 3:1. " +
                    "Consider fusing add/double operations in Jacobian coordinates.",
                    Severity.INFO
                )
        """

class EcdlpDpDetectionDivergence(Rule):
    """
    Analyzes warp divergence in distinguished point detection.

    DP checking often involves conditional trailing zero checks that cause divergence.
    High divergence reduces SIMD efficiency.
    """
    id = "EcdlpDpDetectionDivergence"
    name = "ECDLP DP Detection Divergence"
    description = "Analyzes warp divergence in distinguished point checking"
    category = "ECDLP"
    severity = Severity.WARNING

    def get_implementation(self):
        return """
        // Monitor branch efficiency in DP detection
        branch_eff = metrics["sm__inst_executed.avg.pct_of_peak_sustained_elapsed"].value()

        if branch_eff < 0.8:  # Low branch efficiency indicates divergence
            return Suggestion(
                "Reduce DP detection divergence",
                f"Branch efficiency {branch_eff:.2f} indicates warp divergence in DP checking. " +
                "Consider subgroupBallot for warp-wide vote operations.",
                Severity.WARNING
            )
        """

class EcdlpBiasTableAccess(Rule):
    """
    Analyzes bias table access patterns for bank conflicts.

    Bias table lookups should minimize shared memory bank conflicts
    for optimal SIMD broadcast performance.
    """
    id = "EcdlpBiasTableAccess"
    name = "ECDLP Bias Table Access"
    description = "Analyzes shared memory access patterns for bias tables"
    category = "ECDLP"
    severity = Severity.WARNING

    def get_implementation(self):
        return """
        // Monitor shared memory bank conflicts
        bank_conflicts = metrics["l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"].value()

        if bank_conflicts > 100:  # Significant bank conflicts
            return Suggestion(
                "Optimize bias table access",
                f"Shared memory bank conflicts ({bank_conflicts}) detected. " +
                "Consider padding bias table arrays or using broadcast patterns.",
                Severity.WARNING
            )
        """

class EcdlpConstantTimeRule(Rule):
    """
    Ensures constant-time execution to prevent timing-based side channels.

    Variable-time operations in cryptographic code can leak information.
    """
    id = "EcdlpConstantTime"
    name = "ECDLP Constant-Time Verification"
    description = "Verifies constant-time execution in cryptographic operations"
    category = "ECDLP"
    severity = Severity.ERROR

    def get_implementation(self):
        return """
        // Monitor for variable-time operations
        branch_variance = metrics.get("warp_nonpred_execution_efficiency.variance", 0).value()

        if branch_variance > 5:  # High variance indicates timing leaks
            return Suggestion(
                "Ensure constant-time execution",
                f"Branch efficiency variance {branch_variance:.1f} suggests variable-time operations. " +
                "Remove data-dependent branches in mod_inverse and EC operations.",
                Severity.ERROR
            )
        """

# Register all additional ECDLP rules
def register_additional_rules():
    """Register additional ECDLP-specific rules."""
    additional_rules = [
        EcdlpModularArithmeticEff(),
        EcdlpEcPointMulBalance(),
        EcdlpDpDetectionDivergence(),
        EcdlpBiasTableAccess(),
        EcdlpConstantTimeRule(),
    ]

    for rule in additional_rules:
        register_rule(rule)

    print(f"Registered {len(additional_rules)} additional ECDLP-specific Nsight rules")

# Initialize additional rules on import
register_additional_rules()

print(f"Registered {len(rules)} ECDLP-specific Nsight rules")

# Initialize rules on import
register_ecdlp_rules()