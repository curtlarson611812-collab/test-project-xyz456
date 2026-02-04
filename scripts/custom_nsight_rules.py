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

    print(f"Registered {len(rules)} ECDLP-specific Nsight rules")

# Initialize rules on import
register_ecdlp_rules()