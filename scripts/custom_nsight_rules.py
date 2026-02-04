#!/usr/bin/env python3
"""
Custom Nsight Compute Rules for SpeedBitCrackV3 ECDLP Optimizations

These rules provide domain-specific analysis for elliptic curve discrete logarithm
problems, focusing on kangaroo algorithm optimizations and bias exploitation.
"""

import nsight


class EcdlpBiasEfficiencyRule(nsight.Rule):
    """
    Custom rule for analyzing bias exploitation efficiency in ECDLP kangaroo algorithms.

    Checks if Barrett reduction operations for bias calculations are optimally fused
    and if modular arithmetic is efficiently implemented.
    """
    id = "EcdlpBiasEfficiency"
    name = "ECDLP Bias Exploitation Efficiency"
    description = "Analyzes efficiency of bias-based optimizations in kangaroo algorithm"

    def compute(self, metrics):
        # Check if ALU utilization is high but IPC is low (indicates memory stalls in bias calc)
        alu_util = metrics.get('sm__pipe_alu_cycles_active.average.pct_of_peak_sustained_active', 0)
        ipc = metrics.get('sm__inst_executed.avg.pct_of_peak_sustained_active', 0)

        if alu_util > 80 and ipc < 70:
            return nsight.Suggestion(
                "High ALU but low IPC in bias calculations",
                "Fuse Barrett reduction operations in bias_check_kernel.cu to reduce memory stalls"
            )

        return nsight.Pass()


class EcdlpMemoryCoalescingRule(nsight.Rule):
    """
    Custom rule for analyzing memory coalescing in kangaroo state management.

    Detects if BigInt256 operations are causing uncoalesced memory access patterns
    and suggests Struct-of-Arrays (SoA) layout optimizations.
    """
    id = "EcdlpMemoryCoalescing"
    name = "ECDLP Memory Coalescing Analysis"
    description = "Checks memory access patterns for BigInt256 operations"

    def compute(self, metrics):
        # Check global memory sector efficiency
        sector_efficiency = metrics.get('sm__sass_average_data_bytes_per_sector_mem_global_op_ld', 0)

        if sector_efficiency < 4.0:  # Less than 4 bytes per sector indicates poor coalescing
            return nsight.Suggestion(
                "Poor memory coalescing detected in BigInt256 operations",
                "Convert Array-of-Structs (AoS) to Struct-of-Arrays (SoA) layout: separate x_limbs[], y_limbs[], dist_limbs[] arrays in rho_kernel.cu"
            )

        return nsight.Pass()


class EcdlpSharedMemoryUtilizationRule(nsight.Rule):
    """
    Custom rule for analyzing shared memory utilization in bias table operations.

    Ensures bias tables are efficiently cached in shared memory to avoid
    redundant global memory accesses.
    """
    id = "EcdlpSharedMemoryUtilization"
    name = "ECDLP Shared Memory Bias Table Analysis"
    description = "Analyzes shared memory usage for bias table operations"

    def compute(self, metrics):
        # Check for bank conflicts in shared memory operations
        bank_conflicts = metrics.get('sm__sass_average_bank_conflicts_pipe_lsu_mem_shared_op_ld', 0)

        if bank_conflicts > 0:
            return nsight.Suggestion(
                "Shared memory bank conflicts detected",
                "Optimize bias_table access pattern in bias_check_kernel.cu - ensure stride-1 access and avoid conflicts"
            )

        # Check if shared memory is underutilized
        shared_util = metrics.get('sm__sass_average_data_bytes_per_sector_mem_shared_op_ld', 0)
        if shared_util < 2.0:  # Less than 64-bit words per access
            return nsight.Suggestion(
                "Underutilized shared memory",
                "Load bias_table into shared memory in bias_check_kernel.cu to reduce global memory pressure"
            )

        return nsight.Pass()


class EcdlpDivergenceAnalysisRule(nsight.Rule):
    """
    Custom rule for analyzing control flow divergence in modular arithmetic operations.

    Detects if conditional branches in bias residue calculations are causing
    significant warp divergence.
    """
    id = "EcdlpDivergenceAnalysis"
    name = "ECDLP Control Flow Divergence Analysis"
    description = "Analyzes warp divergence in bias and modular operations"

    def compute(self, metrics):
        # Check warp execution efficiency
        warp_eff = metrics.get('warp_nonpred_execution_efficiency', 100)

        if warp_eff < 90:
            return nsight.Suggestion(
                "High control flow divergence in modular operations",
                "Use subgroup operations (__shfl_sync) for bias residue calculations to reduce warp divergence"
            )

        return nsight.Pass()


class EcdlpL1CacheOptimizationRule(nsight.Rule):
    """
    Custom rule for analyzing L1 cache utilization in elliptic curve operations.

    Suggests optimal L1 cache configuration for BigInt256 arithmetic operations.
    """
    id = "EcdlpL1CacheOptimization"
    name = "ECDLP L1 Cache Optimization"
    description = "Analyzes L1 cache utilization for EC arithmetic"

    def compute(self, metrics):
        # Check L1 hit rate
        l1_hit_rate = metrics.get('l1tex__t_bytes_hit_rate', 0)

        if l1_hit_rate < 80:
            return nsight.Suggestion(
                "Low L1 cache utilization",
                "Set CUDA cache config to PreferL1 for local variables in BigInt256 operations"
            )

        return nsight.Pass()


class EcdlpOccupancyOptimizationRule(nsight.Rule):
    """
    Custom rule for analyzing GPU occupancy in relation to ECDLP workload characteristics.

    Considers the memory-intensive nature of BigInt256 operations and suggests
    optimal occupancy targets.
    """
    id = "EcdlpOccupancyOptimization"
    name = "ECDLP Occupancy Optimization"
    description = "Analyzes occupancy considering ECDLP memory patterns"

    def compute(self, metrics):
        occupancy = metrics.get('achieved_occupancy', 0)
        register_usage = metrics.get('register_usage', 0)

        if occupancy < 60 and register_usage > 64:
            return nsight.Suggestion(
                "Low occupancy due to high register pressure",
                "Reduce register usage in BigInt256 operations by using shared memory for constants and minimizing local variables"
            )

        if occupancy > 80:
            return nsight.Suggestion(
                "Very high occupancy may indicate memory-bound workload",
                "Consider reducing block size to improve L2 cache utilization for BigInt256 operations"
            )

        return nsight.Pass()


# Register custom rules
nsight.register_rule(EcdlpBiasEfficiencyRule)
nsight.register_rule(EcdlpMemoryCoalescingRule)
nsight.register_rule(EcdlpSharedMemoryUtilizationRule)
nsight.register_rule(EcdlpDivergenceAnalysisRule)
nsight.register_rule(EcdlpL1CacheOptimizationRule)
nsight.register_rule(EcdlpOccupancyOptimizationRule)

if __name__ == "__main__":
    print("SpeedBitCrackV3 Custom Nsight Rules Loaded")
    print("Available custom rules:")
    for rule in [EcdlpBiasEfficiencyRule, EcdlpMemoryCoalescingRule, EcdlpSharedMemoryUtilizationRule,
                 EcdlpDivergenceAnalysisRule, EcdlpL1CacheOptimizationRule, EcdlpOccupancyOptimizationRule]:
        print(f"  - {rule.id}: {rule.name}")