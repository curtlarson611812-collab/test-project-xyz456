#!/usr/bin/env python3
"""
Custom Nsight Compute Rule for Modular Arithmetic Efficiency in ECDLP

This rule analyzes the efficiency of modular arithmetic operations in kangaroo algorithm,
specifically focusing on Barrett reduction and bias calculation kernels.
"""

from nsight import *


class ModularEfficiencyRule(Rule):
    """
    Custom rule for analyzing modular arithmetic efficiency in ECDLP operations.

    Monitors ALU utilization for modular operations and suggests optimizations
    when modular arithmetic becomes a bottleneck (>10% of total ALU cycles).
    """
    id = "ModularEfficiency"
    name = "ECDLP Modular Arithmetic Efficiency"
    description = "Analyzes efficiency of modular operations in bias calculations"

    def compute(self, metrics):
        # Check ALU utilization for modular operations
        alu_cycles = metrics.get("sm__pipe_alu_cycles_active.average.pct_of_peak_sustained_active", 0)
        total_cycles = metrics.get("sm__cycles_active.avg.pct_of_peak_sustained_elapsed", 100)

        if total_cycles > 0:
            mod_efficiency_ratio = alu_cycles / total_cycles

            # If modular operations take >10% of total cycles, suggest optimization
            if mod_efficiency_ratio > 0.1:
                return Suggestion(
                    f"High modular ALU utilization ({alu_cycles:.1f}%)",
                    "Optimize Barrett reduction: use precomputed constants in shared memory, fuse modular operations in bias_check_kernel.cu"
                )
            elif mod_efficiency_ratio < 0.05:
                return Suggestion(
                    f"Low modular ALU utilization ({alu_cycles:.1f}%) - may indicate memory bottleneck",
                    "Consider increasing modular operation parallelism or check for memory stalls in bias calculations"
                )

        return Pass()


class MemoryCoalescingEfficiencyRule(Rule):
    """
    Custom rule for BigInt256 memory coalescing efficiency.

    Analyzes global memory access patterns for 256-bit integer operations
    and suggests Struct-of-Arrays (SoA) layout optimizations.
    """
    id = "BigInt256Coalescing"
    name = "BigInt256 Memory Coalescing Efficiency"
    description = "Analyzes coalescing efficiency for 256-bit integer operations"

    def compute(self, metrics):
        # Check average bytes per memory sector
        bytes_per_sector = metrics.get("sm__sass_average_data_bytes_per_sector_mem_global_op_ld", 0)

        # For BigInt256 (32 bytes), optimal coalescing gives ~32 bytes/sector
        if bytes_per_sector < 16:  # Less than half optimal
            return Suggestion(
                f"Poor BigInt256 coalescing ({bytes_per_sector:.1f} bytes/sector)",
                "Implement Struct-of-Arrays (SoA) layout: separate x_limbs[], y_limbs[], dist_limbs[] arrays in rho_kernel.cu for better coalescing"
            )
        elif bytes_per_sector < 24:  # Less than 3/4 optimal
            return Suggestion(
                f"Moderate BigInt256 coalescing ({bytes_per_sector:.1f} bytes/sector)",
                "Consider SoA layout optimization for marginal coalescing improvement"
            )

        return Pass()


class SharedMemoryBankConflictRule(Rule):
    """
    Custom rule for shared memory bank conflict detection in bias tables.

    Monitors shared memory access patterns for bias table lookups and
    suggests padding or access pattern optimizations.
    """
    id = "BiasTableBankConflicts"
    name = "Bias Table Shared Memory Bank Conflicts"
    description = "Analyzes shared memory access patterns for bias table operations"

    def compute(self, metrics):
        # Check for bank conflicts in shared memory loads
        bank_conflicts = metrics.get("sm__sass_average_bank_conflicts_pipe_lsu_mem_shared_op_ld", 0)

        if bank_conflicts > 2:  # Significant conflicts
            return Suggestion(
                f"High shared memory bank conflicts ({bank_conflicts:.1f} avg)",
                "Optimize bias_table access in bias_check_kernel.cu: use padding (add 31 elements) or change access pattern to avoid stride conflicts"
            )
        elif bank_conflicts > 0.5:  # Moderate conflicts
            return Suggestion(
                f"Moderate shared memory bank conflicts ({bank_conflicts:.1f} avg)",
                "Consider bias_table access pattern optimization to reduce bank conflicts"
            )

        return Pass()


class DivergenceAnalysisRule(Rule):
    """
    Custom rule for warp divergence in modular residue calculations.

    Analyzes control flow divergence in bias and modular operations
    and suggests subgroup optimizations.
    """
    id = "ModularDivergence"
    name = "Modular Operation Warp Divergence"
    description = "Analyzes control flow divergence in modular arithmetic"

    def compute(self, metrics):
        # Check warp execution efficiency
        warp_efficiency = metrics.get("warp_nonpred_execution_efficiency", 100)

        if warp_efficiency < 80:  # Significant divergence
            return Suggestion(
                f"High warp divergence in modular ops ({warp_efficiency:.1f}%)",
                "Use subgroup operations (__shfl_sync) for bias residue calculations to reduce control flow divergence"
            )
        elif warp_efficiency < 95:  # Moderate divergence
            return Suggestion(
                f"Moderate warp divergence ({warp_efficiency:.1f}%)",
                "Consider optimizing conditional branches in modular arithmetic operations"
            )

        return Pass()


# Register custom rules
register_rule(ModularEfficiencyRule)
register_rule(MemoryCoalescingEfficiencyRule)
register_rule(SharedMemoryBankConflictRule)
register_rule(DivergenceAnalysisRule)

if __name__ == "__main__":
    print("SpeedBitCrackV3 Custom Modular Efficiency Rules Loaded")
    print("Available rules:")
    for rule_class in [ModularEfficiencyRule, MemoryCoalescingEfficiencyRule,
                      SharedMemoryBankConflictRule, DivergenceAnalysisRule]:
        rule = rule_class()
        print(f"  - {rule.id}: {rule.name}")
        print(f"    {rule.description}")