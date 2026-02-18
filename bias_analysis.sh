#!/bin/bash

# SpeedBitCrack V3 - Bias Analysis Script
# Analyzes valuable_p2pk_pubkeys.txt to generate GOLD cluster and top 100 high-bias lists

set -e

echo "🔍 SpeedBitCrack V3 - Bias Analysis Tool"
echo "========================================"

# Check if valuable_p2pk_pubkeys.txt exists
if [ ! -f "valuable_p2pk_pubkeys.txt" ]; then
    echo "❌ Error: valuable_p2pk_pubkeys.txt not found in current directory"
    echo "Please ensure the valuable P2PK pubkeys file is present"
    exit 1
fi

# Check if cargo is available
if ! command -v cargo &> /dev/null; then
    echo "❌ Error: cargo not found. Please install Rust."
    exit 1
fi

echo "📊 Analyzing valuable_p2pk_pubkeys.txt for bias patterns..."
echo "This will generate:"
echo "  - valuable_p2pk_gold.txt: GOLD cluster addresses (mod81=0)"
echo "  - valuable_p2pk_top100.txt: Top 100 highest bias addresses"
echo "  - valuable_p2pk_high_priority.txt: All high-bias addresses"
echo ""

# Run the bias analysis using standalone analyzer
cd bias_analyzer && cargo run -- \
    --input ../valuable_p2pk_pubkeys.txt \
    --output ../valuable_p2pk_high_priority.txt && cd ..

echo ""
echo "✅ Bias analysis complete!"
echo ""
echo "📁 Generated files:"
if [ -f "valuable_p2pk_gold.txt" ]; then
    gold_count=$(wc -l < valuable_p2pk_gold.txt)
    echo "  🏆 valuable_p2pk_gold.txt: $((gold_count - 5)) GOLD cluster addresses"
fi

if [ -f "valuable_p2pk_top100.txt" ]; then
    top100_count=$(wc -l < valuable_p2pk_top100.txt)
    echo "  🥇 valuable_p2pk_top100.txt: $((top100_count - 5)) top bias addresses"
fi

if [ -f "valuable_p2pk_high_priority.txt" ]; then
    high_count=$(wc -l < valuable_p2pk_high_priority.txt)
    echo "  🎯 valuable_p2pk_high_priority.txt: $((high_count - 5)) high-priority addresses"
fi

echo ""
echo "💡 Use these lists with:"
echo "  --targets valuable_p2pk_gold.txt        # GOLD cluster solving"
echo "  --targets valuable_p2pk_top100.txt      # Top bias solving"
echo "  --targets valuable_p2pk_high_priority.txt # High priority solving"