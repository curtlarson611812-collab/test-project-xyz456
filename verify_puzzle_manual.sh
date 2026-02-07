#!/bin/bash
# Manual Bitcoin Puzzle Verification Script
# Run Bitcoin Core commands to verify puzzle data

PUZZLE_NUM=${1:-35}

echo "🔍 Manual Verification for Bitcoin Puzzle #$PUZZLE_NUM"
echo "=================================================="

# Extract puzzle data from puzzles.txt
PUZZLE_LINE=$(grep "^${PUZZLE_NUM}|" puzzles.txt)
if [ -z "$PUZZLE_LINE" ]; then
    echo "❌ Puzzle #$PUZZLE_NUM not found in puzzles.txt"
    exit 1
fi

echo "📄 Puzzle data: $PUZZLE_LINE"

# Parse the fields (split by |)
IFS='|' read -r num status btc_reward pubkey_hex privkey_hex target_address range_min range_max search_bits <<< "$PUZZLE_LINE"

echo ""
echo "📊 Parsed Data:"
echo "   Status: $status"
echo "   BTC Reward: $btc_reward"
echo "   Target Address: $target_address"
echo "   Has Private Key: $([ -n "$privkey_hex" ] && echo 'YES' || echo 'NO')"
echo "   Search Space: 2^$search_bits operations"

echo ""
echo "🔧 Bitcoin Core RPC Commands to Verify:"
echo "========================================"

echo "1. Check if Bitcoin Core is running:"
echo "   bitcoin-cli getblockchaininfo"
echo ""

echo "2. Get address information:"
echo "   bitcoin-cli getaddressinfo $target_address"
echo ""

echo "3. Check for UTXOs (unspent outputs):"
echo "   bitcoin-cli listunspent 0 9999999 '[\"$target_address\"]'"
echo ""

echo "4. Get address balance:"
echo "   bitcoin-cli getreceivedbyaddress $target_address"
echo ""

if [ -n "$privkey_hex" ]; then
    echo "5. Import private key for verification (CAUTION - only for testing):"
    echo "   bitcoin-cli importprivkey $privkey_hex \"\" false"
    echo ""
    echo "6. Verify the imported key matches the address:"
    echo "   bitcoin-cli getaddressesbylabel \"\""
    echo ""
fi

echo "7. Decode the public key from hex:"
echo "   python3 -c \"import binascii; print(binascii.hexlify(binascii.unhexlify('$pubkey_hex')).decode())\""
echo ""

echo "🎯 Expected Results:"
echo "==================="
if [ "$status" = "SOLVED" ]; then
    echo "✅ Address should have transaction history"
    echo "✅ Address should contain $btc_reward BTC"
    echo "✅ Public key should be extractable from funding transaction"
    if [ -n "$privkey_hex" ]; then
        echo "✅ Private key should generate the correct address"
    fi
else
    echo "❓ Address may or may not have transactions"
    echo "❓ If funded, should contain puzzle reward BTC"
fi

echo ""
echo "🔗 Useful Links:"
echo "==============="
echo "Bitcoin Puzzle Challenge: https://privatekeys.pw/puzzles/bitcoin-puzzle"
echo "Puzzle #$PUZZLE_NUM: https://privatekeys.pw/puzzles/bitcoin-puzzle-$PUZZLE_NUM"