#!/usr/bin/env python3
"""
Bitcoin Puzzle Verification Script using Bitcoin Core RPC
Verifies puzzle data in puzzles.txt against blockchain data

Usage:
    python verify_puzzles_bitcoincore.py [--puzzle N] [--all]

Requirements:
    - Bitcoin Core running with RPC enabled
    - bitcoin-cli accessible
"""

import subprocess
import json
import sys
import argparse
from decimal import Decimal

class BitcoinRPC:
    def __init__(self, wallet_name="default_wallet"):
        self.cli_command = ["bitcoin-cli", "-rpcwallet=" + wallet_name]
        self.wallet_name = wallet_name

    def call(self, method, params=None):
        """Make RPC call to Bitcoin Core using bitcoin-cli"""
        if params is None:
            params = []

        cmd = self.cli_command + [method]
        if params:
            # Convert params to strings for command line
            for param in params:
                if isinstance(param, str):
                    cmd.append(param)
                else:
                    cmd.append(str(param))

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                try:
                    return json.loads(result.stdout.strip())
                except json.JSONDecodeError:
                    # Some commands return non-JSON output
                    return result.stdout.strip()
            else:
                print(f"bitcoin-cli error: {result.stderr.strip()}")
                return None
        except subprocess.TimeoutExpired:
            print("bitcoin-cli timeout")
            return None
        except Exception as e:
            print(f"Error running bitcoin-cli: {e}")
            return None

class PuzzleVerifier:
    def __init__(self, wallet_name="default_wallet"):
        self.rpc = BitcoinRPC(wallet_name)
        self.wallet_name = wallet_name
        self.puzzles = {}

    def load_puzzles(self, filename="puzzles.txt"):
        """Load puzzles from flat file"""
        print(f"Loading puzzles from {filename}...")
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    parts = line.split('|')
                    if len(parts) < 9:
                        continue

                    try:
                        puzzle_num = int(parts[0])
                        status = parts[1]
                        btc_reward = float(parts[2])
                        pubkey_hex = parts[3]
                        privkey_hex = parts[4] if parts[4] != '' else None
                        target_address = parts[5]
                        range_min_hex = parts[6]
                        range_max_hex = parts[7]
                        search_space_bits = float(parts[8])

                        self.puzzles[puzzle_num] = {
                            'status': status,
                            'btc_reward': btc_reward,
                            'pubkey_hex': pubkey_hex,
                            'privkey_hex': privkey_hex,
                            'target_address': target_address,
                            'range_min_hex': range_min_hex,
                            'range_max_hex': range_max_hex,
                            'search_space_bits': search_space_bits
                        }
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Error parsing line {line_num}: {e}")
                        continue

            print(f"Loaded {len(self.puzzles)} puzzles")
            return True
        except FileNotFoundError:
            print(f"Error: {filename} not found")
            return False

    def test_rpc_connection(self):
        """Test Bitcoin Core RPC connection"""
        print("Testing Bitcoin Core RPC connection...")
        info = self.rpc.call("getblockchaininfo")
        if info and isinstance(info, dict):
            print(f"✅ Connected to Bitcoin Core")
            print(f"   Network: {info.get('chain', 'unknown')}")
            print(f"   Blocks: {info.get('blocks', 'unknown')}")
            print(f"   Headers: {info.get('headers', 'unknown')}")
            print(f"   Verification progress: {info.get('verificationprogress', 0):.4f}")
            print(f"   Initial block download: {info.get('initialblockdownload', 'unknown')}")
            return True
        else:
            print("❌ Could not connect to Bitcoin Core")
            print("   Make sure Bitcoin Core is running with RPC enabled")
            return False

    def list_wallets(self):
        """List loaded wallets"""
        wallets = self.rpc.call("listwallets")
        return wallets or []

    def load_wallet(self, wallet_name="default_wallet"):
        """Load or create a wallet for verification"""
        wallets = self.list_wallets()
        if wallet_name in wallets:
            print(f"   📔 Wallet '{wallet_name}' already loaded")
            return True

        # Try to load the wallet
        result = self.rpc.call("loadwallet", [wallet_name])
        if result:
            print(f"   📔 Loaded wallet '{wallet_name}'")
            return True

        # Try to create a new wallet
        result = self.rpc.call("createwallet", [wallet_name])
        if result:
            print(f"   📔 Created new wallet '{wallet_name}'")
            return True

        print(f"   ❌ Could not load or create wallet '{wallet_name}'")
        return False

    def import_address_watchonly(self, address):
        """Import an address as watch-only"""
        result = self.rpc.call("importaddress", [address, "", False])
        return result is not None

    def verify_address_info(self, address):
        """Get address information from Bitcoin Core"""
        info = self.rpc.call("getaddressinfo", [address])
        return info

    def get_address_balance(self, address):
        """Get total received by address"""
        # Try wallet method first
        balance = self.rpc.call("getreceivedbyaddress", [address])
        if balance is not None:
            return balance

        # Fallback: scan blockchain (limited by Bitcoin Core's scanning capabilities)
        print("   ℹ️  Address not in wallet, cannot check balance without full blockchain scan")
        return None

    def get_address_utxos(self, address):
        """Get UTXOs for an address"""
        # Try wallet method first
        utxos = self.rpc.call("listunspent", [0, 9999999, [address]])
        if utxos is not None:
            return utxos

        # Fallback: Note that we can't scan without wallet
        print("   ℹ️  Address not in wallet, cannot check UTXOs")
        return []

    def scan_blockchain_for_address(self, address):
        """Scan blockchain for address activity (basic check)"""
        # This is limited in Bitcoin Core - we'd need to scan the entire blockchain
        # For now, just check if we can derive info from the blockchain
        print("   ℹ️  For full blockchain verification, the address would need to be imported as watch-only")

        # We could try to get the scriptPubKey and search for it, but that's complex
        # For now, just note the limitation
        return None

    def verify_puzzle_address(self, puzzle_num, puzzle_data):
        """Verify a single puzzle's address data"""
        address = puzzle_data['target_address']
        print(f"\n🔍 Verifying Puzzle #{puzzle_num}")
        print(f"   Address: {address}")

        # Load a wallet first (needed for address operations)
        if not self.load_wallet():
            print("   ⚠️  Continuing without wallet (limited verification)")

        # Import address as watch-only if not already known
        addr_info = self.verify_address_info(address)
        if not addr_info:
            print("   📥 Importing address as watch-only...")
            if self.import_address_watchonly(address):
                print("   ✅ Address imported successfully")
                # Try again
                addr_info = self.verify_address_info(address)
            else:
                print("   ❌ Failed to import address")

        if addr_info:
            print(f"   Script type: {addr_info.get('scriptPubKey', 'unknown')}")
            print(f"   Is mine: {addr_info.get('ismine', False)}")
            print(f"   Is watch-only: {addr_info.get('iswatchonly', False)}")

            # Check if solvable (should be for puzzle addresses)
            if addr_info.get('isscript', False):
                print("   ✅ Is script address (expected for puzzles)")
            elif addr_info.get('iscompressed', False):
                print("   ✅ Is compressed key address")
            else:
                print("   ⚠️  Address type unexpected")
        else:
            print("   ❌ Could not get address info even after import")

        # Check balance (this works even without wallet loaded)
        balance = self.get_address_balance(address)
        if balance is not None:
            expected_balance = puzzle_data['btc_reward']
            print(f"   💰 Balance: {balance} BTC (expected: {expected_balance})")
            if balance >= expected_balance:
                if balance == expected_balance:
                    print("   ✅ Balance matches expected reward exactly")
                else:
                    print("   ⚠️  Balance higher than expected reward")
            else:
                print("   ⚠️  Balance lower than expected reward")
                if balance == 0:
                    print("   📭 Address has never received funds")
        else:
            print("   ❌ Could not get balance")

        # Check for UTXOs
        utxos = self.get_address_utxos(address)
        if utxos:
            print(f"   🔗 Found {len(utxos)} UTXO(s)")
            total_value = sum(Decimal(str(utxo['amount'])) for utxo in utxos)
            print(f"   💎 Total UTXO value: {total_value} BTC")

            # Show UTXO details
            for utxo in utxos[:3]:
                print(f"      TXID: {utxo['txid'][:16]}...")
                print(f"      Amount: {utxo['amount']} BTC")
                print(f"      Confirmations: {utxo['confirmations']}")
        else:
            print("   📭 No UTXOs found")

        # Check if we have the private key and can verify
        if puzzle_data['privkey_hex']:
            print(f"   🔑 Has private key: {puzzle_data['privkey_hex'][:16]}...")
            print("   ✅ Private key available for verification")
        else:
            print("   🔒 No private key (unsolved puzzle)")

        return True

    def verify_puzzle(self, puzzle_num):
        """Verify a specific puzzle"""
        if puzzle_num not in self.puzzles:
            print(f"❌ Puzzle #{puzzle_num} not found in database")
            return False

        puzzle_data = self.puzzles[puzzle_num]
        return self.verify_puzzle_address(puzzle_num, puzzle_data)

    def verify_all_puzzles(self, max_puzzles=5):
        """Verify multiple puzzles (with limit for safety)"""
        print(f"Verifying up to {max_puzzles} puzzles...")

        solved_puzzles = [num for num, data in self.puzzles.items() if data['status'] == 'SOLVED']

        count = 0
        for puzzle_num in solved_puzzles:
            if count >= max_puzzles:
                print(f"\n⏹️  Stopped after {max_puzzles} puzzles (use --all with higher --max for more)")
                break

            self.verify_puzzle(puzzle_num)
            count += 1

    def show_stats(self):
        """Show statistics about loaded puzzles"""
        total = len(self.puzzles)
        solved = sum(1 for p in self.puzzles.values() if p['status'] == 'SOLVED')
        unsolved = total - solved

        print("\n📊 Puzzle Database Statistics:")
        print(f"   Total puzzles: {total}")
        print(f"   Solved: {solved}")
        print(f"   Unsolved: {unsolved}")
        print(f"   Average BTC reward: {sum(p['btc_reward'] for p in self.puzzles.values()) / total:.3f}")

def main():
    parser = argparse.ArgumentParser(description="Verify Bitcoin puzzles using Bitcoin Core RPC")
    parser.add_argument('--puzzle', '-p', type=int, help='Verify specific puzzle number')
    parser.add_argument('--all', action='store_true', help='Verify all solved puzzles (limited to 5 by default)')
    parser.add_argument('--max', type=int, default=5, help='Maximum puzzles to verify when using --all')
    parser.add_argument('--file', '-f', default='puzzles.txt', help='Puzzle database file')

    args = parser.parse_args()

    verifier = PuzzleVerifier()

    # Load puzzles
    if not verifier.load_puzzles(args.file):
        sys.exit(1)

    # Test RPC connection
    if not verifier.test_rpc_connection():
        print("\n❌ Bitcoin Core RPC not available")
        print("Manual verification steps:")
        print("1. Make sure Bitcoin Core is running")
        print("2. Check: bitcoin-cli getblockchaininfo")
        print("3. For specific address: bitcoin-cli getaddressinfo <address>")
        sys.exit(1)

    # Show stats
    verifier.show_stats()

    # Verify puzzles
    if args.puzzle:
        verifier.verify_puzzle(args.puzzle)
    elif args.all:
        verifier.verify_all_puzzles(args.max)
    else:
        print("\nUsage:")
        print("  --puzzle N    Verify puzzle #N")
        print("  --all         Verify all solved puzzles (limited)")
        print("\nExample:")
        print("  python verify_puzzles_bitcoincore.py --puzzle 35")

if __name__ == "__main__":
    main()