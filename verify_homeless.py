#!/usr/bin/env python3
"""
Verification script for homeless.txt puzzle database
Checks blockchain for addresses and extracts public keys where possible
"""

import subprocess
import json
import sys
import re

def run_bitcoin_cli(*args):
    """Run bitcoin-cli command and return parsed JSON"""
    try:
        cmd = ['bitcoin-cli'] + list(args)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"bitcoin-cli error: {e}")
        return None
    except json.JSONDecodeError:
        return None

def parse_puzzle_line(line):
    """Parse a line from homeless.txt format"""
    # Format: No. | PRIVATE KEY | ADDRESS | UPPER RANGE | PUBLIC KEY | SOLVED DATE
    parts = line.split('|')
    if len(parts) != 6:
        return None

    try:
        puzzle_num = int(parts[0].strip())
        priv_key = parts[1].strip()
        address = parts[2].strip()
        upper_range = parts[3].strip()
        pub_key = parts[4].strip()
        solved_date = parts[5].strip()

        return {
            'num': puzzle_num,
            'priv_key': priv_key if priv_key != '========================== U N K N O W N =========================' and priv_key.strip() else '',
            'address': address,
            'upper_range': upper_range,
            'pub_key': pub_key if pub_key != '========================== U N K N O W N =========================' and pub_key.strip() else '',
            'solved_date': solved_date if solved_date != '____-__-__' else ''
        }
    except (ValueError, IndexError):
        return None

def verify_address_on_blockchain(address):
    """Check if address exists and get balance"""
    try:
        # Get address info
        addr_info = run_bitcoin_cli('getaddressinfo', address)
        if not addr_info or 'address' not in addr_info:
            return False, 0, None

        # Get balance (received transactions) - these are puzzle addresses, likely empty
        balance = run_bitcoin_cli('getreceivedbyaddress', address, '0')
        balance = balance if isinstance(balance, (int, float)) else 0

        return True, balance, addr_info
    except:
        return False, 0, None

def extract_pubkey_from_address(address):
    """Try to extract public key from blockchain if address has been used"""
    try:
        # Get UTXOs for this address
        utxos = run_bitcoin_cli('listunspent', 0, 9999999, '["' + address + '"]')
        if not utxos:
            return None

        # Get the first transaction that spent from this address
        for utxo in utxos[:1]:  # Check first UTXO
            txid = utxo['txid']
            vout = utxo['vout']

            # Get raw transaction
            raw_tx = run_bitcoin_cli('getrawtransaction', txid)
            if not raw_tx:
                continue

            # Decode transaction
            decoded = run_bitcoin_cli('decoderawtransaction', raw_tx)
            if not decoded or 'vin' not in decoded:
                continue

            # Look for transactions spending from this address
            for vin in decoded['vin']:
                if 'txid' in vin and 'vout' in vin:
                    # This is a spending transaction, get the input
                    input_tx = run_bitcoin_cli('getrawtransaction', vin['txid'])
                    if input_tx:
                        input_decoded = run_bitcoin_cli('decoderawtransaction', input_tx)
                        if input_decoded and 'vout' in input_decoded:
                            for vout_info in input_decoded['vout']:
                                if vout_info['n'] == vin['vout']:
                                    # Found the output that was spent
                                    scriptPubKey = vout_info.get('scriptPubKey', {})
                                    if scriptPubKey.get('type') == 'pubkeyhash':
                                        # This is a P2PKH output, we can't extract pubkey from address alone
                                        # We'd need the actual spending transaction signature
                                        return None

        return None
    except:
        return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 verify_homeless.py <homeless.txt>")
        sys.exit(1)

    filename = sys.argv[1]

    print("🔍 Verifying homeless.txt Bitcoin Puzzle Database")
    print(f"📄 Reading from: {filename}")
    print()

    total_puzzles = 0
    solved_puzzles = 0
    revealed_pubkeys = 0
    valid_addresses = 0
    cashed_out = 0

    try:
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or not line[0].isdigit():
                    continue

                puzzle = parse_puzzle_line(line)
                if not puzzle:
                    continue

                total_puzzles += 1
                n = puzzle['num']

                print(f"🔍 Puzzle #{n}")

                # Check if solved (has private key)
                is_solved = bool(puzzle['priv_key'])
                if is_solved:
                    solved_puzzles += 1
                    print(f"  🔑 SOLVED ({puzzle['solved_date']})")

                # Check if has revealed public key
                has_pubkey = bool(puzzle['pub_key'])
                if has_pubkey:
                    revealed_pubkeys += 1
                    print(f"  🔓 PUBLIC KEY REVEALED")

                # Verify address on blockchain
                valid, balance, addr_info = verify_address_on_blockchain(puzzle['address'])

                if valid:
                    valid_addresses += 1
                    if balance > 0:
                        cashed_out += 1
                        print(f"  ✅ Valid address with balance: {balance:.8f} BTC")
                    else:
                        print("  ✅ Valid address (empty)")
                else:
                    print(f"  ❌ Invalid address: {puzzle['address']}")

                # For every 5th puzzle, try to extract pubkey from blockchain
                if n % 5 == 0:
                    print(f"  🔍 Checking every 5th puzzle - extracting pubkey...")
                    extracted_pubkey = extract_pubkey_from_address(puzzle['address'])
                    if extracted_pubkey:
                        print(f"  🎯 Extracted pubkey: {extracted_pubkey}")
                        if not has_pubkey:
                            print("  ⚠️  Pubkey not in database but found on blockchain!")
                    else:
                        print("  ℹ️  Could not extract pubkey from blockchain")

                print()

        print("📊 Homeless.txt Verification Summary:")
        print(f"  Total puzzles: {total_puzzles}")
        print(f"  Solved puzzles: {solved_puzzles}")
        print(f"  Revealed pubkeys: {revealed_pubkeys}")
        print(f"  Valid addresses: {valid_addresses}")
        print(f"  Cashed out addresses: {cashed_out}")

        if valid_addresses == total_puzzles:
            print("✅ All puzzle addresses verified on blockchain!")
        else:
            print(f"❌ {total_puzzles - valid_addresses} addresses not found on blockchain")

        if cashed_out > 0:
            print(f"💰 {cashed_out} puzzles have been cashed out!")

    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())