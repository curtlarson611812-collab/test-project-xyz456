use hex;
use k256::{elliptic_curve::sec1::ToEncodedPoint, PublicKey, SecretKey};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Puzzle #64 private key: 2^63
    let priv_hex = "8000000000000000000000000000000000000000000000000000000000000000";

    println!("🔍 Calculating correct pubkey for Puzzle #64 (priv = 2^63)...");

    // Parse private key
    let priv_bytes = hex::decode(priv_hex)?;
    let secret_key = SecretKey::from_slice(&priv_bytes)?;

    // Derive public key
    let public_key = secret_key.public_key();

    // Get compressed encoding
    let compressed = public_key.to_encoded_point(true); // true for compressed
    let pub_hex = hex::encode(compressed.as_bytes());

    println!("Private key: {}", priv_hex);
    println!("Compressed pubkey: {}", pub_hex);
    println!("Length: {} (expected: 66)", pub_hex.len());

    Ok(())
}