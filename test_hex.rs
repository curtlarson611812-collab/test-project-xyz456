use hex;

fn main() {
    let hex_str = "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798";
    match hex::decode(hex_str) {
        Ok(bytes) => {
            println!("Hex decoded successfully: {} bytes", bytes.len());
            println!("First few bytes: {:?}", &bytes[..10]);
            if bytes.len() == 33 {
                println!("Correct length for compressed secp256k1 point");
            } else {
                println!("Wrong length: expected 33, got {}", bytes.len());
            }
        }
        Err(e) => {
            println!("Hex decode failed: {}", e);
        }
    }
}
