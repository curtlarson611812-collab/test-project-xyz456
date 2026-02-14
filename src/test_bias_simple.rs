use std::fs;
use speedbitcrack::utils::bias::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Test with a few sample keys
    let test_keys = vec![
        "04fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f".to_string(),
        "04fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2e".to_string(),
        "04aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
    ];
    
    println!("Testing bias analysis with {} keys", test_keys.len());
    
    // Compute global stats for mod3
    let stats_mod3 = compute_global_stats(&test_keys, 3, 3)?;
    println!("Mod3 stats: chi={:.3}, bins={:?}", stats_mod3.chi, stats_mod3.bins);
    
    // Test individual key analysis
    for (i, key) in test_keys.iter().enumerate() {
        let x_hex = &key[2..]; // Remove 04 prefix for x coordinate
        let mod3_score = calculate_mod_bias(x_hex, &stats_mod3, 3, 3)?;
        println!("Key {}: mod3_score = {:.3}", i, mod3_score);
    }
    
    Ok(())
}
