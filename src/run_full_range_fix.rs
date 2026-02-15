pub async fn run_full_range(config: &Config) -> Result<(), Box<dyn std::error::Error>> {
    println!("[LAUNCH] Herd size: {} | DP bits: {} | Near collisions: {:.2}",
             config.herd_size, config.dp_bits, config.enable_near_collisions);

    // ←←← REMOVE THIS ENTIRE EARLY RETURN BLOCK
    // if config.targets.is_empty() {
    //     println!("[ERROR] No targets loaded");
    //     return Ok(());
    // }

    // Automatic fallback (already good, just make sure it continues)
    let targets = if config.targets.exists() {
        config.targets.clone()
    } else {
        println!("[FALLBACK] Using valuable_p2pk_pubkeys.txt");
        std::path::PathBuf::from("valuable_p2pk_pubkeys.txt")
    };

    // Proceed to herd launch
    let mut manager = KangarooManager::new(config.clone())?;
    manager.start_jumps();                    // ← Jumps now fire
    manager.run_hunt().await?;

    Ok(())
}