    async fn step_kangaroos_gpu(&self, kangaroos: &[KangarooState]) -> Result<Vec<KangarooState>> {
        println!("[GPU] Stepping {} kangaroos...", kangaroos.len());

        let hybrid = crate::gpu::hybrid_manager::HybridBackend::new()?;
        let result = hybrid.step_kangaroos(kangaroos, &self.config).await?;

        println!("[GPU] Stepped {} kangaroos", result.len());
        Ok(result)
    }