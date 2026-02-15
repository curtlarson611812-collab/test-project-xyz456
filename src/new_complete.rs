    pub fn new(config: Config) -> Result<Self> {
        // Load targets properly
        use crate::targets::TargetLoader;
        let target_loader = TargetLoader::new();
        let targets = target_loader.load_targets(&config)?;
        info!("Loaded {} targets", targets.len());

        // Initialize components
        let dp_table = Arc::new(Mutex::new(DpTable::new(config.dp_bits)));
        let bloom = None;
        let gpu_backend: Box<dyn GpuBackend> = Box::new(CpuBackend::new()?);
        let generator = KangarooGenerator::new(&config);
        let stepper = std::cell::RefCell::new(KangarooStepper::with_dp_bits(false, config.dp_bits));
        let collision_detector = CollisionDetector::new();
        let parity_checker = ParityChecker::new();

        Ok(KangarooManager {
            config,
            search_config: SearchConfig::default(),
            targets,
            multi_targets: Vec::new(),
            wild_states: Vec::new(),
            tame_states: Vec::new(),
            dp_table,
            bloom,
            gpu_backend,
            generator,
            stepper,
            collision_detector,
            parity_checker,
            total_ops: 0,
            current_steps: 0,
            start_time: std::time::Instant::now(),
        })
    }