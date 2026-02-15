            // Use GPU acceleration when available (hybrid backend with optimizations)
            let step_fut = async {
                if matches!(self.config.gpu_backend, crate::config::GpuBackend::Hybrid) {
                    // Use optimized GPU dispatch for hybrid backend
                    self.step_kangaroos_gpu(&kangaroos).await
                } else {
                    // Fall back to CPU stepping
                    self.stepper.borrow_mut().step_batch(&kangaroos, target_points.first())
                }
            };