        let types: Vec<u32> = kangaroos.iter()
            .map(|k| if k.is_tame { 1 } else { 0 })
            .collect();