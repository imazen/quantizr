use criterion::{black_box, criterion_group, criterion_main, Criterion};

// Simple LCG for reproducible pseudo-random noise
fn lcg_next(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1103515245).wrapping_add(12345);
    *state
}

fn bench_quantize_and_remap(c: &mut Criterion) {
    // Create test image: 512x512 with varied colors (gradient)
    let width = 512usize;
    let height = 512usize;
    let mut pixels = vec![0u8; width * height * 4];
    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) * 4;
            pixels[i] = ((x * 255) / width) as u8;     // R
            pixels[i + 1] = ((y * 255) / height) as u8; // G
            pixels[i + 2] = (((x + y) * 127) / (width + height)) as u8; // B
            pixels[i + 3] = 255; // A
        }
    }

    // Create noisy image: simulates photographic content with more unique colors
    let mut noisy_pixels = vec![0u8; width * height * 4];
    let mut rng_state = 0x12345678u32;
    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) * 4;
            // Base gradient + noise
            let noise_r = (lcg_next(&mut rng_state) % 64) as i16 - 32;
            let noise_g = (lcg_next(&mut rng_state) % 64) as i16 - 32;
            let noise_b = (lcg_next(&mut rng_state) % 64) as i16 - 32;
            noisy_pixels[i] = (((x * 255) / width) as i16 + noise_r).clamp(0, 255) as u8;
            noisy_pixels[i + 1] = (((y * 255) / height) as i16 + noise_g).clamp(0, 255) as u8;
            noisy_pixels[i + 2] = ((((x + y) * 127) / (width + height)) as i16 + noise_b).clamp(0, 255) as u8;
            noisy_pixels[i + 3] = 255;
        }
    }

    c.bench_function("quantize_512x512", |b| {
        b.iter(|| {
            let image = quantizr::Image::new(black_box(&pixels), width, height).unwrap();
            let mut opts = quantizr::Options::default();
            opts.set_max_colors(256).unwrap();
            let result = quantizr::QuantizeResult::quantize(&image, &opts);
            black_box(result)
        })
    });

    // Pre-quantize for remap benchmark
    let image = quantizr::Image::new(&pixels, width, height).unwrap();
    let mut opts = quantizr::Options::default();
    opts.set_max_colors(256).unwrap();
    let mut result = quantizr::QuantizeResult::quantize(&image, &opts);
    let mut output = vec![0u8; width * height];

    c.bench_function("remap_512x512", |b| {
        b.iter(|| {
            result.remap_image(black_box(&image), black_box(&mut output)).unwrap();
        })
    });

    // Noisy image benchmarks (simulates photographic content)
    c.bench_function("quantize_noisy_512x512", |b| {
        b.iter(|| {
            let image = quantizr::Image::new(black_box(&noisy_pixels), width, height).unwrap();
            let mut opts = quantizr::Options::default();
            opts.set_max_colors(256).unwrap();
            let result = quantizr::QuantizeResult::quantize(&image, &opts);
            black_box(result)
        })
    });

    let noisy_image = quantizr::Image::new(&noisy_pixels, width, height).unwrap();
    let mut noisy_result = quantizr::QuantizeResult::quantize(&noisy_image, &opts);

    c.bench_function("remap_noisy_512x512", |b| {
        b.iter(|| {
            noisy_result.remap_image(black_box(&noisy_image), black_box(&mut output)).unwrap();
        })
    });
}

criterion_group!(benches, bench_quantize_and_remap);
criterion_main!(benches);
