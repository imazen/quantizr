use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_quantize_and_remap(c: &mut Criterion) {
    // Create test image: 512x512 with varied colors
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
}

criterion_group!(benches, bench_quantize_and_remap);
criterion_main!(benches);
