use crate::cluster::Cluster;
use crate::colormap::Colormap;
use crate::error::Error;
use crate::histogram::Histogram;
use crate::image::Image;
use crate::options::Options;
use crate::palette::Palette;

const EMPTY_PIX: [u8; 4] = [0; 4];

/// Per-pixel color cache indexed by quantized color key.
/// Uses 15-bit key (5 bits per RGB channel) as direct index into a 32K table.
/// 32K × 22 bytes ≈ 720KB - fits in L2 cache, gives zero-collision direct mapping.
const CACHE_BITS: u32 = 6; // bits per channel (4-unit buckets)
const CACHE_SHIFT: u32 = 8 - CACHE_BITS; // shift to extract
const CACHE_SIZE: usize = 1 << (CACHE_BITS * 3); // 262144

#[derive(Clone, Copy)]
struct CacheEntry {
    /// Palette index result (0xFF = not populated)
    idx: u8,
    /// Palette color for error computation
    color: [f32; 4],
    /// Squared distance from cell center to second-nearest palette entry.
    /// If query distance to best exceeds half this, the cell might be wrong.
    half_gap_sq: f32,
}

impl Default for CacheEntry {
    fn default() -> Self {
        Self { idx: 0xFF, color: [0.0; 4], half_gap_sq: 0.0 }
    }
}

/// Direct-mapped cache index from color.
#[inline(always)]
fn cache_index(color: &[f32; 4]) -> usize {
    let r = (color[0].clamp(0.0, 255.0) as u32) >> CACHE_SHIFT;
    let g = (color[1].clamp(0.0, 255.0) as u32) >> CACHE_SHIFT;
    let b = (color[2].clamp(0.0, 255.0) as u32) >> CACHE_SHIFT;
    (r | (g << CACHE_BITS) | (b << (CACHE_BITS * 2))) as usize
}

// Result of quantization
pub struct QuantizeResult {
    error: f32,
    dithering_level: f32,
    colormap: Colormap,
}

impl QuantizeResult {
    /// Quantizes the provided [`Image`]
    pub fn quantize(image: &Image, attr: &Options) -> Self {
        let mut hist = Histogram::new();
        hist.add_image(image);

        Self::quantize_histogram(&hist, attr)
    }

    /// Quantizes the provided [`Histogram`]
    pub fn quantize_histogram(hist: &Histogram, attr: &Options) -> Self {
        let max_colors = attr.get_max_colors() as usize;

        let colormap = if hist.map.len() <= max_colors {
            Colormap::from_histogram(hist)
        } else {
            let root = Cluster::from_histogram(hist);
            let clusters = root.split_into(max_colors);

            Colormap::from_clusters(&clusters)
        };

        Self {
            error: colormap.error,
            colormap,
            dithering_level: 1.0,
        }
    }

    /// Sets the dithering level.
    ///
    /// Returns [`Error::ValueOutOfRange`] if the provided value is greater
    /// than 1.0 or lesser than 0.0
    pub fn set_dithering_level(&mut self, level: f32) -> Result<(), Error> {
        if !(0.0..=1.0).contains(&level) {
            return Err(Error::ValueOutOfRange);
        }

        self.dithering_level = level;

        Ok(())
    }

    /// Returns quantization error. The lesser the error the better the image
    /// was quantized
    pub fn get_error(&self) -> f32 {
        self.error
    }

    /// Returns the [`Palette`] generated after quantization
    pub fn get_palette(&self) -> &Palette {
        self.colormap.get_palette()
    }

    /// Remaps the proxided [`Image`] to a slize of bytes.
    ///
    /// Returns [`Error::BufferTooSmall`] if the provided buffer is smaller
    /// than `image.width * image.height`
    pub fn remap_image(&self, image: &Image, buf: &mut [u8]) -> Result<(), Error> {
        if buf.len() < image.width * image.height {
            return Err(Error::BufferTooSmall);
        }

        if self.dithering_level > 0.0 {
            self.remap_image_dither(image, buf);
        } else {
            self.remap_image_no_dither(image, buf);
        }

        Ok(())
    }

    fn remap_image_no_dither(&self, image: &Image, buf: &mut [u8]) {
        #[allow(clippy::needless_range_loop)]
        for point in 0..image.width * image.height {
            let data_point = point * 4;

            let pix = pix_or_empty(&image.data[data_point..data_point + 4]);
            let r = pix[0] as f32;
            let g = pix[1] as f32;
            let b = pix[2] as f32;
            let a = pix[3] as f32;

            let (ind, _, _) = self.colormap.nearest_ind(&[r, g, b, a]);

            buf[point] = ind;
        }
    }

    fn remap_image_dither(&self, image: &Image, buf: &mut [u8]) {
        let error_size = image.width + 2;
        let mut error_curr = vec![[0f32; 4]; error_size];
        let mut error_next = vec![[0f32; 4]; error_size];

        let dithering_coeff = self.dithering_level * 15.0 / 16.0 / 16.0;
        let err_threshold = self.error;

        let mut x_reverse = true;
        // Direct-mapped color cache: 32K entries, no collisions
        let mut cache = vec![CacheEntry::default(); CACHE_SIZE];

        for y in 0..image.height {
            x_reverse = !x_reverse;

            for xx in 0..image.width {
                let x = if x_reverse { image.width - 1 - xx } else { xx };

                let point = image.width * y + x;
                let data_point = point * 4;

                let err_ind = x + 1;
                let err_inds = if x_reverse {
                    (err_ind + 1, err_ind, err_ind - 1)
                } else {
                    (err_ind - 1, err_ind, err_ind + 1)
                };

                let err_pix = &mut error_curr[err_ind];

                let err_total = err_pix[0] * err_pix[0]
                    + err_pix[1] * err_pix[1]
                    + err_pix[2] * err_pix[2]
                    + err_pix[3] * err_pix[3];

                if err_total > err_threshold {
                    err_pix[0] *= 0.8;
                    err_pix[1] *= 0.8;
                    err_pix[2] *= 0.8;
                    err_pix[3] *= 0.8;
                }

                let pix = pix_or_empty(&image.data[data_point..data_point + 4]);
                let dith_pix = [
                    pix[0] as f32 + err_pix[0],
                    pix[1] as f32 + err_pix[1],
                    pix[2] as f32 + err_pix[2],
                    pix[3] as f32 + err_pix[3],
                ];

                // Try direct-mapped cache first
                let ci = cache_index(&dith_pix);
                let cached = cache[ci];
                let (ind, pal_pix) = if cached.idx != 0xFF {
                    // Cache populated. Verify: compute distance to cached entry.
                    // If close enough relative to the gap to 2nd-nearest, accept.
                    let dr = dith_pix[0] - cached.color[0];
                    let dg = dith_pix[1] - cached.color[1];
                    let db = dith_pix[2] - cached.color[2];
                    let da = dith_pix[3] - cached.color[3];
                    let dist_sq = dr * dr + dg * dg + db * db + da * da;
                    if dist_sq < cached.half_gap_sq {
                        (cached.idx, cached.color)
                    } else {
                        // Near boundary: full search
                        let (ind, pal_pix, _) = self.colormap.nearest_ind(&dith_pix);
                        (ind, pal_pix)
                    }
                } else {
                    let (ind, pal_pix, _) = self.colormap.nearest_ind(&dith_pix);
                    // Compute gap to 2nd nearest for this cell center
                    let half_gap_sq = self.colormap.second_nearest_dist_sq(ind, &dith_pix) * 0.25;
                    cache[ci] = CacheEntry { idx: ind, color: pal_pix, half_gap_sq };
                    (ind, pal_pix)
                };

                buf[point] = ind;

                let mut err_r = dith_pix[0] - pal_pix[0];
                let mut err_g = dith_pix[1] - pal_pix[1];
                let mut err_b = dith_pix[2] - pal_pix[2];
                let mut err_a = dith_pix[3] - pal_pix[3];

                let err_total = err_r * err_r + err_g * err_g + err_b * err_b + err_a * err_a;
                if err_total > err_threshold {
                    err_r *= 0.75;
                    err_g *= 0.75;
                    err_b *= 0.75;
                    err_a *= 0.75;
                }

                err_r *= dithering_coeff;
                err_g *= dithering_coeff;
                err_b *= dithering_coeff;
                err_a *= dithering_coeff;

                let err = &mut error_next[err_inds.0];
                err[0] += err_r * 3.0;
                err[1] += err_g * 3.0;
                err[2] += err_b * 3.0;
                err[3] += err_a * 3.0;

                let err = &mut error_next[err_inds.1];
                err[0] += err_r * 5.0;
                err[1] += err_g * 5.0;
                err[2] += err_b * 5.0;
                err[3] += err_a * 5.0;

                let err = &mut error_next[err_inds.2];
                err[0] += err_r * 1.0;
                err[1] += err_g * 1.0;
                err[2] += err_b * 1.0;
                err[3] += err_a * 1.0;

                let err = &mut error_curr[err_inds.2];
                err[0] += err_r * 7.0;
                err[1] += err_g * 7.0;
                err[2] += err_b * 7.0;
                err[3] += err_a * 7.0;
            }

            std::mem::swap(&mut error_curr, &mut error_next);
            error_next.fill_with(|| [0f32; 4]);
        }
    }
}

#[inline(always)]
fn pix_or_empty(pix: &[u8]) -> &[u8] {
    if pix[3] == 0 {
        return &EMPTY_PIX;
    }
    pix
}
