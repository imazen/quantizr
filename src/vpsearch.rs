use crate::ord_float::OrdFloat32;

#[derive(Clone)]
struct SearchIdx {
    ind: u8,
    data: [f32; 4],
}

struct SearchVisitor<'a> {
    ind: Option<&'a SearchIdx>,
    distance: f32,
    distance_sq: f32,
}

impl<'a> SearchVisitor<'a> {
    fn new() -> Self {
        Self {
            ind: None,
            distance: f32::MAX,
            distance_sq: f32::MAX,
        }
    }

    #[inline(always)]
    fn visit(&mut self, ind: &'a SearchIdx, distance_sq: f32) {
        if self.distance_sq > distance_sq {
            self.ind = Some(ind);
            self.distance = distance_sq.sqrt();
            self.distance_sq = distance_sq;
        }
    }
}

struct SearchNode {
    ind: SearchIdx,
    near: Option<Box<Self>>,
    far: Option<Box<Self>>,
    rest: Vec<SearchIdx>,
    radius: f32,
    radius_sq: f32,
}

impl SearchNode {
    fn new(indexes: &mut Vec<SearchIdx>, weights: &[f32]) -> Option<Box<Self>> {
        if indexes.is_empty() {
            return None;
        }

        if indexes.len() == 1 {
            let node = Self {
                ind: indexes.pop().unwrap(),
                near: None,
                far: None,
                rest: [].into(),
                radius: f32::MAX,
                radius_sq: f32::MAX,
            };

            return Some(Box::new(node));
        }

        // Find the vantage point by the maximum weight
        // and remove it from the list
        let vp_ind = indexes
            .iter()
            .enumerate()
            .map(|(i, ind)| (i, OrdFloat32::from(weights[usize::from(ind.ind)])))
            .max_by_key(|&(_, w)| w)
            .map(|(i, _)| indexes.swap_remove(i))
            .unwrap();

        indexes.sort_by_cached_key(|i| OrdFloat32::from(dist_scalar(&vp_ind.data, &i.data)));

        let (near, far, rest, radius_sq) = if indexes.len() < 7 {
            (None, None, indexes.to_vec(), f32::MAX)
        } else {
            let half_idx = indexes.len() / 2;
            let (near_indexes, far_indexes) = indexes.split_at_mut(half_idx);
            let radius_sq = dist_scalar(&vp_ind.data, &far_indexes[0].data);

            (
                Self::new(near_indexes.to_vec().as_mut(), weights),
                Self::new(far_indexes.to_vec().as_mut(), weights),
                [].into(),
                radius_sq,
            )
        };

        let node = Self {
            ind: vp_ind,
            near,
            far,
            rest,
            radius: radius_sq.sqrt(),
            radius_sq,
        };

        Some(Box::new(node))
    }

    // SSE version - token passed through, no dispatch overhead
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn visit_sse<'a>(&'a self, token: archmage::X64V2Token, pin: &[f32; 4], nearest: &mut SearchVisitor<'a>) {
        let distance_sq = dist_sse(token, &self.ind.data, pin);

        nearest.visit(&self.ind, distance_sq);

        if !self.rest.is_empty() {
            for r in self.rest.iter() {
                let distance_sq = dist_sse(token, &r.data, pin);
                nearest.visit(r, distance_sq);
            }

            return;
        }

        if distance_sq < self.radius_sq {
            if let Some(near) = &self.near {
                near.visit_sse(token, pin, nearest);
            }
            let diff = self.radius - nearest.distance;
            if diff <= 0.0 || distance_sq >= diff * diff {
                if let Some(far) = &self.far {
                    far.visit_sse(token, pin, nearest);
                }
            }
        } else {
            if let Some(far) = &self.far {
                far.visit_sse(token, pin, nearest);
            }
            let sum = self.radius + nearest.distance;
            if distance_sq <= sum * sum {
                if let Some(near) = &self.near {
                    near.visit_sse(token, pin, nearest);
                }
            }
        }
    }

    // NEON version - token passed through, no dispatch overhead
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn visit_neon<'a>(&'a self, token: archmage::NeonToken, pin: &[f32; 4], nearest: &mut SearchVisitor<'a>) {
        let distance_sq = dist_neon(token, &self.ind.data, pin);

        nearest.visit(&self.ind, distance_sq);

        if !self.rest.is_empty() {
            for r in self.rest.iter() {
                let distance_sq = dist_neon(token, &r.data, pin);
                nearest.visit(r, distance_sq);
            }

            return;
        }

        if distance_sq < self.radius_sq {
            if let Some(near) = &self.near {
                near.visit_neon(token, pin, nearest);
            }
            let diff = self.radius - nearest.distance;
            if diff <= 0.0 || distance_sq >= diff * diff {
                if let Some(far) = &self.far {
                    far.visit_neon(token, pin, nearest);
                }
            }
        } else {
            if let Some(far) = &self.far {
                far.visit_neon(token, pin, nearest);
            }
            let sum = self.radius + nearest.distance;
            if distance_sq <= sum * sum {
                if let Some(near) = &self.near {
                    near.visit_neon(token, pin, nearest);
                }
            }
        }
    }

    // Scalar fallback
    #[inline(always)]
    fn visit_scalar<'a>(&'a self, pin: &[f32; 4], nearest: &mut SearchVisitor<'a>) {
        let distance_sq = dist_scalar(&self.ind.data, pin);

        nearest.visit(&self.ind, distance_sq);

        if !self.rest.is_empty() {
            for r in self.rest.iter() {
                let distance_sq = dist_scalar(&r.data, pin);
                nearest.visit(r, distance_sq);
            }

            return;
        }

        if distance_sq < self.radius_sq {
            if let Some(near) = &self.near {
                near.visit_scalar(pin, nearest);
            }
            let diff = self.radius - nearest.distance;
            if diff <= 0.0 || distance_sq >= diff * diff {
                if let Some(far) = &self.far {
                    far.visit_scalar(pin, nearest);
                }
            }
        } else {
            if let Some(far) = &self.far {
                far.visit_scalar(pin, nearest);
            }
            let sum = self.radius + nearest.distance;
            if distance_sq <= sum * sum {
                if let Some(near) = &self.near {
                    near.visit_scalar(pin, nearest);
                }
            }
        }
    }
}

pub(crate) struct SearchTree {
    root: Option<Box<SearchNode>>,
}

impl SearchTree {
    pub(crate) fn new(data: &[[f32; 4]], weights: &[f32]) -> Self {
        assert!(weights.len() >= data.len());
        assert!(data.len() <= 256);

        let mut indexes = data
            .iter()
            .enumerate()
            .map(|(i, &d)| SearchIdx {
                ind: i as u8,
                data: d,
            })
            .collect::<Vec<SearchIdx>>();

        let root = SearchNode::new(&mut indexes, weights);

        Self { root }
    }

    pub(crate) fn find_nearest(&self, pin: &[f32; 4]) -> (u8, [f32; 4], f32) {
        if let Some(vantage_point) = &self.root {
            let mut nearest = SearchVisitor::new();

            // Summon token ONCE here, then pass through entire traversal
            #[cfg(target_arch = "x86_64")]
            {
                use archmage::SimdToken;
                if let Some(token) = archmage::X64V2Token::summon() {
                    vantage_point.visit_sse(token, pin, &mut nearest);
                } else {
                    vantage_point.visit_scalar(pin, &mut nearest);
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                use archmage::SimdToken;
                if let Some(token) = archmage::NeonToken::summon() {
                    vantage_point.visit_neon(token, pin, &mut nearest);
                } else {
                    vantage_point.visit_scalar(pin, &mut nearest);
                }
            }

            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                vantage_point.visit_scalar(pin, &mut nearest);
            }

            if let Some(nearest_ind) = nearest.ind {
                (nearest_ind.ind, nearest_ind.data, nearest.distance)
            } else {
                (0, [0f32; 4], f32::MAX)
            }
        } else {
            (0, [0f32; 4], f32::MAX)
        }
    }
}

// SSE distance - takes token, no dispatch inside
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn dist_sse(_token: archmage::X64V2Token, c1: &[f32; 4], c2: &[f32; 4]) -> f32 {
    use core::arch::x86_64::*;
    // SAFETY: X64V2Token guarantees SSE4.2 is available
    unsafe {
        let pc1 = _mm_loadu_ps(c1.as_ptr());
        let pc2 = _mm_loadu_ps(c2.as_ptr());

        let diff = _mm_sub_ps(pc1, pc2);
        let sq = _mm_mul_ps(diff, diff);

        // Horizontal sum without memory round-trip
        let hi = _mm_movehl_ps(sq, sq);
        let sum2 = _mm_add_ps(sq, hi);
        let shuf = _mm_shuffle_ps(sum2, sum2, 1);
        let total = _mm_add_ss(sum2, shuf);
        _mm_cvtss_f32(total)
    }
}

// NEON distance - takes token, no dispatch inside
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn dist_neon(_token: archmage::NeonToken, c1: &[f32; 4], c2: &[f32; 4]) -> f32 {
    use core::arch::aarch64::*;
    // SAFETY: NeonToken guarantees NEON is available
    unsafe {
        let pc1 = vld1q_f32(c1.as_ptr());
        let pc2 = vld1q_f32(c2.as_ptr());

        let mut dist = vsubq_f32(pc1, pc2);
        dist = vmulq_f32(dist, dist);

        vaddvq_f32(dist)
    }
}

// Scalar fallback
#[inline(always)]
fn dist_scalar(c1: &[f32; 4], c2: &[f32; 4]) -> f32 {
    (c1[0] - c2[0]).powi(2)
        + (c1[1] - c2[1]).powi(2)
        + (c1[2] - c2[2]).powi(2)
        + (c1[3] - c2[3]).powi(2)
}
