//! Fast pseudo-random orthogonal rotation.
//!
//! This is a port of Weaviate's `FastRotation`
//! (`entities/vectorindex/compression/fast_rotation.go`). A rotation round is:
//!
//! 1. random sign flips (a diagonal `±1` matrix), then
//! 2. a random pairwise permutation (swaps), then
//! 3. a block Fast Walsh-Hadamard Transform (FWHT) in blocks of 256 (falling
//!    back to 64 for the tail).
//!
//! Each piece is orthogonal, so the whole rotation is orthogonal and therefore
//! preserves inner products and L2 norms — exactly the property that lets us
//! estimate distances from quantized codes. The FWHT here is *normalized* so
//! that it is its own inverse (`T·T = I`); that is why [`FastRotation::rotate`]
//! and [`FastRotation::unrotate`] apply the same transform, just with the
//! sign/swap steps reversed for un-rotation.
//!
//! Note on compatibility: we do **not** reproduce Go's `math/rand/v2` byte for
//! byte, so a rotation built from a given seed here will differ from Weaviate's.
//! That is fine — the rotation only has to be self-consistent for encode/decode,
//! and persistence stores the realized swaps/signs ([`FastRotation`] is fully
//! serializable) rather than relying on the RNG.

/// A pair of indices to swap during a rotation round.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Swap {
    /// Lower index (we always store `i < j`).
    pub i: u32,
    /// Higher index.
    pub j: u32,
}

/// A realized random rotation. Cheap to clone; holds one set of swaps and signs
/// per round.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct FastRotation {
    /// Dimension of the rotated output. Always a multiple of 64 and `>= input`.
    output_dim: usize,
    /// Number of rounds applied.
    rounds: usize,
    /// `rounds` slices of `output_dim/2` swaps.
    swaps: Vec<Vec<Swap>>,
    /// `rounds` slices of `output_dim` signs (each `+1.0` or `-1.0`).
    signs: Vec<Vec<f32>>,
}

/// Default seed, matching Weaviate's `DefaultFastRotationSeed`.
pub const DEFAULT_FAST_ROTATION_SEED: u64 = 0x535a_b510_5169_b1df;

/// Round `input_dim` up to the next multiple of 64 (minimum 64).
fn padded_dim(input_dim: usize) -> usize {
    let mut out = 64;
    while out < input_dim {
        out += 64;
    }
    out
}

/// SplitMix64 — a tiny, fast, well-distributed PRNG. We only need determinism
/// and good-enough randomness for sign/permutation generation, not Go parity.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform float in `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        // Top 53 bits → mantissa.
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform integer in `[0, n)`.
    fn next_below(&mut self, n: usize) -> usize {
        (self.next_f64() * n as f64) as usize
    }
}

fn random_signs(dim: usize, rng: &mut SplitMix64) -> Vec<f32> {
    (0..dim)
        .map(|_| if rng.next_f64() < 0.5 { -1.0 } else { 1.0 })
        .collect()
}

/// A random pairing of all `n` indices into `n/2` swaps, each used exactly once,
/// sorted by the lower index for a more sequential access pattern.
fn random_swaps(n: usize, rng: &mut SplitMix64) -> Vec<Swap> {
    // Fisher-Yates shuffle of [0, n). `n` is a multiple of 64; u32 indices
    // support output dimensions up to ~4 billion.
    let mut p: Vec<u32> = (0..n as u32).collect();
    for i in (1..n).rev() {
        let j = rng.next_below(i + 1);
        p.swap(i, j);
    }
    let mut swaps: Vec<Swap> = (0..n / 2)
        .map(|s| {
            let (a, b) = (p[2 * s], p[2 * s + 1]);
            if a < b {
                Swap { i: a, j: b }
            } else {
                Swap { i: b, j: a }
            }
        })
        .collect();
    swaps.sort_by_key(|s| s.i);
    swaps
}

impl FastRotation {
    /// Build a rotation for `input_dim`-dimensional inputs using `rounds` rounds
    /// and the given `seed`. Weaviate uses 3 rounds, which is a good
    /// quality/speed trade-off.
    pub fn new(input_dim: usize, rounds: usize, seed: u64) -> Self {
        let output_dim = padded_dim(input_dim);
        let mut rng = SplitMix64::new(seed ^ 0x385a_b528_5169_b1ac);
        let mut swaps = Vec::with_capacity(rounds);
        let mut signs = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            swaps.push(random_swaps(output_dim, &mut rng));
            signs.push(random_signs(output_dim, &mut rng));
        }
        Self {
            output_dim,
            rounds,
            swaps,
            signs,
        }
    }

    /// Reconstruct a rotation from previously serialized pieces.
    pub fn restore(
        output_dim: usize,
        rounds: usize,
        swaps: Vec<Vec<Swap>>,
        signs: Vec<Vec<f32>>,
    ) -> Self {
        Self {
            output_dim,
            rounds,
            swaps,
            signs,
        }
    }

    /// Output (rotated) dimension. Always a multiple of 64.
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Number of rounds.
    pub fn rounds(&self) -> usize {
        self.rounds
    }

    /// Rotate `x` (length `<= output_dim`; shorter inputs are zero-padded),
    /// returning a freshly allocated `output_dim`-length vector.
    pub fn rotate(&self, x: &[f32]) -> Vec<f32> {
        let mut rx = vec![0.0f32; self.output_dim];
        let n = x.len().min(self.output_dim);
        rx[..n].copy_from_slice(&x[..n]);
        for round in 0..self.rounds {
            let signs = &self.signs[round];
            for s in &self.swaps[round] {
                let (i, j) = (s.i as usize, s.j as usize);
                let (a, b) = (rx[i], rx[j]);
                rx[i] = signs[i] * b;
                rx[j] = signs[j] * a;
            }
            block_fwht(&mut rx);
        }
        rx
    }

    /// Allocate a copy and un-rotate it.
    pub fn unrotate(&self, rx: &[f32]) -> Vec<f32> {
        let mut x = rx.to_vec();
        self.unrotate_in_place(&mut x);
        x
    }

    /// Invert [`rotate`](Self::rotate) in place. `x` must have length
    /// `output_dim`.
    pub fn unrotate_in_place(&self, x: &mut [f32]) {
        for round in (0..self.rounds).rev() {
            // FWHT is self-inverse, so apply it first.
            block_fwht(x);
            // Then undo swaps+signs in reverse order. The forward op is
            //   x[i], x[j] = signs[i]*x[j], signs[j]*x[i]
            // whose inverse is
            //   x[i], x[j] = signs[j]*x[j], signs[i]*x[i]
            let signs = &self.signs[round];
            for s in self.swaps[round].iter().rev() {
                let (i, j) = (s.i as usize, s.j as usize);
                let (a, b) = (x[i], x[j]);
                x[i] = signs[j] * b;
                x[j] = signs[i] * a;
            }
        }
    }
}

/// Apply the block FWHT to `x` in place: blocks of 256 where possible, otherwise
/// 64. `x.len()` is always a multiple of 64.
fn block_fwht(x: &mut [f32]) {
    let mut pos = 0;
    let len = x.len();
    while pos < len {
        if len - pos >= 256 {
            fwht256(&mut x[pos..pos + 256]);
            pos += 256;
        } else {
            fwht64(&mut x[pos..pos + 64]);
            pos += 64;
        }
    }
}

/// Normalized 16-point FWHT (the `normalize` factor is folded into the inputs).
/// Unrolled exactly as in Weaviate for speed and bit-faithfulness.
#[inline]
fn fwht16(x: &mut [f32], normalize: f32) {
    let mut x0 = normalize * x[0];
    let mut x1 = normalize * x[1];
    let mut x2 = normalize * x[2];
    let mut x3 = normalize * x[3];
    let mut x4 = normalize * x[4];
    let mut x5 = normalize * x[5];
    let mut x6 = normalize * x[6];
    let mut x7 = normalize * x[7];
    let mut x8 = normalize * x[8];
    let mut x9 = normalize * x[9];
    let mut x10 = normalize * x[10];
    let mut x11 = normalize * x[11];
    let mut x12 = normalize * x[12];
    let mut x13 = normalize * x[13];
    let mut x14 = normalize * x[14];
    let mut x15 = normalize * x[15];

    (x0, x1) = (x0 + x1, x0 - x1);
    (x2, x3) = (x2 + x3, x2 - x3);
    (x0, x2) = (x0 + x2, x0 - x2);
    (x1, x3) = (x1 + x3, x1 - x3);

    (x4, x5) = (x4 + x5, x4 - x5);
    (x6, x7) = (x6 + x7, x6 - x7);
    (x4, x6) = (x4 + x6, x4 - x6);
    (x5, x7) = (x5 + x7, x5 - x7);

    (x0, x4) = (x0 + x4, x0 - x4);
    (x1, x5) = (x1 + x5, x1 - x5);
    (x2, x6) = (x2 + x6, x2 - x6);
    (x3, x7) = (x3 + x7, x3 - x7);

    (x8, x9) = (x8 + x9, x8 - x9);
    (x10, x11) = (x10 + x11, x10 - x11);
    (x8, x10) = (x8 + x10, x8 - x10);
    (x9, x11) = (x9 + x11, x9 - x11);

    (x12, x13) = (x12 + x13, x12 - x13);
    (x14, x15) = (x14 + x15, x14 - x15);
    (x12, x14) = (x12 + x14, x12 - x14);
    (x13, x15) = (x13 + x15, x13 - x15);

    (x8, x12) = (x8 + x12, x8 - x12);
    (x9, x13) = (x9 + x13, x9 - x13);
    (x10, x14) = (x10 + x14, x10 - x14);
    (x11, x15) = (x11 + x15, x11 - x15);

    (x0, x8) = (x0 + x8, x0 - x8);
    (x1, x9) = (x1 + x9, x1 - x9);
    (x2, x10) = (x2 + x10, x2 - x10);
    (x3, x11) = (x3 + x11, x3 - x11);
    (x4, x12) = (x4 + x12, x4 - x12);
    (x5, x13) = (x5 + x13, x5 - x13);
    (x6, x14) = (x6 + x14, x6 - x14);
    (x7, x15) = (x7 + x15, x7 - x15);

    x[0] = x0;
    x[1] = x1;
    x[2] = x2;
    x[3] = x3;
    x[4] = x4;
    x[5] = x5;
    x[6] = x6;
    x[7] = x7;
    x[8] = x8;
    x[9] = x9;
    x[10] = x10;
    x[11] = x11;
    x[12] = x12;
    x[13] = x13;
    x[14] = x14;
    x[15] = x15;
}

/// Normalized 64-point FWHT (`1/8`), so that `fwht64(fwht64(x)) == x`.
fn fwht64(x: &mut [f32]) {
    const NORMALIZE: f32 = 0.125;
    fwht16(&mut x[..16], NORMALIZE);
    fwht16(&mut x[16..32], NORMALIZE);
    for i in 0..16 {
        (x[i], x[16 + i]) = (x[i] + x[16 + i], x[i] - x[16 + i]);
    }
    fwht16(&mut x[32..48], NORMALIZE);
    fwht16(&mut x[48..64], NORMALIZE);
    for i in 32..48 {
        (x[i], x[16 + i]) = (x[i] + x[16 + i], x[i] - x[16 + i]);
    }
    for i in 0..32 {
        (x[i], x[32 + i]) = (x[i] + x[32 + i], x[i] - x[32 + i]);
    }
}

/// 64-point FWHT building block used inside the 256-point transform, with the
/// `1/16` normalization that makes the full 256-point transform self-inverse.
fn block64_fwht256(x: &mut [f32]) {
    const NORMALIZE: f32 = 0.0625;
    fwht16(&mut x[0..16], NORMALIZE);
    fwht16(&mut x[16..32], NORMALIZE);
    for i in 0..16 {
        (x[i], x[16 + i]) = (x[i] + x[16 + i], x[i] - x[16 + i]);
    }
    fwht16(&mut x[32..48], NORMALIZE);
    fwht16(&mut x[48..64], NORMALIZE);
    for i in 32..48 {
        (x[i], x[16 + i]) = (x[i] + x[16 + i], x[i] - x[16 + i]);
    }
    for i in 0..32 {
        (x[i], x[32 + i]) = (x[i] + x[32 + i], x[i] - x[32 + i]);
    }
}

/// Normalized 256-point FWHT (`1/16`), so that `fwht256(fwht256(x)) == x`.
fn fwht256(x: &mut [f32]) {
    block64_fwht256(&mut x[0..64]);
    block64_fwht256(&mut x[64..128]);
    for i in 0..64 {
        (x[i], x[64 + i]) = (x[i] + x[64 + i], x[i] - x[64 + i]);
    }
    block64_fwht256(&mut x[128..192]);
    block64_fwht256(&mut x[192..256]);
    for i in 128..192 {
        (x[i], x[64 + i]) = (x[i] + x[64 + i], x[i] - x[64 + i]);
    }
    for i in 0..128 {
        (x[i], x[128 + i]) = (x[i] + x[128 + i], x[i] - x[128 + i]);
    }
}
