//! Rotational quantization itself: encoding f32 vectors to compact codes and
//! estimating distances between codes.
//!
//! Port of Weaviate's `RotationalQuantizer`
//! (`adapters/repos/db/vector/compressionhelpers/rotational_quantization.go`).

use crate::error::{Error, Result};
use crate::rotation::{DEFAULT_FAST_ROTATION_SEED, FastRotation};

/// Distance metric. The estimator supports exactly the three metrics Weaviate's
/// RQ supports; all are computed from the same dot-product estimate.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    /// Cosine distance, `1 - cos(x, y)`. Assumes (and is exact for) unit
    /// vectors; the code does not normalize for you.
    Cosine,
    /// Negative inner product, `-⟨x, y⟩` (smaller = more similar), matching
    /// Weaviate's "dot" convention.
    Dot,
    /// Squared Euclidean distance, `‖x - y‖²`.
    L2,
}

impl Metric {
    /// `(cos, l2)` indicator pair used by the unified distance formula.
    #[inline]
    fn indicators(self) -> (f32, f32) {
        match self {
            Metric::Cosine => (1.0, 0.0),
            Metric::Dot => (0.0, 0.0),
            Metric::L2 => (0.0, 1.0),
        }
    }
}

/// Bit-width of the per-dimension codes.
///
/// Only [`Bits::Eight`] is fully implemented and tested. [`Bits::Four`] is
/// reserved for the future; smaller bit-rates than 4 are explicitly a non-goal. The
/// enum exists so the public API is stable if 4-bit lands later.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bits {
    /// 4-bit codes (reserved, not yet implemented).
    Four,
    /// 8-bit codes (one byte per dimension).
    Eight,
}

impl Bits {
    /// Numeric width.
    pub fn as_u32(self) -> u32 {
        match self {
            Bits::Four => 4,
            Bits::Eight => 8,
        }
    }

    /// Largest representable code, `2^bits - 1`.
    #[inline]
    fn max_code(self) -> f32 {
        ((1u32 << self.as_u32()) - 1) as f32
    }
}

/// A quantized vector.
///
/// Layout mirrors Weaviate's `RQCode`: four f32 metadata fields plus one byte
/// per (rotated) dimension. `code_sum` stores `step * Σ codes` and `norm2`
/// stores `⟨x, x⟩`; both are precomputed at encode time so distance estimation
/// is a handful of multiplies plus one `u8·u8` dot product.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct RqCode {
    lower: f32,
    step: f32,
    code_sum: f32,
    norm2: f32,
    codes: Vec<u8>,
}

/// Number of metadata bytes in the flat byte representation (4 × f32).
pub const RQ_METADATA_SIZE: usize = 16;

impl RqCode {
    /// The code representing the zero vector (also returned for degenerate
    /// input such as an empty or all-zero vector).
    pub fn zero(dim: usize) -> Self {
        Self {
            lower: 0.0,
            step: 0.0,
            code_sum: 0.0,
            norm2: 0.0,
            codes: vec![0u8; dim],
        }
    }

    /// Rotated dimension of this code.
    pub fn dimension(&self) -> usize {
        self.codes.len()
    }

    /// Per-dimension code bytes.
    pub fn codes(&self) -> &[u8] {
        &self.codes
    }

    /// Quantization offset (the minimum rotated value).
    pub fn lower(&self) -> f32 {
        self.lower
    }

    /// Quantization step size.
    pub fn step(&self) -> f32 {
        self.step
    }

    /// `⟨x, x⟩` of the original (pre-rotation) vector.
    pub fn norm2(&self) -> f32 {
        self.norm2
    }

    /// Serialize to a flat `[u8]`: four little-endian f32 metadata fields then
    /// the code bytes. Little-endian is a zero-swap `memcpy` on x86-64 / aarch64
    /// and matches bincode's default byte order. Useful when storing alongside
    /// non-serde formats.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(RQ_METADATA_SIZE + self.codes.len());
        out.extend_from_slice(&self.lower.to_le_bytes());
        out.extend_from_slice(&self.step.to_le_bytes());
        out.extend_from_slice(&self.code_sum.to_le_bytes());
        out.extend_from_slice(&self.norm2.to_le_bytes());
        out.extend_from_slice(&self.codes);
        out
    }

    /// Inverse of [`to_bytes`](Self::to_bytes).
    pub fn from_bytes(b: &[u8]) -> Result<Self> {
        if b.len() < RQ_METADATA_SIZE {
            return Err(Error::DimensionMismatch {
                expected: RQ_METADATA_SIZE,
                actual: b.len(),
            });
        }
        let f = |o: usize| f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        Ok(Self {
            lower: f(0),
            step: f(4),
            code_sum: f(8),
            norm2: f(12),
            codes: b[RQ_METADATA_SIZE..].to_vec(),
        })
    }
}

/// The quantizer. Holds the rotation and the metric/bit configuration; encoding
/// and distance estimation are stateless beyond that, so it is cheap to share
/// (`&self`) across threads.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct RotationalQuantizer {
    input_dim: usize,
    bits: Bits,
    metric: Metric,
    rotation: FastRotation,
}

impl RotationalQuantizer {
    /// Number of rotation rounds. Three is Weaviate's choice: two leaves a
    /// detectable encoding bias, more buys little.
    const ROTATION_ROUNDS: usize = 3;

    /// Build a quantizer for `input_dim`-dimensional vectors using the default
    /// seed.
    pub fn new(input_dim: usize, bits: Bits, metric: Metric) -> Self {
        Self::with_seed(input_dim, bits, metric, DEFAULT_FAST_ROTATION_SEED)
    }

    /// Build a quantizer with an explicit rotation seed (useful for tests or for
    /// reproducing a specific rotation).
    pub fn with_seed(input_dim: usize, bits: Bits, metric: Metric, seed: u64) -> Self {
        let rotation = FastRotation::new(input_dim, Self::ROTATION_ROUNDS, seed);
        Self {
            input_dim,
            bits,
            metric,
            rotation,
        }
    }

    /// Reconstruct from a previously stored rotation (RNG-independent).
    pub fn from_rotation(
        input_dim: usize,
        bits: Bits,
        metric: Metric,
        rotation: FastRotation,
    ) -> Self {
        Self {
            input_dim,
            bits,
            metric,
            rotation,
        }
    }

    /// The declared input dimension.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// The rotated/output dimension (a multiple of 64). Codes have this many
    /// bytes.
    pub fn output_dim(&self) -> usize {
        self.rotation.output_dim()
    }

    /// The configured metric.
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// The configured bit width.
    pub fn bits(&self) -> Bits {
        self.bits
    }

    /// Access the underlying rotation (e.g. to serialize it separately).
    pub fn rotation(&self) -> &FastRotation {
        &self.rotation
    }

    /// Encode a vector into a quantized code.
    ///
    /// Inputs longer than `output_dim` are truncated; shorter inputs are
    /// zero-padded by the rotation. Degenerate inputs (empty / all-zero / no
    /// spread) produce [`RqCode::zero`].
    pub fn encode(&self, x: &[f32]) -> RqCode {
        let out_dim = self.output_dim();
        if x.is_empty() {
            return RqCode::zero(out_dim);
        }
        let x = if x.len() > out_dim { &x[..out_dim] } else { x };

        let rx = self.rotation.rotate(x);
        let max_code = self.bits.max_code();
        let lower = rx.iter().copied().fold(f32::INFINITY, f32::min);
        let upper = rx.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let step = (upper - lower) / max_code;
        if step <= 0.0 {
            // Zero vector or indistinguishable from it.
            return RqCode::zero(out_dim);
        }

        let mut codes = vec![0u8; out_dim];
        let mut code_sum_int: u32 = 0;
        for (i, &v) in rx.iter().enumerate() {
            // (v - lower)/step is in [0, max_code]; +0.5 then truncate = round.
            let c = ((v - lower) / step + 0.5) as u8;
            code_sum_int += c as u32;
            codes[i] = c;
        }
        RqCode {
            lower,
            step,
            code_sum: step * code_sum_int as f32,
            norm2: dot(x, x),
            codes,
        }
    }

    /// Reconstruct the *rotated* vector from a code (the lossy inverse of
    /// quantization, before un-rotation).
    pub fn restore_rotated(&self, code: &RqCode) -> Vec<f32> {
        code.codes
            .iter()
            .map(|&c| code.lower + code.step * c as f32)
            .collect()
    }

    /// Decode a code back to an approximation of the original (un-rotated)
    /// vector. The result has `output_dim` entries; the first `input_dim` are
    /// the meaningful ones.
    pub fn decode(&self, code: &RqCode) -> Vec<f32> {
        let mut rotated = self.restore_rotated(code);
        self.rotation.unrotate_in_place(&mut rotated);
        rotated
    }

    /// Estimate `⟨x, y⟩` from two codes. Rotation preserves inner products, so
    /// this estimates the original dot product.
    ///
    /// This is the unchecked fast path: it assumes both codes came from this
    /// quantizer (length `output_dim()`). [`dot_bytes`] zips the code slices, so
    /// mismatched lengths silently truncate rather than error. Use
    /// [`distance`](Self::distance) when the inputs are untrusted.
    #[inline]
    pub fn dot_estimate(&self, x: &RqCode, y: &RqCode) -> f32 {
        let d = self.output_dim() as f32;
        let a = d * x.lower * y.lower;
        let b = x.lower * y.code_sum;
        let c = y.lower * x.code_sum;
        let dd = x.step * y.step * dot_bytes(&x.codes, &y.codes) as f32;
        a + b + c + dd
    }

    /// Estimate the configured distance between two codes.
    ///
    /// Returns [`Error::DimensionMismatch`] if either code's length differs from
    /// this quantizer's [`output_dim`](Self::output_dim) — which also catches
    /// two equally-but-wrongly-sized codes from a different quantizer.
    pub fn distance(&self, x: &RqCode, y: &RqCode) -> Result<f32> {
        let expected = self.output_dim();
        for code in [x, y] {
            if code.codes.len() != expected {
                return Err(Error::DimensionMismatch {
                    expected,
                    actual: code.codes.len(),
                });
            }
        }
        let (cos, l2) = self.metric.indicators();
        let dot = self.dot_estimate(x, y);
        Ok(l2 * (x.norm2 + y.norm2) + cos - (1.0 + l2) * dot)
    }

    /// Build a reusable distancer for a query vector. Encoding the query once
    /// and scoring many candidate codes is the common search-loop pattern.
    pub fn query_distancer(&self, query: &[f32]) -> QueryDistancer<'_> {
        QueryDistancer {
            quantizer: self,
            query_code: self.encode(query),
        }
    }
}

/// Holds an encoded query so distances to many candidate codes share the
/// query-side precomputation.
pub struct QueryDistancer<'a> {
    quantizer: &'a RotationalQuantizer,
    query_code: RqCode,
}

impl QueryDistancer<'_> {
    /// The encoded query.
    pub fn query_code(&self) -> &RqCode {
        &self.query_code
    }

    /// Distance from the query to a candidate code.
    pub fn distance(&self, candidate: &RqCode) -> Result<f32> {
        self.quantizer.distance(&self.query_code, candidate)
    }
}

/// Plain inner product of two equal-length f32 slices.
#[inline]
fn dot(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}

/// Integer dot product of two code-byte slices. `u8·u8` maxes at 65 025, and
/// even ~2048 dims keep the sum well within `u32`.
#[inline]
fn dot_bytes(x: &[u8], y: &[u8]) -> u32 {
    x.iter().zip(y).map(|(&a, &b)| a as u32 * b as u32).sum()
}
