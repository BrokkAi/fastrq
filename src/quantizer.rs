//! Rotational quantization itself: encoding f32 vectors to compact codes and
//! estimating distances between codes.
//!
//! The 8-bit path is a port of Weaviate's `RotationalQuantizer`
//! (`adapters/repos/db/vector/compressionhelpers/rotational_quantization.go`).
//! The 4-bit path packs two codes per byte in a split-nibble layout (following
//! Lucene's int4 scalar quantization) and scores queries asymmetrically at
//! 8 bits (following Lucene BBQ and Weaviate's 1-bit RQ, both of which encode
//! the query several bits finer than the data).

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
/// [`Bits::Eight`] stores one byte per rotated dimension. [`Bits::Four`] packs
/// two dimensions per byte (see [`RqCode`] for the layout) and scores queries
/// asymmetrically at 8 bits. Smaller bit-rates than 4 are a non-goal.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bits {
    /// 4-bit codes, nibble-packed two per byte.
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

    /// Number of code bytes used to store `dim` per-dimension codes:
    /// nibble-packed for 4-bit, one byte per dimension for 8-bit. `dim` is
    /// always the rotated dimension (a multiple of 64), so the 4-bit case
    /// divides evenly.
    #[inline]
    pub fn code_bytes(self, dim: usize) -> usize {
        match self {
            Bits::Four => dim / 2,
            Bits::Eight => dim,
        }
    }

    /// Number of dimensions represented by `bytes` code bytes (the inverse of
    /// [`code_bytes`](Self::code_bytes)).
    #[inline]
    fn dims(self, bytes: usize) -> usize {
        match self {
            Bits::Four => bytes * 2,
            Bits::Eight => bytes,
        }
    }

    /// Canonical file extension for codes stored in the flat
    /// [`RqCode::to_bytes`] layout. The layout itself is headerless — a 4-bit
    /// code for `2d` dimensions is byte-identical in length to an 8-bit code
    /// for `d` — so the bit width has to travel out-of-band, and the extension
    /// is the conventional place to put it.
    pub fn extension(self) -> &'static str {
        match self {
            Bits::Four => "rq4",
            Bits::Eight => "rq8",
        }
    }
}

/// A quantized vector.
///
/// Layout mirrors Weaviate's `RQCode`: four f32 metadata fields plus the code
/// bytes. `code_sum` stores `step * Σ codes` and `norm2` stores `⟨x, x⟩`; both
/// are precomputed at encode time so distance estimation is a handful of
/// multiplies plus one integer dot product.
///
/// 8-bit codes store one byte per rotated dimension, in dimension order.
/// 4-bit codes are nibble-packed in a **split layout**: with `half = dim / 2`,
/// byte `i` holds dimension `i` in its low nibble and dimension `i + half` in
/// its high nibble. Pairing dimension `i` with `i + half` (rather than `i + 1`)
/// means both nibble streams unpack with uniform masks and no interleaving —
/// the layout Lucene uses for int4. The rotation makes all dimensions
/// statistically identical, so the pairing choice costs nothing.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct RqCode {
    lower: f32,
    step: f32,
    code_sum: f32,
    norm2: f32,
    bits: Bits,
    codes: Vec<u8>,
}

/// Number of metadata bytes in the flat byte representation (4 × f32).
pub const RQ_METADATA_SIZE: usize = 16;

impl RqCode {
    /// The code representing the zero vector (also returned for degenerate
    /// input such as an empty or all-zero vector). `dim` is the rotated
    /// dimension; the stored byte count follows from `bits`.
    pub fn zero(dim: usize, bits: Bits) -> Self {
        Self {
            lower: 0.0,
            step: 0.0,
            code_sum: 0.0,
            norm2: 0.0,
            bits,
            codes: vec![0u8; bits.code_bytes(dim)],
        }
    }

    /// Rotated dimension of this code.
    pub fn dimension(&self) -> usize {
        self.bits.dims(self.codes.len())
    }

    /// Bit width of the stored codes.
    pub fn bits(&self) -> Bits {
        self.bits
    }

    /// Code bytes: one per dimension for 8-bit, nibble-packed (split layout,
    /// see the type docs) for 4-bit.
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
    /// the code bytes. Little-endian is a zero-swap `memcpy` on x86-64 /
    /// aarch64 and matches bincode's default byte order.
    ///
    /// The layout is headerless — it does not record the bit width, so readers
    /// must know it out-of-band (conventionally via the [`Bits::extension`]
    /// file extension). The 8-bit layout is byte-identical to fastrq 0.1.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(RQ_METADATA_SIZE + self.codes.len());
        out.extend_from_slice(&self.lower.to_le_bytes());
        out.extend_from_slice(&self.step.to_le_bytes());
        out.extend_from_slice(&self.code_sum.to_le_bytes());
        out.extend_from_slice(&self.norm2.to_le_bytes());
        out.extend_from_slice(&self.codes);
        out
    }

    /// Inverse of [`to_bytes`](Self::to_bytes). The caller supplies the bit
    /// width the bytes were written with; the layout cannot self-describe it.
    ///
    /// Prefer [`RotationalQuantizer::code_from_bytes`], which supplies the bit
    /// width and validates the dimension.
    pub fn from_bytes(b: &[u8], bits: Bits) -> Result<Self> {
        let view = RqCodeRef::from_bytes(b, bits)?;
        Ok(Self {
            lower: view.lower,
            step: view.step,
            code_sum: view.code_sum,
            norm2: view.norm2,
            bits,
            codes: view.codes.to_vec(),
        })
    }

    /// Borrow this code as a zero-copy [`RqCodeRef`].
    pub fn as_view(&self) -> RqCodeRef<'_> {
        RqCodeRef {
            lower: self.lower,
            step: self.step,
            code_sum: self.code_sum,
            norm2: self.norm2,
            bits: self.bits,
            codes: &self.codes,
        }
    }
}

/// A borrowed, zero-copy view over a code in its flat [`to_bytes`](RqCode::to_bytes)
/// layout: the four f32 metadata fields are parsed (cheap, no allocation) while
/// the code bytes are borrowed in place.
///
/// This is what makes the scan allocation-free: a candidate stored as raw bytes
/// (e.g. a slice of an mmap'd column) can be scored via
/// [`QueryDistancer::distance_bytes`] without building an owned [`RqCode`] per
/// candidate in the hot loop.
#[derive(Clone, Copy, Debug)]
pub struct RqCodeRef<'a> {
    lower: f32,
    step: f32,
    code_sum: f32,
    norm2: f32,
    bits: Bits,
    codes: &'a [u8],
}

impl<'a> RqCodeRef<'a> {
    /// Parse a view from the flat little-endian byte layout without copying the
    /// code bytes. The caller supplies the bit width (the layout is
    /// headerless). Returns [`Error::DimensionMismatch`] if `b` is too short to
    /// hold the metadata.
    pub fn from_bytes(b: &'a [u8], bits: Bits) -> Result<Self> {
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
            bits,
            codes: &b[RQ_METADATA_SIZE..],
        })
    }

    /// Rotated dimension of this code.
    pub fn dimension(&self) -> usize {
        self.bits.dims(self.codes.len())
    }

    /// Bit width of the codes.
    pub fn bits(&self) -> Bits {
        self.bits
    }

    /// Code bytes (nibble-packed for 4-bit; see [`RqCode`] docs).
    pub fn codes(&self) -> &'a [u8] {
        self.codes
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

    /// The rotated/output dimension (a multiple of 64).
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

    /// Size in bytes of one code in the flat [`RqCode::to_bytes`] layout:
    /// metadata plus code bytes. Useful as the stride when scanning a packed
    /// column of codes.
    pub fn code_size(&self) -> usize {
        RQ_METADATA_SIZE + self.bits.code_bytes(self.output_dim())
    }

    /// Encode a vector into a quantized code.
    ///
    /// Inputs longer than `output_dim` are truncated; shorter inputs are
    /// zero-padded by the rotation. Degenerate inputs (empty / all-zero / no
    /// spread) produce [`RqCode::zero`].
    pub fn encode(&self, x: &[f32]) -> RqCode {
        self.encode_with_bits(x, self.bits)
    }

    /// [`encode`](Self::encode) straight into the flat [`RqCode::to_bytes`]
    /// layout with a single allocation — the write path for storing codes.
    pub fn encode_to_bytes(&self, x: &[f32]) -> Vec<u8> {
        let out_dim = self.output_dim();
        let code_bytes = self.bits.code_bytes(out_dim);
        let mut out = Vec::with_capacity(RQ_METADATA_SIZE + code_bytes);
        out.resize(RQ_METADATA_SIZE, 0);

        let mut meta = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        match self.prepare(x, self.bits) {
            None => out.resize(RQ_METADATA_SIZE + code_bytes, 0),
            Some((x, rx, lower, step)) => {
                let sum = quantize_into(&rx, lower, step, self.bits, &mut out);
                meta = (lower, step, step * sum as f32, dot(x, x));
            }
        }
        out[0..4].copy_from_slice(&meta.0.to_le_bytes());
        out[4..8].copy_from_slice(&meta.1.to_le_bytes());
        out[8..12].copy_from_slice(&meta.2.to_le_bytes());
        out[12..16].copy_from_slice(&meta.3.to_le_bytes());
        out
    }

    /// Rotate and compute the quantization range for the given width. `None`
    /// means degenerate input (empty / all-zero / no spread), i.e. the zero
    /// code.
    fn prepare<'a>(&self, x: &'a [f32], bits: Bits) -> Option<(&'a [f32], Vec<f32>, f32, f32)> {
        if x.is_empty() {
            return None;
        }
        let out_dim = self.output_dim();
        let x = if x.len() > out_dim { &x[..out_dim] } else { x };
        let rx = self.rotation.rotate(x);
        let lower = rx.iter().copied().fold(f32::INFINITY, f32::min);
        let upper = rx.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let step = (upper - lower) / bits.max_code();
        if step <= 0.0 {
            // Zero vector or indistinguishable from it.
            return None;
        }
        Some((x, rx, lower, step))
    }

    /// Encode at an explicit bit width. The public `encode` always uses the
    /// configured width; the 8-bit override exists for the asymmetric query
    /// path (see [`query_distancer`](Self::query_distancer)).
    fn encode_with_bits(&self, x: &[f32], bits: Bits) -> RqCode {
        let out_dim = self.output_dim();
        let Some((x, rx, lower, step)) = self.prepare(x, bits) else {
            return RqCode::zero(out_dim, bits);
        };
        let mut codes = Vec::with_capacity(bits.code_bytes(out_dim));
        let sum = quantize_into(&rx, lower, step, bits, &mut codes);
        RqCode {
            lower,
            step,
            code_sum: step * sum as f32,
            norm2: dot(x, x),
            bits,
            codes,
        }
    }

    /// Parse a flat code produced by this quantizer
    /// ([`encode_to_bytes`](Self::encode_to_bytes) or [`RqCode::to_bytes`]),
    /// supplying the bit width and validating the dimension.
    pub fn code_from_bytes(&self, b: &[u8]) -> Result<RqCode> {
        let code = RqCode::from_bytes(b, self.bits)?;
        let expected = self.output_dim();
        if code.dimension() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                actual: code.dimension(),
            });
        }
        Ok(code)
    }

    /// Parse a flat code and decode it back to an approximate f32 vector in
    /// one step — the read path matching [`encode_to_bytes`](Self::encode_to_bytes).
    pub fn decode_bytes(&self, b: &[u8]) -> Result<Vec<f32>> {
        Ok(self.decode(&self.code_from_bytes(b)?))
    }

    /// Reconstruct the *rotated* vector from a code (the lossy inverse of
    /// quantization, before un-rotation).
    pub fn restore_rotated(&self, code: &RqCode) -> Vec<f32> {
        let (lower, step) = (code.lower, code.step);
        match code.bits {
            Bits::Eight => code
                .codes
                .iter()
                .map(|&c| lower + step * c as f32)
                .collect(),
            Bits::Four => {
                let half = code.codes.len();
                let mut out = vec![0.0f32; half * 2];
                for (i, &b) in code.codes.iter().enumerate() {
                    out[i] = lower + step * (b & 15) as f32;
                    out[i + half] = lower + step * (b >> 4) as f32;
                }
                out
            }
        }
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
    /// quantizer (dimension `output_dim()`); mismatched lengths silently
    /// truncate rather than error. Use [`distance`](Self::distance) when the
    /// inputs are untrusted.
    #[inline]
    pub fn dot_estimate(&self, x: &RqCode, y: &RqCode) -> f32 {
        dot_estimate_views(self.output_dim(), x.as_view(), y.as_view())
    }

    /// Estimate the configured distance between two codes. Codes of different
    /// bit widths (e.g. an 8-bit query code against a 4-bit stored code) are
    /// supported.
    ///
    /// Returns [`Error::DimensionMismatch`] if either code's dimension differs
    /// from this quantizer's [`output_dim`](Self::output_dim) — which also
    /// catches codes from a different quantizer.
    pub fn distance(&self, x: &RqCode, y: &RqCode) -> Result<f32> {
        distance_views(self.metric, self.output_dim(), x.as_view(), y.as_view())
    }

    /// Build a reusable distancer for a query vector. Encoding the query once
    /// and scoring many candidate codes is the common search-loop pattern.
    ///
    /// The query is always encoded at **8 bits**, regardless of the index bit
    /// width. For an 8-bit index this is the symmetric scheme; for a 4-bit
    /// index it is asymmetric (fine query × coarse data), which recovers most
    /// of the accuracy a full-precision query would — the data-side error
    /// dominates by ~16² in variance — at integer-kernel speed. Lucene BBQ and
    /// Weaviate's 1-bit RQ make the same trade.
    pub fn query_distancer(&self, query: &[f32]) -> QueryDistancer {
        QueryDistancer {
            metric: self.metric,
            output_dim: self.output_dim(),
            candidate_bits: self.bits,
            query_code: self.encode_with_bits(query, Bits::Eight),
        }
    }
}

/// Holds an encoded query so distances to many candidate codes share the
/// query-side precomputation. Owns everything it needs (no borrow of the
/// quantizer), so it can be stored, sent across threads, and outlive the
/// quantizer that built it.
pub struct QueryDistancer {
    metric: Metric,
    output_dim: usize,
    candidate_bits: Bits,
    query_code: RqCode,
}

impl QueryDistancer {
    /// The encoded query (always 8-bit; see
    /// [`RotationalQuantizer::query_distancer`]).
    pub fn query_code(&self) -> &RqCode {
        &self.query_code
    }

    /// Distance from the query to a candidate code.
    pub fn distance(&self, candidate: &RqCode) -> Result<f32> {
        distance_views(
            self.metric,
            self.output_dim,
            self.query_code.as_view(),
            candidate.as_view(),
        )
    }

    /// Distance from the query to a candidate stored in its flat
    /// [`to_bytes`](RqCode::to_bytes) layout, scored without allocating an
    /// owned [`RqCode`]. This is the allocation-free scan path: point it
    /// straight at a slice of an mmap'd / packed code column. The candidate is
    /// parsed at the quantizer's configured bit width.
    ///
    /// Returns [`Error::DimensionMismatch`] if the candidate is too short for
    /// the metadata or its dimension differs from the quantizer's `output_dim`.
    pub fn distance_bytes(&self, candidate: &[u8]) -> Result<f32> {
        let candidate = RqCodeRef::from_bytes(candidate, self.candidate_bits)?;
        distance_views(
            self.metric,
            self.output_dim,
            self.query_code.as_view(),
            candidate,
        )
    }

    /// Score a batch of flat codes ("query this in-memory list"), failing on
    /// the first malformed candidate. Results are in input order. The loop is
    /// embarrassingly parallel; callers wanting parallelism can partition the
    /// list and call [`distance_bytes`](Self::distance_bytes) per shard.
    pub fn distances_bytes<'a, I>(&self, candidates: I) -> Result<Vec<f32>>
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        candidates
            .into_iter()
            .map(|c| self.distance_bytes(c))
            .collect()
    }
}

/// Quantize the rotated vector into `out` at the given width, returning the
/// integer code sum. 4-bit packs the split-nibble layout in a single pass by
/// walking the two halves together.
fn quantize_into(rx: &[f32], lower: f32, step: f32, bits: Bits, out: &mut Vec<u8>) -> u32 {
    // (v - lower)/step is in [0, max_code]; +0.5 then truncate = round.
    let q = |v: f32| ((v - lower) / step + 0.5) as u8;
    let mut sum: u32 = 0;
    match bits {
        Bits::Eight => {
            for &v in rx {
                let c = q(v);
                sum += c as u32;
                out.push(c);
            }
        }
        Bits::Four => {
            let half = rx.len() / 2;
            let (lo_half, hi_half) = rx.split_at(half);
            for (&vl, &vh) in lo_half.iter().zip(hi_half) {
                let (cl, ch) = (q(vl), q(vh));
                sum += (cl + ch) as u32;
                out.push(cl | (ch << 4));
            }
        }
    }
    sum
}

/// Shared dot-product estimate over borrowed views; dispatches to the kernel
/// matching the two codes' bit widths.
#[inline]
fn dot_estimate_views(output_dim: usize, x: RqCodeRef<'_>, y: RqCodeRef<'_>) -> f32 {
    let ip = match (x.bits, y.bits) {
        (Bits::Eight, Bits::Eight) => dot_u8(x.codes, y.codes),
        (Bits::Four, Bits::Four) => dot_u4(x.codes, y.codes),
        (Bits::Eight, Bits::Four) => dot_u8_u4(x.codes, y.codes),
        (Bits::Four, Bits::Eight) => dot_u8_u4(y.codes, x.codes),
    };
    let d = output_dim as f32;
    let a = d * x.lower * y.lower;
    let b = x.lower * y.code_sum;
    let c = y.lower * x.code_sum;
    let dd = x.step * y.step * ip as f32;
    a + b + c + dd
}

/// Shared distance implementation: the owned, byte-slice, and query-distancer
/// entry points all funnel through here.
#[inline]
fn distance_views(
    metric: Metric,
    output_dim: usize,
    x: RqCodeRef<'_>,
    y: RqCodeRef<'_>,
) -> Result<f32> {
    for dim in [x.dimension(), y.dimension()] {
        if dim != output_dim {
            return Err(Error::DimensionMismatch {
                expected: output_dim,
                actual: dim,
            });
        }
    }
    let (cos, l2) = metric.indicators();
    let dot = dot_estimate_views(output_dim, x, y);
    Ok(l2 * (x.norm2 + y.norm2) + cos - (1.0 + l2) * dot)
}

/// Plain inner product of two equal-length f32 slices.
#[inline]
fn dot(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}

/// Integer dot product of two 8-bit code slices. `u8·u8` maxes at 65 025, and
/// even ~2048 dims keep the sum well within `u32`.
#[inline]
fn dot_u8(x: &[u8], y: &[u8]) -> u32 {
    x.iter().zip(y).map(|(&a, &b)| a as u32 * b as u32).sum()
}

/// Integer dot product of two nibble-packed 4-bit code slices (split layout).
/// Each byte contributes at most `2 * 15 * 15 = 450`, so a 64-byte chunk maxes
/// at 28 800 — accumulating chunks in `u16` keeps the inner loop in the
/// narrower lanes the autovectorizer can double up on.
#[inline]
fn dot_u4(x: &[u8], y: &[u8]) -> u32 {
    let mut sum: u32 = 0;
    for (cx, cy) in x.chunks(64).zip(y.chunks(64)) {
        let mut s: u16 = 0;
        for (&a, &b) in cx.iter().zip(cy) {
            let lo = (a & 15) * (b & 15);
            let hi = (a >> 4) * (b >> 4);
            s += lo as u16 + hi as u16;
        }
        sum += s as u32;
    }
    sum
}

/// Asymmetric integer dot product: 8-bit codes (natural dimension order)
/// against nibble-packed 4-bit codes. The split layout pairs packed byte `i`
/// with query bytes `i` and `i + half`, so only the candidate side unpacks.
#[inline]
fn dot_u8_u4(q: &[u8], packed: &[u8]) -> u32 {
    let half = packed.len().min(q.len() / 2);
    let (q_lo, q_hi) = q.split_at(half);
    let mut sum: u32 = 0;
    for ((&b, &ql), &qh) in packed.iter().zip(q_lo).zip(q_hi) {
        sum += ql as u32 * (b & 15) as u32 + qh as u32 * (b >> 4) as u32;
    }
    sum
}
