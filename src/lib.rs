//! # rq8 — 8-bit rotational quantization
//!
//! A small, dependency-light implementation of **rotational quantization (RQ)**
//! for compressing high-dimensional float vectors to ~1 byte per dimension while
//! preserving enough information to estimate cosine / dot / L2 distances
//! accurately. It is a faithful port of [Weaviate's RQ][weaviate] (itself an
//! optimized variant of RaBitQ); see Weaviate's write-up
//! ["8-bit Rotational Quantization"][blog] for the theory.
//!
//! [weaviate]: https://github.com/weaviate/weaviate
//! [blog]: https://weaviate.io/blog/8-bit-rotational-quantization
//!
//! ## How it works
//!
//! 1. A seeded, orthogonal [`FastRotation`] (random signs + swaps + a
//!    Walsh-Hadamard transform, 3 rounds) "Gaussianizes" the vector so every
//!    coordinate has a similar distribution.
//! 2. A single global scalar quantizer maps each rotated coordinate to a byte
//!    using a per-vector `(lower, step)`.
//! 3. Because the rotation is orthogonal it preserves inner products, so the
//!    original distance can be estimated from the codes plus a little metadata.
//!
//! No training/codebook is required — encoding a vector is independent of any
//! other vector.
//!
//! ## Quick start
//!
//! ```
//! use rq8::{Bits, Metric, RotationalQuantizer};
//!
//! let q = RotationalQuantizer::new(8, Bits::Eight, Metric::Dot);
//! let a = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
//! let b = [0.2, 0.1, 0.4, 0.3, 0.6, 0.5, 0.8, 0.7];
//!
//! let ca = q.encode(&a);
//! let cb = q.encode(&b);
//!
//! let estimated_dot = q.dot_estimate(&ca, &cb);
//! let true_dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
//! assert!((estimated_dot - true_dot).abs() < 0.05);
//! ```
//!
//! ## Compatibility
//!
//! The API intentionally echoes `rvector`'s quantizer conventions (a `Bits`
//! enum à la `NvqBits`, `&[f32]` in / `Vec<u8>` codes out, a plain `dot`) and is
//! serde-friendly (enable the `serde` feature) so `bifrost`'s NLP store can
//! persist codes with `bincode`. It is **not** wire-compatible with Weaviate's
//! Go encoder (different RNG); persist [`FastRotation`] if you need to reproduce
//! a rotation across processes.
//!
//! ## Bit widths
//!
//! [`Bits::Eight`] is the supported, tested configuration. [`Bits::Four`] is
//! reserved: it currently encodes correctly but stores one byte per dimension
//! (no packing), so it saves no space yet. Smaller bit-rates are a non-goal.

#![forbid(unsafe_code)]

mod error;
mod quantizer;
mod rotation;

pub use error::{Error, Result};
pub use quantizer::{
    Bits, Metric, QueryDistancer, RQ_METADATA_SIZE, RotationalQuantizer, RqCode,
};
pub use rotation::{DEFAULT_FAST_ROTATION_SEED, FastRotation, Swap};
