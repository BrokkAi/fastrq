//! # fastrq — 8-bit rotational quantization
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
//! use fastrq::{Bits, Metric, RotationalQuantizer};
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
//! The API uses plain types — a [`Bits`] enum, `&[f32]` in / `Vec<u8>` codes
//! out, a free-standing dot product — so it drops into an existing vector index
//! without adapters. Enable the `serde` feature to persist the quantizer,
//! [`FastRotation`], and [`RqCode`] with `bincode` or any serde format. The
//! rotation is RNG-seeded but persistence does not depend on the RNG:
//! [`FastRotation`] stores the realized swaps and signs, so a deserialized
//! quantizer reproduces byte-identical codes across processes and versions.
//!
//! ## Bit widths
//!
//! [`Bits::Eight`] is the supported, tested configuration. [`Bits::Four`] is
//! reserved: it currently encodes correctly but stores one byte per dimension
//! (no packing), so it saves no space yet. Smaller bit-rates than 4 are a
//! non-goal.

#![forbid(unsafe_code)]

mod error;
mod quantizer;
mod rotation;

pub use error::{Error, Result};
pub use quantizer::{
    Bits, Metric, QueryDistancer, RQ_METADATA_SIZE, RotationalQuantizer, RqCode, RqCodeRef,
};
pub use rotation::{DEFAULT_FAST_ROTATION_SEED, FastRotation, Swap};
