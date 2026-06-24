# rq8

8-bit **rotational quantization (RQ)** for compressing high-dimensional float
vectors to ~1 byte per dimension while preserving cosine / dot / L2 distance
estimates. A faithful, standalone Rust port of [Weaviate's RQ][weaviate]
implementation (an optimized RaBitQ variant). See Weaviate's
["8-bit Rotational Quantization"][blog] for the theory.

[weaviate]: https://github.com/weaviate/weaviate/tree/main/adapters/repos/db/vector/compressionhelpers
[blog]: https://weaviate.io/blog/8-bit-rotational-quantization

## Why rotational quantization?

Plain scalar quantization (one byte per dimension) is cheap but usually
*inaccurate*, because real embedding coordinates are badly behaved for it:

- their per-dimension ranges and variances differ wildly, so a single global
  `(lower, step)` wastes most code points on a few wide dimensions;
- heavy tails / outliers stretch the range, pushing the bulk of values into a
  handful of buckets;
- some dimensions carry far more energy than others, so uniform bit allocation
  is mismatched to where the information actually is.

A random **rotation fixes all three at once.** Multiplying by a random
orthogonal matrix (here: random signs + swaps + a Walsh-Hadamard transform)
spreads each coordinate's energy across *all* output coordinates. By
concentration of measure the rotated coordinates become near-identically
distributed (approximately Gaussian) with equal variance and no outliers — so a
*single* global scalar quantizer is now near-optimal for every dimension at
once. That is the whole RaBitQ idea: **rotate, then scalar-quantize**, and the
rotation is exactly what makes the cheap scalar step accurate.

And because the rotation is *orthogonal*, it preserves inner products and L2
norms. So distances computed on the rotated/quantized codes still estimate the
original distances — no need to ever un-rotate at query time.

### How it works

1. A seeded, orthogonal **fast rotation** (random signs + swaps + a normalized
   Walsh-Hadamard transform, 3 rounds) Gaussianizes the vector. The FWHT makes
   this `O(d log d)` rather than `O(d²)` for a dense matrix.
2. A single global scalar quantizer maps each rotated coordinate to a byte via a
   per-vector `(lower, step)`.
3. Because rotation preserves inner products, the original distance is recovered
   from the codes plus `(lower, step, codeSum, norm²)` metadata.

## Why rq8?

rq8 deliberately targets the **fast-scan path at high accuracy**, rather than
chasing the smallest possible bit-rate:

- **8 bits = high accuracy.** Eight-bit codes keep the quantization error small
  enough that ranking is essentially preserved: on random 256-d unit vectors
  this crate measures **recall@10 ≈ 0.99** and dot-product **MAE ≈ 3e-4** vs
  exact f32, at **~4× compression** (1 byte/dim + 16 bytes metadata). Aggressive
  1-/2-bit schemes need a reranking pass over full vectors to recover this;
  8-bit RQ usually does not.
- **A scan-friendly inner loop.** Codes are a contiguous `[u8]`, one byte per
  dimension, so scoring a candidate is a single integer `u8·u8` dot product plus
  a few precomputed scalars — **no per-query lookup tables** (unlike 4-bit PQ
  FastScan), **no branches**, **no un-rotation**. The accumulator is `u32`, and
  the loop autovectorizes (SSE2/AVX widening multiply-add) so a linear scan over
  millions of codes stays memory-bound rather than compute-bound.
- **One formula, three metrics.** Cosine, dot, and L2 all fall out of the same
  estimate via indicator flags, so the hot loop is identical regardless of
  metric.
- **No training, no codebook.** Each vector encodes independently, so there is
  no fit step, no drift, and encoding parallelizes trivially.

The trade-off is explicit: 8-bit RQ is ~2× larger than a 4-bit scheme, and we
take that to keep the scan branch-free, LUT-free, and accurate enough to skip
reranking. Smaller bit-rates are a non-goal (see [Bit widths](#bit-widths)).

## Usage

```rust
use rq8::{Bits, Metric, RotationalQuantizer};

let q = RotationalQuantizer::new(/* input_dim */ 768, Bits::Eight, Metric::Cosine);

let code_a = q.encode(&vec_a);   // RqCode: 16-byte metadata + 1 byte/rotated-dim
let code_b = q.encode(&vec_b);

// Estimated distance directly from compressed codes:
let dist = q.distance(&code_a, &code_b).unwrap();

// Search loop: encode the query once, score many candidates:
let distancer = q.query_distancer(&query);
for code in &candidates {
    let d = distancer.distance(code).unwrap();
}

// Decode back to an approximate f32 vector if needed:
let approx = q.decode(&code_a);
```

### Metrics

`Metric::Cosine` (assumes unit vectors), `Metric::Dot` (returns negative inner
product, smaller = more similar), and `Metric::L2` (squared Euclidean).

## Persistence / compatibility

- The API mirrors [`rvector`](../rvector)'s quantizer conventions (a `Bits` enum
  like `NvqBits`, `&[f32]` in / `Vec<u8>` codes out) so it drops into the same
  call sites.
- Enable the `serde` feature to derive `Serialize`/`Deserialize` on
  `RotationalQuantizer`, `FastRotation`, and `RqCode` — this is how
  [`bifrost`](../bifrost)'s NLP store can persist codes with `bincode`.
- `RqCode::to_bytes` / `from_bytes` give a flat big-endian layout matching
  Weaviate's wire format for the code body.
- It is **not** byte-compatible with Weaviate's Go encoder (different RNG). If
  you need to reproduce a rotation across processes, persist the
  `FastRotation` (it stores the realized swaps and signs, so it is
  RNG-independent).

## Bit widths

`Bits::Eight` is the supported, tested configuration. `Bits::Four` is reserved:
it currently encodes correctly but stores one byte per dimension (no packing),
so it saves no space yet. Smaller bit-rates are an explicit non-goal.

## Tests

```sh
cargo test --features serde -- --nocapture
```

The suite verifies rotation self-inversion and norm preservation, quantization
error bounds, distance-estimate accuracy vs exact f32 (cosine/dot/L2),
RaBitQ concentration bounds, code-point distribution, end-to-end recall@10 vs
exact search, and serde/bincode round-tripping.
