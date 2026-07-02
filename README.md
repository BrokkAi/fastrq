# fastrq

8-bit and 4-bit **rotational quantization (RQ)** for compressing
high-dimensional float vectors to ~1 byte (or ~half a byte) per dimension while
preserving cosine / dot / L2 distance estimates. The 8-bit path is a faithful,
standalone Rust port of [Weaviate's RQ][weaviate] implementation (an optimized
RaBitQ variant); see Weaviate's ["8-bit Rotational Quantization"][blog] for the
theory. The 4-bit path adds nibble packing and asymmetric queries following
Lucene's int4 / BBQ designs.

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

## Why fastrq?

fastrq deliberately targets the **fast-scan path at high accuracy**, rather than
chasing the smallest possible bit-rate:

- **8 bits = high accuracy.** Eight-bit codes keep the quantization error small
  enough that ranking is essentially preserved: on random 256-d unit vectors
  this crate measures **recall@10 ≈ 0.99** and dot-product **MAE ≈ 3e-4** vs
  exact f32, at **~4× compression** (1 byte/dim + 16 bytes metadata). Aggressive
  1-/2-bit schemes need a reranking pass over full vectors to recover this;
  8-bit RQ usually does not.
- **A scan-friendly inner loop.** Codes are a contiguous `[u8]` (one byte per
  dimension at 8 bits, nibble-packed at 4), so scoring a candidate is a single
  integer dot product plus a few precomputed scalars — **no per-query lookup
  tables** (unlike 4-bit PQ FastScan), **no branches**, **no un-rotation**. The
  loops autovectorize (widening multiply-add) so a linear scan over millions of
  codes stays memory-bound rather than compute-bound.
- **One formula, three metrics.** Cosine, dot, and L2 all fall out of the same
  estimate via indicator flags, so the hot loop is identical regardless of
  metric.
- **No training, no codebook.** Each vector encodes independently, so there is
  no fit step, no drift, and encoding parallelizes trivially.

When storage pressure outweighs the last bit of recall, `Bits::Four` halves the
code size (~8× vs f32) while keeping the scan branch-free and LUT-free: codes
pack two dimensions per byte, and queries are still encoded at 8 bits
(asymmetric scoring, as in Lucene BBQ and Weaviate's 1-bit RQ), so most of the
lost precision comes back for free at query time. Smaller bit-rates than 4 are
a non-goal (see [Bit widths](#bit-widths)).

## Usage

```rust
use fastrq::{Bits, Metric, RotationalQuantizer};

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

// Flat-bytes persistence (single-allocation write, validated read):
let flat = q.encode_to_bytes(&vec_a);        // metadata + code bytes
let back = q.code_from_bytes(&flat).unwrap();
let d = distancer.distance_bytes(&flat).unwrap();  // allocation-free scan
```

For 4-bit codes, build the quantizer with `Bits::Four`: stored codes halve in
size, and `query_distancer` transparently encodes the query at 8 bits so
scoring stays accurate (fine query × coarse data).

### Metrics

`Metric::Cosine` (assumes unit vectors), `Metric::Dot` (returns negative inner
product, smaller = more similar), and `Metric::L2` (squared Euclidean).

## Persistence / compatibility

- Plain, predictable types: `&[f32]` in, `Vec<u8>` codes out, a `Bits` enum, and
  a free-standing `dot` — so it slots into an existing vector index without
  adapters.
- Enable the `serde` feature to derive `Serialize`/`Deserialize` on
  `RotationalQuantizer`, `FastRotation`, and `RqCode`, so codes (and the
  quantizer itself) persist with `bincode` or any other serde format.
- `encode_to_bytes` / `code_from_bytes` give a flat little-endian layout (4 × f32
  metadata then the code bytes) for storing alongside non-serde formats. It is a
  zero-swap `memcpy` on x86-64 / aarch64 and agrees with bincode's byte order.
  The layout is headerless — it does not record the bit width — so store the
  width out-of-band; the convention is the `Bits::extension()` file extension
  (`.rq8` / `.rq4`). The 8-bit layout is byte-identical to fastrq 0.1 and is
  pinned by a golden test.
- The rotation is RNG-seeded, but persistence does not depend on the RNG:
  `FastRotation` stores the realized swaps and signs, so a deserialized quantizer
  reproduces byte-identical codes across processes and crate versions.

## Bit widths

- **`Bits::Eight`** — one byte per rotated dimension; the high-accuracy
  configuration (recall@10 ≈ 0.99, dot MAE ≈ 3e-4 on random 256-d unit
  vectors). Usually no reranking needed.
- **`Bits::Four`** — two dimensions per byte in a split-nibble layout (byte `i`
  holds dim `i` and dim `i + d/2`, so unpacking is two masks and no shuffles,
  as in Lucene's int4). Queries score asymmetrically at 8 bits; measured
  recall@10 ≈ 0.85 on random 256-d unit vectors, so plan on reranking when
  exact order matters.

Smaller bit-rates than 4 are an explicit non-goal.

## Tests

```sh
cargo test --features serde -- --nocapture
```

The suite verifies rotation self-inversion and norm preservation, quantization
error bounds, distance-estimate accuracy vs exact f32 (cosine/dot/L2),
RaBitQ concentration bounds, code-point distribution, end-to-end recall@10 vs
exact search, and serde/bincode round-tripping.
