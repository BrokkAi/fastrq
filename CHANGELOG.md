# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-07-02

Flat rq8 bytes written by 0.1.x parse and score identically — the on-disk
format is unchanged and now pinned by a golden test against the published
0.1.1 output. The *API* around it has breaking changes (below).

### Added
- **4-bit codes**: `Bits::Four` now packs two dimensions per byte in a
  split-nibble layout (byte `i` = dim `i` low nibble, dim `i + dim/2` high
  nibble, following Lucene's int4), halving code size vs 8-bit. Symmetric
  4×4 and mixed 8×4 integer dot kernels.
- **Asymmetric queries**: `query_distancer` always encodes the query at
  8 bits. With a 4-bit index this scores fine-query × coarse-data (à la
  Lucene BBQ / Weaviate's 1-bit RQ); measured dot MAE improves ~1/√2 vs
  symmetric 4×4. rq4 recall@10 ≈ 0.85 on random 256-d unit vectors
  (rq8: ≈ 0.99).
- `RotationalQuantizer::encode_to_bytes`: encode straight into the flat
  layout with one allocation (the write path).
- `RotationalQuantizer::code_from_bytes` / `decode_bytes`: parse flat codes
  with the quantizer's bit width and a dimension check (the read path).
- `RotationalQuantizer::code_size`: flat-code byte size (scan stride).
- `QueryDistancer::distances_bytes`: score an in-memory list of flat codes.
- `Bits::extension()`: canonical file extensions (`"rq8"` / `"rq4"`) — the
  flat layout is headerless, so the bit width travels out-of-band.
- `Bits::code_bytes(dim)`, `RqCode::bits()`, `RqCodeRef::bits()`.
- Golden-bytes test pinning the flat encoding for the default seed.

### Changed (breaking)
- `QueryDistancer` no longer borrows the quantizer (no lifetime parameter);
  it owns the encoded query and can outlive the quantizer / cross threads
  without `Box::leak` workarounds.
- `RqCode::zero(dim)` → `RqCode::zero(dim, bits)`.
- `RqCode::from_bytes(b)` / `RqCodeRef::from_bytes(b)` now take the bit
  width: `from_bytes(b, bits)` (the flat layout cannot self-describe it).
  Prefer `RotationalQuantizer::code_from_bytes`, which also validates.
- `RqCode`'s serde representation gained a `bits` field (bincode streams
  from 0.1.x do not deserialize; the flat `to_bytes` layout is unaffected).

## [0.1.1] - 2026-06-24

### Added
- `RqCodeRef<'a>`: a zero-copy view over a code's flat byte layout (metadata
  parsed, code bytes borrowed), plus `RqCode::as_view`.
- `QueryDistancer::distance_bytes(&[u8])`: score a candidate straight from its
  `to_bytes` layout with no per-candidate allocation — the allocation-free scan
  path for mmap'd / packed code columns. The owned `distance` and the byte path
  share one internal implementation (no public ref/owned API split).

## [0.1.0] - 2026-06-24

Initial release.

### Added
- `RotationalQuantizer`: 8-bit rotational quantization (RaBitQ-style), ported
  from Weaviate's RQ. Encodes f32 vectors to ~1 byte/dimension and estimates
  cosine / dot / L2 distances directly from the compressed codes.
- `FastRotation`: seeded orthogonal rotation (random signs + swaps + a
  normalized Walsh-Hadamard transform, self-inverse) with `rotate` / `unrotate`.
- `RqCode`: compact code (16-byte metadata + 1 byte/rotated-dim) with
  `to_bytes` / `from_bytes` for a flat big-endian layout.
- `QueryDistancer` for the search loop: encode a query once, score many codes
  via a LUT-free, autovectorizable `u8·u8` dot product.
- `Metric` (`Cosine`, `Dot`, `L2`) and `Bits` (`Eight`; `Four` reserved).
- Optional `serde` feature deriving `Serialize`/`Deserialize` on the quantizer,
  rotation, and codes (for `bincode` persistence in downstream consumers).
- Accuracy test suite: distance-estimate error vs exact f32, end-to-end
  recall@10 (~0.99 on random 256-d unit vectors), RaBitQ concentration bounds,
  rotation round-trip and norm preservation, code-point distribution, and
  serde/bincode round-tripping.

[Unreleased]: https://github.com/BrokkAi/fastrq/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/BrokkAi/fastrq/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/BrokkAi/fastrq/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/BrokkAi/fastrq/releases/tag/v0.1.0
