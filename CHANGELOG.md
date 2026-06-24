# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/BrokkAi/fastrq/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/BrokkAi/fastrq/releases/tag/v0.1.0
