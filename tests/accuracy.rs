//! Accuracy and correctness tests for rotational quantization.
//!
//! The headline tests verify that distance estimates from compressed codes stay
//! close to the true f32 distances, mirroring the bounds Weaviate asserts in its
//! own RQ test suite, plus an end-to-end nearest-neighbor recall check that
//! shows compression preserves ranking.

use fastrq::{Bits, FastRotation, Metric, RotationalQuantizer, RqCode};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const METRICS: [Metric; 3] = [Metric::Cosine, Metric::Dot, Metric::L2];

fn random_unit_vector(d: usize, rng: &mut StdRng) -> Vec<f32> {
    let mut x: Vec<f32> = (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let norm = dot(&x, &x).sqrt();
    if norm > 0.0 {
        for v in &mut x {
            *v /= norm;
        }
    }
    x
}

fn random_uniform_vector(d: usize, rng: &mut StdRng) -> Vec<f32> {
    (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

/// Two d-dimensional unit vectors with cosine similarity `alpha`, with all mass
/// in the first two coordinates (matches Weaviate's `correlatedVectors`).
fn correlated_vectors(d: usize, alpha: f32) -> (Vec<f32>, Vec<f32>) {
    let mut x = vec![0.0f32; d];
    let mut y = vec![0.0f32; d];
    x[0] = 1.0;
    y[0] = alpha;
    y[1] = (1.0 - alpha * alpha).sqrt();
    (x, y)
}

fn dot(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}

fn true_distance(metric: Metric, x: &[f32], y: &[f32]) -> f32 {
    match metric {
        Metric::Cosine => 1.0 - dot(x, y), // assumes unit vectors
        Metric::Dot => -dot(x, y),
        Metric::L2 => dot(x, x) + dot(y, y) - 2.0 * dot(x, y),
    }
}

// ---------------------------------------------------------------------------
// Rotation correctness
// ---------------------------------------------------------------------------

#[test]
fn rotation_is_self_inverse() {
    for &dim in &[64usize, 100, 128, 256, 384, 512, 1000] {
        let rot = FastRotation::new(dim, 3, 42);
        let mut rng = StdRng::seed_from_u64(dim as u64);
        let original = random_uniform_vector(dim, &mut rng);
        let rotated = rot.rotate(&original);
        let back = rot.unrotate(&rotated);
        for i in 0..dim {
            assert!(
                (back[i] - original[i]).abs() < 1e-4,
                "dim {dim} idx {i}: {} vs {}",
                back[i],
                original[i]
            );
        }
    }
}

#[test]
fn rotation_preserves_norm() {
    let rot = FastRotation::new(256, 3, 7);
    let mut rng = StdRng::seed_from_u64(99);
    let x = random_uniform_vector(200, &mut rng);
    let rx = rot.rotate(&x);
    let n0 = dot(&x, &x).sqrt();
    let n1 = dot(&rx, &rx).sqrt();
    assert!((n0 - n1).abs() < 1e-3, "norm not preserved: {n0} vs {n1}");
}

// ---------------------------------------------------------------------------
// Encode / restore
// ---------------------------------------------------------------------------

#[test]
fn restore_rotated_within_quant_step() {
    let mut rng = StdRng::seed_from_u64(7542);
    for _ in 0..10 {
        let d = 2 + rng.gen_range(0..1000);
        let q = RotationalQuantizer::with_seed(d, Bits::Eight, Metric::Cosine, rng.r#gen());
        let s: f32 = 1000.0 * rng.r#gen::<f32>();
        let mut x = random_uniform_vector(d, &mut rng);
        for v in &mut x {
            *v *= s;
        }
        let bound = (s as f64) * (d as f64).sqrt() / 128.0; // Weaviate's bound
        let code = q.encode(&x);
        let target = q.rotation().rotate(&x);
        let restored = q.restore_rotated(&code);
        for i in 0..target.len() {
            assert!(
                (target[i] - restored[i]).abs() as f64 <= bound,
                "d={d} i={i} diff={} bound={bound}",
                (target[i] - restored[i]).abs()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Distance estimate accuracy vs f32 (the headline requirement)
// ---------------------------------------------------------------------------

#[test]
fn distance_estimate_close_to_f32() {
    let mut rng = StdRng::seed_from_u64(6789);
    for _ in 0..250 {
        let d = 2 + rng.gen_range(0..2000);
        let alpha = -1.0 + 2.0 * rng.r#gen::<f32>();
        let (qv, x) = correlated_vectors(d, alpha);
        for metric in METRICS {
            let q = RotationalQuantizer::with_seed(d, Bits::Eight, metric, rng.r#gen());
            let dist = q.query_distancer(&qv);
            let cx = q.encode(&x);
            let estimated = dist.distance(&cx).unwrap();
            let expected = true_distance(metric, &qv, &x);
            // Weaviate's flat bound for its symmetric distancer test is 0.0051.
            // L2 carries twice the dot-estimate error (its `-2*dot` term), so we
            // allow 2x for it.
            let eps = if matches!(metric, Metric::L2) {
                0.0102
            } else {
                0.0051
            };
            assert!(
                (estimated - expected).abs() < eps,
                "metric {metric:?} d={d}: estimated {estimated} vs expected {expected}"
            );
        }
    }
}

#[test]
fn symmetric_and_query_distance_agree() {
    let mut rng = StdRng::seed_from_u64(64521467);
    for _ in 0..100 {
        let d = 2 + rng.gen_range(0..2000);
        for metric in METRICS {
            let q = RotationalQuantizer::with_seed(d, Bits::Eight, metric, rng.r#gen());
            let qv = random_unit_vector(d, &mut rng);
            let xv = random_unit_vector(d, &mut rng);
            let cq = q.encode(&qv);
            let cx = q.encode(&xv);
            let via_query = q.query_distancer(&qv).distance(&cx).unwrap();
            let via_codes = q.distance(&cq, &cx).unwrap();
            assert!(
                (via_query - via_codes).abs() < 2e-6,
                "metric {metric:?} d={d}: {via_query} vs {via_codes}"
            );
        }
    }
}

/// Weaviate's concentration-bound check: the estimator's error should shrink
/// with dimension as roughly `2^-bits / sqrt(d)`.
#[test]
fn estimation_concentration_bounds() {
    let mut rng = StdRng::seed_from_u64(12345);
    for _ in 0..100 {
        let d = 2 + rng.gen_range(0..2000);
        let alpha = -1.0 + 2.0 * rng.r#gen::<f32>();
        let bits = 8.0f64;
        let eps = 2.0f64.powf(-bits) * 5.75 / (d as f64).sqrt() * 1.5;
        let (qv, x) = correlated_vectors(d, alpha);
        let q = RotationalQuantizer::with_seed(d, Bits::Eight, Metric::Dot, rng.r#gen());
        let cx = q.encode(&x);
        let estimate = q.query_distancer(&qv).distance(&cx).unwrap();
        let cos_sim_estimate = -estimate; // dot metric returns negative dot
        assert!(
            (cos_sim_estimate - alpha).abs() as f64 <= eps,
            "d={d} alpha={alpha} estimate={cos_sim_estimate} eps={eps}"
        );
    }
}

/// Codes should use the full 0..=255 range fairly evenly (the rotation
/// Gaussianizes coordinates). Min/max bytes are intentionally over-represented.
#[test]
fn code_point_distribution_is_uniformish() {
    let mut rng = StdRng::seed_from_u64(999);
    let in_dim = 256;
    let q = RotationalQuantizer::with_seed(in_dim, Bits::Eight, Metric::Dot, rng.r#gen());
    let m = 200;
    let mut counts = [0usize; 256];
    for _ in 0..m {
        let x = random_unit_vector(in_dim, &mut rng);
        for &b in q.encode(&x).codes() {
            counts[b as usize] += 1;
        }
    }
    let expectation = (m * q.output_dim()) as f64 / 256.0;
    for (i, &c) in counts.iter().enumerate() {
        if i == 0 || i == 255 {
            continue;
        }
        assert!(c > 0, "byte {i} never used");
        assert!(
            (c as f64) < 3.0 * expectation,
            "byte {i} over-represented: {c}"
        );
    }
}

// ---------------------------------------------------------------------------
// End-to-end: ranking / recall vs exact f32
// ---------------------------------------------------------------------------

/// The practical test of "accuracy vs f32": does searching with compressed codes
/// return (almost) the same nearest neighbors as exact f32 search?
#[test]
fn recall_at_10_vs_exact() {
    let mut rng = StdRng::seed_from_u64(2024);
    let d = 256;
    let n = 2000;
    let queries = 100;
    let k = 10;

    let q = RotationalQuantizer::with_seed(d, Bits::Eight, Metric::Dot, rng.r#gen());
    let data: Vec<Vec<f32>> = (0..n).map(|_| random_unit_vector(d, &mut rng)).collect();
    let codes: Vec<RqCode> = data.iter().map(|v| q.encode(v)).collect();

    let mut total_recall = 0.0f64;
    for _ in 0..queries {
        let query = random_unit_vector(d, &mut rng);

        // Exact top-k by true distance.
        let mut exact: Vec<(usize, f32)> = data
            .iter()
            .enumerate()
            .map(|(i, v)| (i, true_distance(Metric::Dot, &query, v)))
            .collect();
        exact.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let exact_top: std::collections::HashSet<usize> =
            exact.iter().take(k).map(|(i, _)| *i).collect();

        // Approximate top-k by estimated distance.
        let dist = q.query_distancer(&query);
        let mut approx: Vec<(usize, f32)> = codes
            .iter()
            .enumerate()
            .map(|(i, c)| (i, dist.distance(c).unwrap()))
            .collect();
        approx.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let hits = approx
            .iter()
            .take(k)
            .filter(|(i, _)| exact_top.contains(i))
            .count();
        total_recall += hits as f64 / k as f64;
    }

    let recall = total_recall / queries as f64;
    println!("recall@{k} (RQ8 vs exact f32, d={d}, n={n}): {recall:.4}");
    // 8-bit RQ should preserve ranking almost perfectly on random unit vectors.
    assert!(recall > 0.95, "recall@{k} too low: {recall:.4}");
}

/// Report mean absolute error of the dot estimate, for visibility.
#[test]
fn report_mean_absolute_dot_error() {
    let mut rng = StdRng::seed_from_u64(555);
    let d = 512;
    let q = RotationalQuantizer::with_seed(d, Bits::Eight, Metric::Dot, rng.r#gen());
    let trials = 2000;
    let mut sum_abs = 0.0f64;
    let mut max_abs = 0.0f64;
    for _ in 0..trials {
        let a = random_unit_vector(d, &mut rng);
        let b = random_unit_vector(d, &mut rng);
        let est = q.dot_estimate(&q.encode(&a), &q.encode(&b));
        let truth = dot(&a, &b);
        let e = (est - truth).abs() as f64;
        sum_abs += e;
        max_abs = max_abs.max(e);
    }
    let mae = sum_abs / trials as f64;
    println!("dot estimate over {trials} pairs (d={d}): MAE={mae:.6}, max={max_abs:.6}");
    assert!(mae < 0.001, "MAE unexpectedly high: {mae}");
}

// ---------------------------------------------------------------------------
// Degenerate input + persistence
// ---------------------------------------------------------------------------

#[test]
fn handles_abnormal_vectors() {
    let in_dim = 97;
    let q = RotationalQuantizer::with_seed(in_dim, Bits::Eight, Metric::Dot, 42);
    let out_dim = q.output_dim();
    let zero = RqCode::zero(out_dim, Bits::Eight);

    assert_eq!(q.encode(&[]), zero);
    assert_eq!(q.encode(&[0.0f32; 572]), zero);
    assert_eq!(q.encode(&[0.0f32; 15]), zero);

    // Only the first out_dim entries are used.
    let x: Vec<f32> = (0..243).map(|i| i as f32).collect();
    assert_eq!(q.encode(&x[..out_dim]), q.encode(&x));
}

#[test]
fn distance_rejects_codes_of_wrong_dimension() {
    let q = RotationalQuantizer::with_seed(64, Bits::Eight, Metric::Dot, 1);
    assert_eq!(q.output_dim(), 64);

    // Two codes that match each other but not the quantizer (e.g. produced by a
    // different quantizer) must be rejected, not silently mis-scored.
    let wrong = RqCode::zero(128, Bits::Eight);
    assert!(matches!(
        q.distance(&wrong, &wrong),
        Err(fastrq::Error::DimensionMismatch {
            expected: 64,
            actual: 128
        })
    ));

    // Codes of differing lengths are still rejected.
    let right = RqCode::zero(64, Bits::Eight);
    assert!(matches!(
        q.distance(&right, &wrong),
        Err(fastrq::Error::DimensionMismatch { .. })
    ));

    // Correctly-sized codes score fine.
    assert!(q.distance(&right, &right).is_ok());
}

#[test]
fn handles_dimensions_above_u16() {
    // output_dim pads to 65536, which overflows a u16 swap index. Must not panic
    // and must still round-trip the rotation.
    let d = 65_500usize;
    let q = RotationalQuantizer::with_seed(d, Bits::Eight, Metric::Cosine, 1);
    assert_eq!(q.output_dim(), 65_536);

    let mut rng = StdRng::seed_from_u64(1);
    let x = random_unit_vector(d, &mut rng);
    let code = q.encode(&x);
    assert_eq!(code.dimension(), 65_536);

    // Decode (un-rotate) recovers the original within quantization error.
    let decoded = q.decode(&code);
    for i in 0..d {
        assert!(
            (decoded[i] - x[i]).abs() < 1e-2,
            "idx {i}: {} vs {}",
            decoded[i],
            x[i]
        );
    }
}

#[test]
fn distance_bytes_matches_owned() {
    let mut rng = StdRng::seed_from_u64(4242);
    let d = 256;
    let q = RotationalQuantizer::with_seed(d, Bits::Eight, Metric::Cosine, 7);
    let query = random_unit_vector(d, &mut rng);
    let dist = q.query_distancer(&query);

    for _ in 0..50 {
        let code = q.encode(&random_unit_vector(d, &mut rng));
        let owned = dist.distance(&code).unwrap();
        // The alloc-free path over the flat bytes must give an identical result.
        let from_bytes = dist.distance_bytes(&code.to_bytes()).unwrap();
        assert_eq!(owned, from_bytes);
    }

    // Too-short input is rejected, not read out of bounds.
    assert!(dist.distance_bytes(&[0u8; 4]).is_err());
    // Wrong code dimension is rejected.
    let wrong = RqCode::zero(128, Bits::Eight).to_bytes();
    assert!(matches!(
        dist.distance_bytes(&wrong),
        Err(fastrq::Error::DimensionMismatch { .. })
    ));
}

#[test]
fn byte_roundtrip() {
    let mut rng = StdRng::seed_from_u64(321);
    let d = 384;
    let q = RotationalQuantizer::with_seed(d, Bits::Eight, Metric::Cosine, 1);
    let code = q.encode(&random_unit_vector(d, &mut rng));
    let bytes = code.to_bytes();
    assert_eq!(bytes.len(), fastrq::RQ_METADATA_SIZE + q.output_dim());

    // The flat layout is little-endian metadata then code bytes; pin it so the
    // on-disk format can't silently change.
    assert_eq!(&bytes[0..4], &code.lower().to_le_bytes());
    assert_eq!(&bytes[4..8], &code.step().to_le_bytes());
    assert_eq!(&bytes[12..16], &code.norm2().to_le_bytes());
    assert_eq!(&bytes[fastrq::RQ_METADATA_SIZE..], code.codes());

    let restored = RqCode::from_bytes(&bytes, Bits::Eight).unwrap();
    assert_eq!(code, restored);
}

// ---------------------------------------------------------------------------
// Flat-format read/write APIs (encode_to_bytes / code_from_bytes / scan)
// ---------------------------------------------------------------------------

#[test]
fn encode_to_bytes_matches_encode() {
    let mut rng = StdRng::seed_from_u64(808);
    for bits in [Bits::Eight, Bits::Four] {
        let d = 300;
        let q = RotationalQuantizer::with_seed(d, bits, Metric::Dot, 5);
        for _ in 0..20 {
            let x = random_unit_vector(d, &mut rng);
            assert_eq!(q.encode_to_bytes(&x), q.encode(&x).to_bytes());
        }
        // Degenerate inputs produce the zero code in both paths.
        assert_eq!(q.encode_to_bytes(&[]), q.encode(&[]).to_bytes());
        let zeros = vec![0.0f32; d];
        assert_eq!(q.encode_to_bytes(&zeros), q.encode(&zeros).to_bytes());
        assert_eq!(
            q.encode_to_bytes(&x_short()),
            q.encode(&x_short()).to_bytes()
        );
        assert_eq!(q.encode_to_bytes(&[]).len(), q.code_size());
    }
}

fn x_short() -> Vec<f32> {
    vec![0.25f32; 3]
}

#[test]
fn code_from_bytes_validates_and_decodes() {
    let mut rng = StdRng::seed_from_u64(909);
    for bits in [Bits::Eight, Bits::Four] {
        let d = 256;
        let q = RotationalQuantizer::with_seed(d, bits, Metric::Cosine, 3);
        let x = random_unit_vector(d, &mut rng);
        let bytes = q.encode_to_bytes(&x);
        assert_eq!(bytes.len(), q.code_size());

        let code = q.code_from_bytes(&bytes).unwrap();
        assert_eq!(code, q.encode(&x));
        assert_eq!(q.decode_bytes(&bytes).unwrap(), q.decode(&code));

        // Wrong-dimension flat codes are rejected.
        let wrong = RqCode::zero(128, bits).to_bytes();
        assert!(matches!(
            q.code_from_bytes(&wrong),
            Err(fastrq::Error::DimensionMismatch { .. })
        ));
        // Too short for metadata is rejected.
        assert!(q.code_from_bytes(&bytes[..8]).is_err());
    }
}

#[test]
fn distances_bytes_matches_individual() {
    let mut rng = StdRng::seed_from_u64(1111);
    let d = 256;
    for bits in [Bits::Eight, Bits::Four] {
        let q = RotationalQuantizer::with_seed(d, bits, Metric::Dot, 11);
        let dist = q.query_distancer(&random_unit_vector(d, &mut rng));
        let flats: Vec<Vec<u8>> = (0..50)
            .map(|_| q.encode_to_bytes(&random_unit_vector(d, &mut rng)))
            .collect();

        let batch = dist
            .distances_bytes(flats.iter().map(|f| f.as_slice()))
            .unwrap();
        assert_eq!(batch.len(), flats.len());
        for (flat, &score) in flats.iter().zip(&batch) {
            assert_eq!(score, dist.distance_bytes(flat).unwrap());
        }

        // A malformed candidate fails the whole batch.
        assert!(
            dist.distances_bytes([&flats[0][..], &[0u8; 4][..]])
                .is_err()
        );
    }
}

#[test]
fn extensions_and_code_size() {
    assert_eq!(Bits::Eight.extension(), "rq8");
    assert_eq!(Bits::Four.extension(), "rq4");

    let q8 = RotationalQuantizer::with_seed(256, Bits::Eight, Metric::Dot, 1);
    let q4 = RotationalQuantizer::with_seed(256, Bits::Four, Metric::Dot, 1);
    assert_eq!(q8.code_size(), fastrq::RQ_METADATA_SIZE + 256);
    assert_eq!(q4.code_size(), fastrq::RQ_METADATA_SIZE + 128);
}

/// Pin the exact encoded bytes for the default seed. Bifrost (and anyone else
/// persisting flat codes without the quantizer) depends on encode being
/// byte-identical across processes and crate versions; this golden test is
/// what turns that assumption into a contract. If it ever fails, the change
/// is format-breaking and must not ship in a patch/minor release.
#[test]
fn golden_bytes_default_seed() {
    let x: Vec<f32> = (0..8).map(|i| (i as f32 + 1.0) / 10.0).collect();

    let q8 = RotationalQuantizer::new(8, Bits::Eight, Metric::Dot);
    let got8 = q8.encode_to_bytes(&x);
    println!("rq8 golden: {got8:?}");

    let q4 = RotationalQuantizer::new(8, Bits::Four, Metric::Dot);
    let got4 = q4.encode_to_bytes(&x);
    println!("rq4 golden: {got4:?}");

    let expected8: Vec<u8> = vec![
        255, 255, 215, 190, 60, 9, 86, 59, 227, 150, 217, 65, 92, 143, 2, 64, 46, 88, 101, 169, 96,
        53, 130, 190, 243, 137, 210, 131, 112, 130, 192, 61, 104, 67, 102, 105, 103, 60, 132, 156,
        135, 80, 89, 163, 204, 142, 208, 255, 155, 160, 96, 0, 204, 88, 57, 191, 143, 160, 133,
        141, 142, 134, 205, 165, 43, 144, 52, 107, 51, 70, 70, 192, 70, 160, 192, 148, 160, 123,
        146, 232,
    ];
    let expected4: Vec<u8> = vec![
        255, 255, 215, 190, 208, 105, 99, 61, 218, 192, 216, 65, 92, 143, 2, 64, 147, 149, 102, 10,
        198, 83, 56, 187, 142, 152, 140, 136, 135, 136, 203, 164, 54, 132, 54, 102, 54, 68, 72,
        185, 72, 149, 181, 154, 156, 120, 156, 239,
    ];
    assert_eq!(got8, expected8, "rq8 flat encoding changed — format break!");
    assert_eq!(got4, expected4, "rq4 flat encoding changed — format break!");
}

// ---------------------------------------------------------------------------
// 4-bit codes
// ---------------------------------------------------------------------------

#[test]
fn rq4_pack_roundtrip() {
    let mut rng = StdRng::seed_from_u64(4004);
    for _ in 0..10 {
        let d = 2 + rng.gen_range(0..1000);
        let q = RotationalQuantizer::with_seed(d, Bits::Four, Metric::Cosine, rng.r#gen());
        let x = random_unit_vector(d, &mut rng);
        let code = q.encode(&x);

        assert_eq!(code.dimension(), q.output_dim());
        assert_eq!(code.codes().len(), q.output_dim() / 2);

        // Every rotated coordinate must be restored within half a step
        // (rounding) — this fails if packing scrambles dimension order.
        let target = q.rotation().rotate(&x);
        let restored = q.restore_rotated(&code);
        let bound = code.step() * 0.5 + 1e-6;
        for i in 0..target.len() {
            assert!(
                (target[i] - restored[i]).abs() <= bound,
                "d={d} i={i}: {} vs {} (step {})",
                target[i],
                restored[i],
                code.step()
            );
        }
    }
}

#[test]
fn rq4_byte_roundtrip() {
    let mut rng = StdRng::seed_from_u64(4114);
    let d = 384;
    let q = RotationalQuantizer::with_seed(d, Bits::Four, Metric::Cosine, 1);
    let code = q.encode(&random_unit_vector(d, &mut rng));
    let bytes = code.to_bytes();
    assert_eq!(bytes.len(), fastrq::RQ_METADATA_SIZE + q.output_dim() / 2);

    let restored = RqCode::from_bytes(&bytes, Bits::Four).unwrap();
    assert_eq!(code, restored);
}

/// The distancer's asymmetric path must equal scoring its 8-bit query code
/// against the 4-bit candidate through the plain code-to-code API.
#[test]
fn rq4_query_distancer_matches_mixed_distance() {
    let mut rng = StdRng::seed_from_u64(4224);
    let d = 256;
    for metric in METRICS {
        let q = RotationalQuantizer::with_seed(d, Bits::Four, metric, 21);
        let query = random_unit_vector(d, &mut rng);
        let dist = q.query_distancer(&query);
        assert_eq!(dist.query_code().bits(), Bits::Eight);
        for _ in 0..20 {
            let c = q.encode(&random_unit_vector(d, &mut rng));
            assert_eq!(
                dist.distance(&c).unwrap(),
                q.distance(dist.query_code(), &c).unwrap()
            );
        }
    }
}

#[test]
fn rq4_distance_estimate_close_to_f32() {
    let mut rng = StdRng::seed_from_u64(4334);
    let mut max_err = [0.0f32; 3];
    for _ in 0..250 {
        let d = 2 + rng.gen_range(0..2000);
        let alpha = -1.0 + 2.0 * rng.r#gen::<f32>();
        let (qv, x) = correlated_vectors(d, alpha);
        for (mi, metric) in METRICS.iter().enumerate() {
            let q = RotationalQuantizer::with_seed(d, Bits::Four, *metric, rng.r#gen());
            let dist = q.query_distancer(&qv);
            let estimated = dist.distance(&q.encode(&x)).unwrap();
            let expected = true_distance(*metric, &qv, &x);
            max_err[mi] = max_err[mi].max((estimated - expected).abs());
            // Empirical bounds at ~2.5x the observed max over this seed
            // (cos/dot 0.0164, L2 0.0443). The 4-bit step is 17x the 8-bit
            // one, but only the data side pays it — the query is 8-bit.
            let eps = if matches!(metric, Metric::L2) {
                0.10
            } else {
                0.04
            };
            assert!(
                (estimated - expected).abs() < eps,
                "metric {metric:?} d={d}: estimated {estimated} vs expected {expected}"
            );
        }
    }
    println!("rq4 max errors (cos, dot, l2): {max_err:?}");
}

/// The asymmetric 8-bit query must beat scoring with a 4-bit query code —
/// the entire point of encoding the query finer than the data.
#[test]
fn rq4_asymmetric_query_beats_symmetric() {
    let mut rng = StdRng::seed_from_u64(4444);
    let d = 512;
    let q = RotationalQuantizer::with_seed(d, Bits::Four, Metric::Dot, 17);
    let trials = 1000;
    let (mut mae_asym, mut mae_sym) = (0.0f64, 0.0f64);
    for _ in 0..trials {
        let a = random_unit_vector(d, &mut rng);
        let b = random_unit_vector(d, &mut rng);
        let truth = -dot(&a, &b) as f64;

        let asym = q.query_distancer(&a).distance(&q.encode(&b)).unwrap();
        let sym = q.distance(&q.encode(&a), &q.encode(&b)).unwrap();
        mae_asym += (asym as f64 - truth).abs();
        mae_sym += (sym as f64 - truth).abs();
    }
    mae_asym /= trials as f64;
    mae_sym /= trials as f64;
    println!("rq4 dot MAE (d={d}): asymmetric={mae_asym:.6}, symmetric={mae_sym:.6}");
    assert!(
        mae_asym < mae_sym,
        "asymmetric ({mae_asym}) should beat symmetric ({mae_sym})"
    );
}

#[test]
fn rq4_recall_at_10_vs_exact() {
    let mut rng = StdRng::seed_from_u64(2025);
    let d = 256;
    let n = 2000;
    let queries = 100;
    let k = 10;

    let q = RotationalQuantizer::with_seed(d, Bits::Four, Metric::Dot, rng.r#gen());
    let data: Vec<Vec<f32>> = (0..n).map(|_| random_unit_vector(d, &mut rng)).collect();
    let codes: Vec<RqCode> = data.iter().map(|v| q.encode(v)).collect();

    let mut total_recall = 0.0f64;
    for _ in 0..queries {
        let query = random_unit_vector(d, &mut rng);

        let mut exact: Vec<(usize, f32)> = data
            .iter()
            .enumerate()
            .map(|(i, v)| (i, true_distance(Metric::Dot, &query, v)))
            .collect();
        exact.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let exact_top: std::collections::HashSet<usize> =
            exact.iter().take(k).map(|(i, _)| *i).collect();

        let dist = q.query_distancer(&query);
        let mut approx: Vec<(usize, f32)> = codes
            .iter()
            .enumerate()
            .map(|(i, c)| (i, dist.distance(c).unwrap()))
            .collect();
        approx.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let hits = approx
            .iter()
            .take(k)
            .filter(|(i, _)| exact_top.contains(i))
            .count();
        total_recall += hits as f64 / k as f64;
    }

    let recall = total_recall / queries as f64;
    println!("recall@{k} (RQ4 vs exact f32, d={d}, n={n}): {recall:.4}");
    assert!(recall > 0.80, "recall@{k} too low: {recall:.4}");
}

#[cfg(feature = "serde")]
#[test]
fn serde_bincode_roundtrip() {
    let mut rng = StdRng::seed_from_u64(123);
    let d = 256;
    let q = RotationalQuantizer::with_seed(d, Bits::Eight, Metric::L2, 9);
    let code = q.encode(&random_unit_vector(d, &mut rng));

    // The quantizer (incl. its rotation) and codes must survive a bincode round
    // trip so bifrost can persist them.
    let q_bytes = bincode::serialize(&q).unwrap();
    let q2: RotationalQuantizer = bincode::deserialize(&q_bytes).unwrap();
    assert_eq!(q, q2);

    let c_bytes = bincode::serialize(&code).unwrap();
    let c2: RqCode = bincode::deserialize(&c_bytes).unwrap();
    assert_eq!(code, c2);

    // And the reloaded quantizer estimates identical distances.
    let other = q.encode(&random_unit_vector(d, &mut rng));
    assert_eq!(
        q.distance(&code, &other).unwrap(),
        q2.distance(&c2, &other).unwrap()
    );
}
