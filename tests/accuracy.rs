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
    let zero = RqCode::zero(out_dim);

    assert_eq!(q.encode(&[]), zero);
    assert_eq!(q.encode(&[0.0f32; 572]), zero);
    assert_eq!(q.encode(&[0.0f32; 15]), zero);

    // Only the first out_dim entries are used.
    let x: Vec<f32> = (0..243).map(|i| i as f32).collect();
    assert_eq!(q.encode(&x[..out_dim]), q.encode(&x));
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

    let restored = RqCode::from_bytes(&bytes).unwrap();
    assert_eq!(code, restored);
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
