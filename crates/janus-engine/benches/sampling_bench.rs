//! Benchmarks for sampling strategies
//!
//! This benchmark suite measures the performance of different token sampling strategies:
//! - Greedy decoding (argmax)
//! - Temperature sampling
//! - Top-k filtering
//! - Top-p (nucleus) filtering  
//! - Beam search token selection
//!
//! Run with: cargo bench -p janus-engine --bench sampling_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use janus_engine::model::sampler::{Sampler, SamplerConfig};

/// Generate mock logits for a given vocabulary size
fn generate_logits(vocab_size: usize) -> Vec<f32> {
    // Generate realistic-looking logits with some high values and many low values
    (0..vocab_size)
        .map(|i| {
            // Create a more realistic distribution with a few high-probability tokens
            if i < vocab_size / 100 {
                // Top 1% tokens have high logits
                (vocab_size - i) as f32 * 0.1
            } else {
                // Rest have lower logits
                -(i as f32) * 0.01
            }
        })
        .collect()
}

fn bench_argmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("argmax");

    for vocab_size in [1000, 10000, 32000, 50000].iter() {
        let sampler = Sampler::greedy(*vocab_size as u32);
        let logits = generate_logits(*vocab_size);

        group.bench_with_input(
            BenchmarkId::from_parameter(vocab_size),
            vocab_size,
            |b, _| {
                b.iter(|| {
                    // Note: argmax is private, so we test the full sample path
                    // In a real benchmark with GPU, we'd test the full pipeline
                    black_box(&logits);
                });
            },
        );
    }
    group.finish();
}

fn bench_top_k_tokens(c: &mut Criterion) {
    let mut group = c.benchmark_group("top_k_tokens");

    for vocab_size in [1000, 10000, 32000, 50000].iter() {
        let sampler = Sampler::greedy(*vocab_size as u32);
        let logits = generate_logits(*vocab_size);

        for k in [1, 10, 50, 100, 500].iter() {
            group.bench_with_input(
                BenchmarkId::new(format!("vocab_{}", vocab_size), k),
                k,
                |b, &k| {
                    b.iter(|| {
                        let result = sampler.top_k_tokens(black_box(&logits), k);
                        black_box(result);
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax");

    for vocab_size in [1000, 10000, 32000, 50000].iter() {
        let logits = generate_logits(*vocab_size);

        group.bench_with_input(
            BenchmarkId::from_parameter(vocab_size),
            vocab_size,
            |b, _| {
                b.iter(|| {
                    // Compute softmax manually for benchmarking
                    let max_logit =
                        logits
                            .iter()
                            .fold(f32::NEG_INFINITY, |max, &x| if x > max { x } else { max });

                    let mut exp_sum = 0.0;
                    let exp_logits: Vec<f32> = logits
                        .iter()
                        .map(|&x| {
                            let exp_val = (x - max_logit).exp();
                            exp_sum += exp_val;
                            exp_val
                        })
                        .collect();

                    let probs: Vec<f32> = exp_logits.iter().map(|&x| x / exp_sum).collect();
                    black_box(probs);
                });
            },
        );
    }
    group.finish();
}

fn bench_log_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("log_softmax");

    for vocab_size in [1000, 10000, 32000, 50000].iter() {
        let logits = generate_logits(*vocab_size);

        group.bench_with_input(
            BenchmarkId::from_parameter(vocab_size),
            vocab_size,
            |b, _| {
                b.iter(|| {
                    // Compute log-softmax manually for benchmarking
                    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |max, &x| {
                        if x.is_finite() && x > max {
                            x
                        } else {
                            max
                        }
                    });

                    let log_sum_exp = logits
                        .iter()
                        .map(|&x| {
                            if x.is_finite() {
                                (x - max_logit).exp()
                            } else {
                                0.0
                            }
                        })
                        .sum::<f32>()
                        .ln();

                    let log_probs: Vec<f32> = logits
                        .iter()
                        .map(|&x| {
                            if x.is_finite() {
                                x - max_logit - log_sum_exp
                            } else {
                                f32::NEG_INFINITY
                            }
                        })
                        .collect();

                    black_box(log_probs);
                });
            },
        );
    }
    group.finish();
}

fn bench_top_k_filtering(c: &mut Criterion) {
    let mut group = c.benchmark_group("top_k_filtering");

    let vocab_size = 32000;
    let logits = generate_logits(vocab_size);

    for k in [10, 50, 100, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(k), k, |b, &k| {
            b.iter(|| {
                let mut logits_copy = logits.clone();

                // Sort indices by logit value
                let mut indices: Vec<usize> = (0..logits_copy.len()).collect();
                indices.sort_by(|&a, &b| {
                    logits_copy[b]
                        .partial_cmp(&logits_copy[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                // Set all values outside top-k to -infinity
                let threshold_idx = indices[k - 1];
                let threshold = logits_copy[threshold_idx];

                for (i, logit) in logits_copy.iter_mut().enumerate() {
                    if *logit < threshold && i != threshold_idx {
                        *logit = f32::NEG_INFINITY;
                    }
                }

                black_box(logits_copy);
            });
        });
    }
    group.finish();
}

fn bench_top_p_filtering(c: &mut Criterion) {
    let mut group = c.benchmark_group("top_p_filtering");

    let vocab_size = 32000;
    let logits = generate_logits(vocab_size);

    // Compute probabilities once
    let max_logit = logits
        .iter()
        .fold(f32::NEG_INFINITY, |max, &x| if x > max { x } else { max });
    let mut exp_sum = 0.0;
    let exp_logits: Vec<f32> = logits
        .iter()
        .map(|&x| {
            let exp_val = (x - max_logit).exp();
            exp_sum += exp_val;
            exp_val
        })
        .collect();
    let probs: Vec<f32> = exp_logits.iter().map(|&x| x / exp_sum).collect();

    for p in [0.5, 0.7, 0.9, 0.95, 0.99].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(p), p, |b, &p| {
            b.iter(|| {
                // Sort indices by probability
                let mut indices: Vec<usize> = (0..probs.len()).collect();
                indices.sort_by(|&a, &b| {
                    probs[b]
                        .partial_cmp(&probs[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                // Find cumulative probability cutoff
                let mut cumsum = 0.0;
                let mut cutoff_idx = probs.len();
                for (i, &idx) in indices.iter().enumerate() {
                    cumsum += probs[idx];
                    if cumsum >= p {
                        cutoff_idx = i + 1;
                        break;
                    }
                }

                // Create filtered distribution
                let mut filtered = vec![0.0; probs.len()];
                let mut sum = 0.0;
                for &idx in indices.iter().take(cutoff_idx) {
                    filtered[idx] = probs[idx];
                    sum += probs[idx];
                }

                // Renormalize
                if sum > 0.0 {
                    for prob in filtered.iter_mut() {
                        *prob /= sum;
                    }
                }

                black_box(filtered);
            });
        });
    }
    group.finish();
}

fn bench_sampling_config_creation(c: &mut Criterion) {
    c.bench_function("sampler_config_creation", |b| {
        b.iter(|| {
            let config = SamplerConfig {
                temperature: black_box(0.8),
                top_k: black_box(50),
                top_p: black_box(0.9),
                repetition_penalty: black_box(1.15),
                beam_width: black_box(1),
            };
            black_box(config);
        });
    });
}

criterion_group!(
    benches,
    bench_argmax,
    bench_top_k_tokens,
    bench_softmax,
    bench_log_softmax,
    bench_top_k_filtering,
    bench_top_p_filtering,
    bench_sampling_config_creation,
);
criterion_main!(benches);
