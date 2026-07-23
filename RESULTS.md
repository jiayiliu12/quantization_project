# vLLM Throughput Benchmark — `sid-tech/sid-1-fp8-x-baseten`

> Status: **COMPLETE** — all runs finished; data saved under `raw/`.
> Last updated: 2026-06-18.
>
> **TL;DR:** 1× B200 = **2,593 out tok/s** (peak). 8× H100 (8× TP=1 replicas) = **10,090 out tok/s** (best whole-node, beats 4× TP=2's 9,847). B200 ≈ **2× per-GPU** vs H100. TP=1 ≥ TP=2 for throughput/GPU. See Recommendation at the bottom.

## Model
- `sid-tech/sid-1-fp8-x-baseten` — FP8-quantized, based on Qwen3-14B (vocab 151,643, `max_position_embeddings` 40,960).
- Served via `vllm/vllm-openai:v0.19.0`, OpenAI-compatible API, served name `sid-1`.
- Tool calling: `--enable-auto-tool-choice --tool-call-parser hermes`, model's own chat template (extracted from tokenizer, 2,121 chars, supports `<think>`).

## Hardware
| Box | SSH host | GPUs | Notes |
|-----|----------|------|-------|
| H100 cluster | `mithril_corporate_shark` | 8× H100 80GB HBM3, full NVLink (NV18/NVSwitch) | All 8 idle/free for benchmarking |
| B200 | `faithful-emu` | 1× B200 183GB | |
| (prod, untouched) | `gpu3` | 2× H100 NVL 94GB | Production host — read-only inspected, **not** benchmarked |

The prod launch config on `gpu3` is identical to the provided recipe (TP=2, same flags) — confirmed by `docker inspect`.

## Workload (identical across every run)
Multi-turn conversational benchmark via the image's `benchmarks/multi_turn/benchmark_serving_multi_turn.py`.

| Parameter | Value | Mapping |
|-----------|-------|---------|
| Turns per conversation | Uniform[3, 4] **exchanges** | `num_turns` ~ Uniform[6, 8] half-turns (user+assistant), rounded even |
| New input tokens / turn | 2000 | `num_tokens` = 2000 (constant) — new user message each turn |
| Output tokens / turn | 150 | `prompt_output.num_tokens` = 150 (constant) |
| Cache hit rate target | ~50% | shared system prefix (`common_prefix_num_tokens`) + within-conversation history reuse; **calibrated empirically** (see below) |
| Conversations / run | 500 | `num_conversations` = 500 |
| Sampling | seed 12345, fixed | identical generated dataset across all hardware configs |

**Load model:** max-throughput sweep — concurrency (`--num-clients` = `--max-active-conversations`) stepped 16 → 32 → 64 → 128 → 256 until the server saturates. `temperature=0`, streaming on.

### On the "cache hit rate"
Two numbers are reported:
- **`approx_cached_percent`** (benchmark-side): mean over requests of `history_tokens / input_tokens`. Deterministic — a pure function of the generated dataset, identical across all hardware. This is the controlled cache variable.
- **server prefix-cache hit %** (engine-side): vLLM's `prefix_cache_hits_total / prefix_cache_queries_total` over the run. Reflects real KV reuse (can vary with eviction at high concurrency).

Calibration results and the final locked config are recorded below once measured.

## Metrics reported per run
- Throughput: requests/s, **output tokens/s**, total tokens/s
- Latency: TTFT (mean/p50/p90/p99), TPOT (per-output-token), end-to-end per request
- Achieved cache hit % (both definitions), mean input/output tokens, mean turns

---

## Run matrix
| # | Box | Config | Status | Peak out tok/s |
|---|-----|--------|--------|----------------|
| 1 | H100 | TP=2 (recipe), 2 GPUs | ✅ done | 2,527 |
| 2 | H100 | TP=1, 1 GPU | ✅ done | 1,280 |
| 3 | H100 | throughput search: 8× TP=1 vs 4× TP=2 (8 GPUs) | ✅ done | **10,090** (8× TP=1) |
| 4 | B200 | best single-GPU config (+ tuning pass) | ✅ done | **2,593** |

---

## Calibration (locked config)
Measured on B200 @ concurrency 16, 64-conversation calibration set, seed 12345.

| common_prefix | server-side prefix-cache hit % | notes |
|---------------|-------------------------------|-------|
| 1024 tokens | 64.3% | shared system-prompt inflates cross-conversation hits |
| **0 tokens (LOCKED)** | **57.8%** | pure multi-turn history reuse — closest controllable to 50% |

**On the 50% target:** with turns fixed at 3–4 exchanges and 2000 new tokens/turn, cache reuse is intrinsic to the conversation structure. The achievable range is ~44% (benchmark per-request mean of `history/input`, which weights the always-uncached first turn) up to ~58% (token-weighted / engine-reported). The workload therefore brackets ~50%; exact 50% engine-side is not reachable without changing the turn/token spec. Locked config uses `common_prefix=0` → engine hit ≈ 58%, the value closest to target. Dataset is seed-fixed, so cache behavior is **identical across all hardware configs** (the comparison is apples-to-apples).

Verified workload (generated): mean 3.75 exchanges, 2000-token user turns, 150 output tokens, max input 8446 tokens.

**Locked `gen_main.json`:** `num_turns`~U[6,8], `num_tokens`=2000, `output`=150, `common_prefix`=0, `prefix`=0, `num_conversations`=500, seed=12345.

## Results

### B200 (1× B200 183GB, TP=1) — DONE
KV cache: **935,312 tokens**. 500 conversations/level, seed 12345. cache hit % = engine prefix-cache hit rate over the run.

| Concurrency | req/s | **out tok/s** | TTFT mean / p99 (ms) | TPOT mean / p99 (ms) | E2E mean / p99 (ms) | cache hit % |
|---|---|---|---|---|---|---|
| 16  | 10.49 | 1573 | 248 / 626 | 8.5 / 10.3 | 1507 / 1724 | 61.0 |
| 32  | 13.01 | 1951 | 586 / 1254 | 12.4 / 17.1 | 2440 / 2919 | 59.1 |
| 64  | 15.50 | 2325 | 714 / 2069 | 22.6 / 30.1 | 4084 / 4967 | 58.6 |
| **128** | **17.29** | **2593** | 822 / 4137 | 43.7 / 55.0 | 7327 / 8954 | 57.4 |
| 256 | 11.37 | 1705 | 8392 / 21729 | 93.4 / 140.6 | 22310 / 41218 | 7.5 |

- **Peak throughput: ~2593 output tok/s at concurrency 128.**
- **Best latency/throughput balance: c64** (2325 tok/s, p99 TTFT 2.1s, p99 E2E 5.0s).
- **c256 collapses**: 256 concurrent conversations (avg ~4.7k input tok each) oversubscribe the 935k-token KV cache → preemption/recompute, cache hit rate crashes to 7.5%, p99 latency explodes to 41s. Capping `--max-num-seqs`≈128 would prevent this collapse (would plateau at peak rather than degrade).

**B200 tuning experiment** (`--gpu-memory-utilization 0.95 --max-num-seqs 192`): peak unchanged at **2597 tok/s @ c128** (c128: 17.32 req/s; c192: 11.64; c256: 11.49). 192 concurrent conversations already oversubscribe the KV cache, so raising the cap doesn't add throughput — it only bounds the in-flight set so the c256 latency thrash is less catastrophic. **Conclusion: best single-B200 config = the recipe (TP=1, defaults); operate at concurrency ≈128; set `--max-num-seqs`≈128 in production to avoid the over-subscription collapse.**

### H100 TP=2 (2× H100 80GB, GPUs 0–1, the recipe) — DONE
KV cache: **777,216 tokens**. 500 conversations/level, seed 12345.

| Concurrency | req/s | **out tok/s** | TTFT mean / p99 (ms) | TPOT mean (ms) | E2E mean / p99 (ms) | cache hit % |
|---|---|---|---|---|---|---|
| 16  | 9.68  | 1452 | 317 / 736   | 8.8  | 1633 / 1954 | 64.9 |
| 32  | 12.60 | 1890 | 628 / 1264  | 12.7 | 2519 / 3108 | 64.9 |
| 64  | 15.02 | 2253 | 849 / 2085  | 22.6 | 4214 / 5533 | 64.6 |
| **128** | **16.85** | **2527** | 1117 / 4532 | 42.9 | 7513 / 10815 | 64.1 |
| 256 | 11.85 | 1777 | 8588 / 20890 | 85.9 | 21380 / 40559 | 25.7 |

- **Peak: ~2527 out tok/s @ c128** (2 GPUs). c64 is the latency sweet spot (2253 tok/s, p99 E2E 5.5s). Same c256 KV-oversubscription collapse.
- **A single B200 (2593 tok/s) ≈ slightly beats two H100s in TP=2 (2527 tok/s)** for this workload → ~2× per-GPU throughput advantage for B200.

### H100 TP=1 (1× H100 80GB, GPU 2) — DONE
KV cache smaller than TP=2 (single GPU) → over-subscribes earlier.

| Concurrency | req/s | **out tok/s** | TTFT mean / p99 (ms) | TPOT mean (ms) | E2E mean / p99 (ms) | cache hit % |
|---|---|---|---|---|---|---|
| 16  | 6.00 | 900  | 575 / 1117 | 13.9 | 2644 / 3170 | 64.9 |
| 32  | 7.51 | 1126 | 982 / 1968 | 21.9 | 4238 / 5307 | 64.9 |
| **64**  | **8.53** | **1280** | 1173 / 6567 | 42.0 | 7432 / 15741 | 59.1 |
| 128 | 6.01 | 901  | 8571 / 18448 | 84.5 | 21163 / 33363 | 20.1 |
| 256 | 6.51 | 977  | 25568 / 51777 | 88.7 | 38782 / 67094 | 19.6 |

- **Peak: ~1280 out tok/s @ c64** (1 GPU). Collapses at c128 (smaller KV cache oversubscribes 2× sooner than TP=2/B200).
- **Per-GPU throughput: TP=1 (1280) ≈ TP=2 (1264/GPU)** — tensor-parallelism here adds latency headroom + a bigger shared KV cache, but **not** more throughput per GPU.

### Per-GPU summary (peak)
| Config | tok/s per GPU | best concurrency / GPU |
|---|---|---|
| B200 | **~2593** | 128 |
| H100 TP=1 | ~1280 | 64 |
| H100 TP=2 | ~1264 | 64 (c128 total) |

→ **B200 ≈ 2× an H100 per GPU** on this workload.

### H100 throughput search (8 GPUs)
Goal: best whole-node config for aggregate throughput. Method: N independent replicas, each driven by its own benchmark client in parallel (direct, no LB), throughput summed. 300 conv/replica, c = per-replica concurrency.

| Config | replicas × GPUs | per-replica conc | **aggregate out tok/s** | aggregate req/s | scaling eff. | per-replica E2E mean |
|---|---|---|---|---|---|---|
| **8× TP=1** | 8 × 1 GPU | 64 | **10,090** | 67.3 | 98.5% (vs 8×single) | 7554 ms |
| 4× TP=2 | 4 × 2 GPU | 128 | 9,847 | 65.6 | 97.4% (vs 4×single) | 7636 ms |
| 5× TP=1 (nginx LB, single endpoint) | 5 × 1 GPU | 160 total | ~5,080 (33.9 req/s) | — | — | — |

- **8× TP=1 scales near-linearly** (10,090 vs 8×1280 = 10,240 ideal; 98.5%). Each replica held ~1261 tok/s with identical latency to the standalone TP=1 run — replicas are independent (no cross-replica interference; full NVLink unused since TP=1).
- The nginx single-endpoint path works to ~c160 but disconnects under higher concurrency (LB tuning issue, not a model limit). For a single production endpoint, an LB with proper keepalive/`proxy_http_version 1.1` + higher worker_connections is needed; for raw capacity, independent replicas are the ceiling.

---

## Cross-config comparison (peak aggregate throughput)

| Deployment | GPUs | Peak out tok/s | tok/s per GPU |
|---|---|---|---|
| 1× B200 (TP=1) | 1 | 2,593 | **2,593** |
| 8× H100 (8× TP=1 replicas) | 8 | **10,090** | 1,261 |
| **8× H100 (8× TP=1 replicas)** | 8 | **10,090** | 1,261 |
| 4× H100 pairs (4× TP=2) | 8 | 9,847 | 1,231 |
| 2× H100 (TP=2, the recipe) | 2 | 2,527 | 1,264 |
| 1× H100 (TP=1) | 1 | 1,280 | 1,280 |

(All at the locked workload: U[3–4] turns, 2000 in/turn, 150 out/turn, ~58% engine cache hit, 500 conv/sweep; aggregates 300 conv/replica.)

## Recommendation

1. **Per-GPU, the B200 is ~2× an H100** on this workload (2,593 vs ~1,270 tok/s/GPU). A single B200 ≈ two H100s in TP=2 (the recipe). So **4× B200 ≈ 8× H100** for throughput (~10.4k vs 10.1k tok/s). Choose on $/GPU: if a B200 costs less than ~2× an H100, B200 is the better throughput buy; it's also far simpler operationally (1 GPU, no tensor-parallel, no multi-replica LB).

2. **Best whole-node H100 config = 8× TP=1 replicas (10,090 tok/s)**, marginally ahead of 4× TP=2 (9,847). Tensor-parallelism gives **no per-GPU throughput gain** here (the model is small enough for one GPU); TP=2's value is lower latency and a larger shared KV cache, not throughput. For max throughput, run independent TP=1 replicas; for a single low-latency endpoint, the recipe (TP=2) is fine.

3. **The recipe (TP=2) is a reasonable default** — good latency, 2,527 tok/s on 2 GPUs — but for a throughput-oriented fleet, prefer **1 replica per GPU (TP=1)** behind a properly-tuned load balancer (keepalive + `proxy_http_version 1.1` + higher `worker_connections`; the stock nginx config disconnected above ~c160).

4. **Operating point / avoid the cliff:** throughput peaks then *collapses* once concurrent conversations oversubscribe the KV cache (B200 at c256, H100 TP=2 at c256, H100 TP=1 at c128 — latency jumps 5–10× and cache-hit craters). Set **`--max-num-seqs`** to the peak batch (≈128 B200 / ≈64 single-H100-TP=1) so excess load queues gracefully instead of thrashing.

5. **Sweet spots (latency-bound serving):** B200 c64 (2,325 tok/s, p99 E2E 5.0s); H100 TP=2 c64 (2,253 tok/s, p99 E2E 5.5s).

### TL;DR
- **Max throughput, 1 GPU:** B200 @ c128 → **2,593 tok/s**.
- **Max throughput, 8× H100 node:** 8× TP=1 replicas → **10,090 tok/s**.
- **B200 ≈ 2× H100 per GPU.** TP=1 ≥ TP=2 for throughput/GPU.
- Cache hit rate achieved ≈ **58%** engine-side (target 50%; intrinsic to 3–4 turn reuse).

---
### Reproducibility
- Image `vllm/vllm-openai:v0.19.0`; serve flags per the provided recipe (TP=2) / single-GPU (no TP) / replicas (TP=1).
- Workload config `gen_main.json` and benchmark via `benchmarks/multi_turn/benchmark_serving_multi_turn.py` (client patched to send `Authorization: Bearer` + full-width stats). Raw per-run logs/metrics under `raw/`.
- Note: H100 multi-replica aggregates used 300 conv/replica (parallel direct clients, throughput summed); single-GPU/recipe sweeps used 500 conv. Cache hit % is engine `prefix_cache_hits/queries`; B200 vs H100 differ slightly (57–65%) due to KV size / run dynamics, but the generated dataset (and thus offered cache reuse) is identical across all configs.
