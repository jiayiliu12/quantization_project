"""
Multi-turn serving throughput benchmark for the two Qwen3.6-35B-A3B servers
(NVFP4 vs BF16) deployed by qwen36_compare_modal.py.

Runs vLLM's benchmarks/multi_turn/benchmark_serving_multi_turn.py from INSIDE
Modal (same backbone as the servers) so client<->server latency doesn't skew
TTFT / throughput. The client hits the public Modal web endpoints.

Usage:
    modal run qwen36_bench_multiturn.py --model nvfp4 --concurrency 64
    modal run qwen36_bench_multiturn.py --model bf16  --concurrency 64
    modal run qwen36_bench_multiturn.py --model both  --concurrency 64

Workload (identical across every run; dataset fixed by seed + shared tokenizer):
    500 conversations, 3-4 exchanges each (num_turns ~ Uniform[6,8] messages),
    2000 new input tokens/turn, 150 output tokens/turn, streaming, temp n/a
    (script has no temperature flag; output is length-capped so throughput is
    unaffected). common_prefix_num_tokens is the cache-hit lever (calibrate).
"""

import json
import subprocess
import time

import modal

# Same pinned vLLM build as the servers, so tokenizer + client match the stack.
VLLM_IMAGE = "vllm/vllm-openai@sha256:397273c7a694aeccd3b081d3fb07f4c2983958f22cdfec9c7cde3d70b828224d"
MT = "https://raw.githubusercontent.com/vllm-project/vllm/main/benchmarks/multi_turn"

# One tokenizer for BOTH models -> byte-identical generated dataset per seed.
# (NVFP4 is a quantization of this same model; tokenizer is identical.)
TOKENIZER = "Qwen/Qwen3.6-35B-A3B"

ENDPOINTS = {
    "nvfp4": ("https://sidhq--qwen36-compare-serve-nvfp4.modal.run",
              "nvidia/Qwen3.6-35B-A3B-NVFP4"),
    "bf16": ("https://sidhq--qwen36-compare-serve-bf16.modal.run",
             "Qwen/Qwen3.6-35B-A3B"),
}

bench_image = (
    modal.Image.from_registry(VLLM_IMAGE, add_python="3.12")
    .entrypoint([])
    .apt_install("wget")
    .pip_install(
        "numpy>=1.24", "pandas>=2.0.0", "aiohttp>=3.10", "transformers>=4.46",
        "xlsxwriter>=3.2.1", "tqdm>=4.66", "huggingface_hub[hf_transfer]",
    )
    .run_commands(
        "mkdir -p /bench",
        f"cd /bench && wget -q {MT}/benchmark_serving_multi_turn.py "
        f"{MT}/bench_dataset.py {MT}/bench_utils.py",
        # Deterministic corpus the synthetic generator samples tokens from.
        "cd /bench && wget -q -O pg1184.txt "
        "https://www.gutenberg.org/cache/epub/1184/pg1184.txt",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
app = modal.App("qwen36-bench-mt")


def _config(common_prefix: int, num_conversations: int) -> dict:
    return {
        "filetype": "generate_conversations",
        "num_conversations": num_conversations,
        "text_files": ["pg1184.txt"],
        "print_stats": True,
        "prompt_input": {
            "num_turns": {"distribution": "uniform", "min": 6, "max": 8},
            "common_prefix_num_tokens": {"distribution": "constant", "value": common_prefix},
            "prefix_num_tokens": {"distribution": "constant", "value": 0},
            "num_tokens": {"distribution": "constant", "value": 2000},
        },
        "prompt_output": {
            "num_tokens": {"distribution": "constant", "value": 150},
        },
    }


def _prefix_cache_counters(metrics_text: str) -> dict:
    """Pull vLLM prefix-cache query/hit counters out of Prometheus /metrics."""
    out = {}
    for line in metrics_text.splitlines():
        if line.startswith("#"):
            continue
        for key in ("vllm:prefix_cache_queries_total",
                    "vllm:prefix_cache_hits_total",
                    "vllm:gpu_prefix_cache_queries",
                    "vllm:gpu_prefix_cache_hits"):
            if line.startswith(key):
                try:
                    out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
                except Exception:
                    pass
    return out


@app.function(image=bench_image, timeout=60 * 60,
              volumes={"/root/.cache/huggingface": hf_cache})
def run_bench(model_key: str, concurrency: int, common_prefix: int,
              num_conversations: int, seed: int) -> dict:
    import urllib.request

    url, served_name = ENDPOINTS[model_key]

    def _get(path: str, timeout: int = 15) -> str:
        with urllib.request.urlopen(f"{url}{path}", timeout=timeout) as r:
            return r.read().decode()

    # Wait for the (possibly cold) server to be ready before hammering it.
    print(f"[{model_key}] waiting for {url}/v1/models ...", flush=True)
    deadline = time.time() + 30 * 60
    while time.time() < deadline:
        try:
            _get("/v1/models")
            print(f"[{model_key}] server ready.", flush=True)
            break
        except Exception:
            time.sleep(10)
    else:
        raise RuntimeError(f"{model_key} server never became ready")

    with open("/bench/config.json", "w") as f:
        json.dump(_config(common_prefix, num_conversations), f)

    # Patch the vendored client to count reasoning_content as output tokens.
    # With the qwen3 reasoning parser ON and only 150 output tokens, the model
    # never closes </think>, so vLLM streams delta.reasoning_content (not
    # delta.content). The stock script reads only delta["content"] -> counts 0
    # tokens -> marks every request invalid -> empty stats. Mapping
    # reasoning_content into content fixes measurement without changing what the
    # GPU generates (token throughput / TTFT / TPOT are identical either way).
    # vLLM's qwen3 reasoning parser streams thinking tokens under delta["reasoning"]
    # (NOT "reasoning_content" and NOT "content"). Map whichever is present into
    # "content" so the stock counter sees the tokens.
    script = "/bench/benchmark_serving_multi_turn.py"
    with open(script) as f:
        src = f.read()
    anchor = '                    delta = data["choices"][0]["delta"]\n'
    inject = (
        anchor
        + '                    _MT_PATCH = delta.get("reasoning") or delta.get("reasoning_content")\n'
        + '                    if not delta.get("content") and _MT_PATCH:\n'
        + '                        delta["content"] = _MT_PATCH\n'
    )
    if "_MT_PATCH" in src:
        print(f"[{model_key}] patch already present", flush=True)
    elif anchor in src:
        with open(script, "w") as f:
            f.write(src.replace(anchor, inject, 1))
        print(f"[{model_key}] patched client to count delta.reasoning tokens", flush=True)
    else:
        print(f"[{model_key}] WARNING: patch anchor not found", flush=True)

    # Baseline prefix-cache counters so we measure THIS run's hit rate.
    try:
        cache_before = _prefix_cache_counters(_get("/metrics"))
    except Exception as e:
        cache_before = {}
        print(f"[{model_key}] /metrics baseline failed: {e}", flush=True)

    cmd = [
        "python", "/bench/benchmark_serving_multi_turn.py",
        "--model", TOKENIZER,
        "--served-model-name", served_name,
        "--url", url,
        "--input-file", "/bench/config.json",
        "--num-clients", str(concurrency),
        "--max-active-conversations", str(concurrency),
        "--seed", str(seed),
        "--warmup-step",
        "--stats-json-output", "/bench/stats.json",
    ]
    print(f"[{model_key}] running: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, cwd="/bench", capture_output=True, text=True)
    wall = time.time() - t0

    # Full benchmark stdout/stderr (no slicing) so the perf summary survives.
    # Most of the script's metrics + HTTP warnings go through logger -> stderr,
    # so always print it (not just on failure).
    print(f"\n===== [{model_key}] BENCHMARK STDOUT =====", flush=True)
    print(res.stdout, flush=True)
    print(f"\n===== [{model_key}] BENCHMARK STDERR =====", flush=True)
    print(res.stderr, flush=True)

    try:
        cache_after = _prefix_cache_counters(_get("/metrics"))
    except Exception:
        cache_after = {}

    # Compute prefix-cache hit rate over the run from the counter deltas.
    hit_rate = None
    for qk, hk in (("vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total"),
                   ("vllm:gpu_prefix_cache_queries", "vllm:gpu_prefix_cache_hits")):
        dq = cache_after.get(qk, 0) - cache_before.get(qk, 0)
        dh = cache_after.get(hk, 0) - cache_before.get(hk, 0)
        if dq > 0:
            hit_rate = dh / dq
            break

    stats = None
    try:
        with open("/bench/stats.json") as f:
            stats = json.load(f)
    except Exception as e:
        print(f"[{model_key}] could not load stats.json: {e}", flush=True)

    print(f"\n===== [{model_key}] SERVER PREFIX-CACHE HIT RATE ====="
          f"\nbefore={cache_before}\nafter={cache_after}"
          f"\nhit_rate={hit_rate}", flush=True)

    return {
        "model": model_key, "concurrency": concurrency,
        "common_prefix": common_prefix, "wall_s": round(wall, 1),
        "returncode": res.returncode,
        "prefix_cache_hit_rate": hit_rate,
        "stats": stats,
    }


@app.local_entrypoint()
def main(model: str = "both", concurrency: int = 64,
         common_prefix: int = 1650,  # calibrated: ~48% prefix-cache hit rate
         num_conversations: int = 500, seed: int = 12345):
    import os
    os.makedirs("results/multiturn", exist_ok=True)
    keys = ["nvfp4", "bf16"] if model == "both" else [model]
    for k in keys:
        print(f"\n{'=' * 70}\n=== {k}  concurrency={concurrency}  "
              f"common_prefix={common_prefix}\n{'=' * 70}")
        r = run_bench.remote(k, concurrency, common_prefix, num_conversations, seed)
        out = f"results/multiturn/{k}_c{concurrency}_p{common_prefix}.json"
        with open(out, "w") as f:
            json.dump(r, f, indent=2)
        print(f"[{k}] wall={r['wall_s']}s  returncode={r['returncode']}  "
              f"prefix_cache_hit_rate={r['prefix_cache_hit_rate']}")
        print(f"[{k}] full result saved -> {out}")
        if r["stats"] and isinstance(r["stats"], dict):
            print(f"[{k}] stats.json top-level keys: {list(r['stats'].keys())}")
