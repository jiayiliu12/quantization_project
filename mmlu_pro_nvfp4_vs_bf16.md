# MMLU Pro — Qwen3.6-35B-A3B: NVFP4 vs BF16

**Date:** 2026-07-17
**Task:** `mmlu_pro` (lm-eval-harness 0.4.12), 5-shot CoT, custom-extract
**Sample:** `--limit 50` → 50 questions × 14 subjects = 700 questions per model

## Result

| Metric        | NVFP4 (ours) | BF16 (ours) | NVIDIA NVFP4 | NVIDIA BF16 |
|---------------|-------------:|------------:|-------------:|------------:|
| MMLU Pro      | **84.86%**   | **84.29%**  | 85.0         | 85.6        |
| Std error     | ±1.32        | ±1.32       | —            | —           |

**Conclusion:** Quantization is effectively lossless on MMLU Pro. NVFP4 scored +0.57
points above BF16 — within one standard error, so the two are statistically
indistinguishable. Absolute scores land within ~1 point of NVIDIA's published
numbers (expected, given 700 questions vs their full ~12k set).

Caveats:
- At 700 questions you can conclude "no measurable degradation," not that NVFP4
  ≥ BF16. For a decimal-exact reproduction, drop `--limit` and run the full set.
- Per-subject scores (50 Qs each) swing a lot; only the aggregate is reliable.

### Per-subject (exact_match)

| Subject          | NVFP4 | BF16 |
|------------------|------:|-----:|
| biology          | 0.96  | 0.98 |
| business         | 0.88  | 0.88 |
| chemistry        | 0.92  | 0.92 |
| computer_science | 0.88  | 0.86 |
| economics        | 0.94  | 0.92 |
| engineering      | 0.86  | 0.90 |
| health           | 0.78  | 0.86 |
| history          | 0.66  | 0.66 |
| law              | 0.72  | 0.68 |
| math             | 0.96  | 0.96 |
| other            | 0.72  | 0.68 |
| philosophy       | 0.82  | 0.76 |
| physics          | 0.98  | 0.98 |
| psychology       | 0.80  | 0.76 |

## Deployment config (`qwen36_compare_modal.py`)

| Setting                    | Value                                                                 |
|----------------------------|-----------------------------------------------------------------------|
| Image                      | `vllm/vllm-openai@sha256:397273c7a694aeccd3b081d3fb07f4c2983958f22cdfec9c7cde3d70b828224d` (nightly, pinned) |
| GPU                        | 1× B200 per model                                                     |
| `--max-model-len`          | 131072                                                                 |
| `--gpu-memory-utilization` | 0.90                                                                   |
| `--reasoning-parser`       | qwen3 (on)                                                             |
| `--quantization`           | `modelopt` (NVFP4 only; BF16 has no quant flag)                        |

vLLM build serving both: `vllm-0.23.1rc1.dev1133+g647213129`. Verified live:
both endpoints report `max_model_len: 131072`, KV cache 89.7 GiB.

## lm-eval command (per model)

```bash
uv run lm_eval --model local-chat-completions --tasks mmlu_pro \
  --model_args "model=<MODEL>,base_url=<URL>/v1/chat/completions,num_concurrent=32,tokenized_requests=False,tokenizer_backend=None,timeout=1200,max_length=131072,max_retries=3" \
  --apply_chat_template \
  --gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,max_gen_toks=128000,until=None" \
  --output_path <OUT> --log_samples --limit 50
```

| Model | `<MODEL>`                        | `<URL>`                                              | `<OUT>`                  |
|-------|----------------------------------|------------------------------------------------------|--------------------------|
| NVFP4 | `nvidia/Qwen3.6-35B-A3B-NVFP4`   | `https://sidhq--qwen36-compare-serve-nvfp4.modal.run` | `results/nvfp4_mmlu_pro` |
| BF16  | `Qwen/Qwen3.6-35B-A3B`           | `https://sidhq--qwen36-compare-serve-bf16.modal.run`  | `results/bf16_mmlu_pro`  |

## Three non-obvious settings that mattered

1. **`until=None`** — overrides mmlu_pro's built-in `until: ["Question:"]` stop.
   The reasoning model restates "Question:" while thinking, tripping that stop
   mid-`<think>`, leaving `content` empty → all scored `[invalid]`. This was the
   4% → 85% fix.
2. **`max_gen_toks=128000`** (not 131072) — this is the *output* budget and must
   stay below `max_model_len` (131072) to leave room for the ~1,800-token 5-shot
   prompt. Setting it equal to the context window makes vLLM reject every request
   with HTTP 400.
3. **`temperature=1.0`, `top_p=0.95`** (no `top_k`/`min_p`) — matches NVIDIA's
   published MMLU-Pro methodology. (Their SciCode run instead uses temp 0.6.)

## Methodology vs NVIDIA's table

- NVIDIA ran the full ~12k-question test set; we ran 700 (`--limit 50`).
- NVIDIA used B300; we used B200. Irrelevant to scores, only speed.
- NVIDIA's "max num tokens 131072" = model context length; generation is
  effectively unbounded for MMLU (no answer approaches 128k tokens of thinking).
- NVIDIA sampling: MMLU Pro / GPQA / others at temp 1.0, top_p 0.95, 131072;
  SciCode at temp 0.6, top_p 0.95, 131072.
