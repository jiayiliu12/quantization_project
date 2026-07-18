"""
Serve nvidia/Qwen3.6-35B-A3B-NVFP4 and the BF16 baseline Qwen/Qwen3.6-35B-A3B
on Modal, each on a single Blackwell B200, as OpenAI-compatible endpoints.

Deploy:      modal deploy qwen36_compare_modal.py
Warm up:     modal run qwen36_compare_modal.py            # optional: pre-downloads weights
URLs (after deploy):
    NVFP4 ->  https://<workspace>--qwen36-compare-serve-nvfp4.modal.run
    BF16  ->  https://<workspace>--qwen36-compare-serve-bf16.modal.run
Both speak the OpenAI API at  <url>/v1  (try <url>/docs in a browser).
"""

import subprocess

import modal

# --- config -----------------------------------------------------------------

NVFP4_MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"
BF16_MODEL = "Qwen/Qwen3.6-35B-A3B"

# Both fit on one B200 (192 GB): BF16 weights ~70 GB, NVFP4 ~20 GB.
# Keep this identical across both servers so the comparison is fair.
# You can push toward 262144 (the model's max); if BF16 OOMs on KV cache, lower it.
MAX_LEN = 131072

VLLM_PORT = 8000
MINUTES = 60
GPU = "B200"  # or "B200+" to also accept B300, or "B200:2" for tensor parallelism
USE_REASONING_PARSER = True
# --- image & caches ---------------------------------------------------------

# NVIDIA recommends the vLLM nightly image for this NVFP4 checkpoint. It ships
# the FlashInfer / CUTLASS FP4 kernels needed for native Blackwell acceleration.
# Pinned by digest (nightly of 2026-07-17) so both servers — and any future
# redeploy — run the exact same vLLM build; the inference stack is part of the
# A/B comparison. To upgrade, look up the new digest of vllm/vllm-openai:nightly.
VLLM_IMAGE = "vllm/vllm-openai@sha256:397273c7a694aeccd3b081d3fb07f4c2983958f22cdfec9c7cde3d70b828224d"

vllm_image = (
    # add_python gives Modal a standalone interpreter to run our functions with.
    # Without it, Modal can't detect Python in this registry image and the build
    # fails with "unable to determine the version of Python".
    modal.Image.from_registry(VLLM_IMAGE, add_python="3.12")
    .entrypoint([])  # drop the image's built-in `vllm serve` entrypoint
    # download() runs under the add_python interpreter, so it needs its own copy
    # of huggingface_hub. (vllm serve still runs under the image's own Python.)
    .pip_install("huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("qwen36-compare")

VOLUMES = {
    "/root/.cache/huggingface": hf_cache,
    "/root/.cache/vllm": vllm_cache,
}


def _serve(model: str, *quant_args: str) -> None:
    """Launch a vLLM OpenAI server as a subprocess and return."""
    cmd = [
        "vllm", "serve", model,
        "--served-model-name", model,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--max-model-len", str(MAX_LEN),
        "--trust-remote-code",
        "--gpu-memory-utilization", "0.90",
        *quant_args,
    ]
    if USE_REASONING_PARSER:
        cmd += ["--reasoning-parser", "qwen3"]
    subprocess.Popen(cmd)


# --- NVFP4 server -----------------------------------------------------------

@app.function(
    image=vllm_image,
    gpu=GPU,
    volumes=VOLUMES,
    scaledown_window=15 * MINUTES,   # stay warm this long after the last request
    timeout=30 * MINUTES,
)
@modal.concurrent(max_inputs=32)     # requests one replica handles before scaling out
@modal.web_server(port=VLLM_PORT, startup_timeout=30 * MINUTES)
def serve_nvfp4():
    # --quantization modelopt tells vLLM to read the ModelOpt NVFP4 checkpoint
    _serve(NVFP4_MODEL, "--quantization", "modelopt")


# --- BF16 baseline server ---------------------------------------------------

@app.function(
    image=vllm_image,
    gpu=GPU,
    volumes=VOLUMES,
    scaledown_window=15 * MINUTES,
    timeout=30 * MINUTES,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=30 * MINUTES)
def serve_bf16():
    _serve(BF16_MODEL)  # no quantization flag -> plain BF16


# --- optional: pre-download weights into the volume -------------------------

@app.function(
    image=vllm_image,
    volumes=VOLUMES,
    timeout=60 * MINUTES,
)
def download():
    from huggingface_hub import snapshot_download

    for repo in (NVFP4_MODEL, BF16_MODEL):
        print(f"downloading {repo} ...")
        snapshot_download(repo, ignore_patterns=["*.pt", "*.bin"])
    hf_cache.commit()


@app.local_entrypoint()
def main():
    # `modal run` triggers this: warm the cache so first real request is fast.
    download.remote()