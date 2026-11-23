# In this deployment iteration, the focus would be again using a single gpu but with different optimization techniques. Going to use this dataset:https://huggingface.co/datasets/DiscoPosse/lmcache-agentic-traces. 
# config would be same model, same gpus but different optimization techniques. note all the metrics with different combinations and post the best optimization.

# config_1: no quant(bf_16), flash_attn, prefix_caching, model: qwen3.5:4B, gpu: A10:1
# config_2: AWQ INT4 weights, INT8 KV cache, Triton attention, prefix caching, A10:1
# config_3: no quant(bf_16), flash_attn, prefix_caching, model: qwen3.5:4B, gpu: A10:1, dflash-speculative decoding, max-model-len 80000(after reading the runs.)
# config_4 (active): AWQ INT4 target weights, INT8 KV cache, Triton attention, prefix caching, DFlash, A10:1

import json
from typing import Any
import aiohttp
import modal

vllm_image = (modal.Image.from_registry("nvidia/cuda:13.0.2-devel-ubuntu22.04", add_python="3.12")
              .entrypoint([])
              .uv_pip_install(
                  "vllm==0.34.0",
                  "transformers==5.22.2",
                  extra_options="--torch-backend=cu130",
              )
              .env(
                  {
                      "HF_XET_HIGH_PERFORMANCE": "1",
                      "VLLM_LOG_STATS_INTERVAL": "1",
                      "VLLM_USE_RUST_FRONTEND": "1",
                      "VLLM_FORCE_AOT_LOAD": "1"
                  }
              )
)
MODEL_NAME = "Qwen/Qwen3.5-4B"
MODEL_CHECKPOINT = "cyankiwi/Qwen3.5-4B-AWQ-BF16-INT4"
MODEL_REVISION = "182bf9f57531614c67150b06f3141524ce8b52a1"

# MODEL_CHECKPOINT = "Qwen/Qwen3.5-4B"
# MODEL_REVISION = "bbaaae1f57d79c7bf696d619ffbbb0c2c8e5ef98"

hf_cache_vol = modal.Volume.from_name("huggingface-cache")

vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

FAST_BOOT = False

app = modal.App("example-vllm-inference")

N_GPU = 1
MINUTES = 60
VLLM_PORT = 8000
ROUTING_REGION = "us-east"

@app.server(
    image=vllm_image,
    gpu=f"A10:{N_GPU}",
    scaledown_window=15*MINUTES,
    startup_timeout=10*MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol
    },
    port=VLLM_PORT,
    routing_region=ROUTING_REGION,
    target_concurrency=100,
    unauthenticated=True
)
class Server:
    @modal.enter()
    def start(self):
        import subprocess
        cmd = [
            "vllm",
            "serve",
            MODEL_CHECKPOINT,
            "--revision",
            MODEL_REVISION,
            "--served-model-name",
            MODEL_NAME,
            "llm",
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--uvicorn-log-level=info",
            "--async-scheduling", 
            "--trust-remote-code",
            "--enforce-eager" if FAST_BOOT else "--no-enforce-eager",
            "--tensor-parallel-size",  str(N_GPU),
            "--limit-mm-per-prompt" , json.dumps({"image" : 0, "video": 0, "audio": 0}),
            "--enable-auto-tool-choice",
            "--reasoning-parser",
            "qwen3",
            "--tool-call-parser",
            "qwen3_coder",
            "--enable-prefix-caching",
            "--attention-backend", "TRITON_ATTN",
            "--kv-cache-dtype", "int8_per_token_head",
            "--max-model-len", "80000",
            "--speculative-config", json.dumps({
                                        "method": "dflash",
                                        "model": "z-lab/Qwen3.5-4B-DFlash",
                                        "num_speculative_tokens": 4,
                                    }),
            "--max-num-batched-tokens", "2048"
        ]
        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self):
        self.process.terminate()
