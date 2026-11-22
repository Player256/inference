import json
from typing import Any
import aiohttp
import modal

vllm_image = (modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
              .entrypoint([])
              .uv_pip_install("vllm==0.21.0", "transformers==5.8.1")
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
MODEL_REVISION = "bbaaae1f57d79c7bf696d619ffbbb0c2c8e5ef98" 

hf_cache_vol = modal.Volume.from_name("huggingface-cache")

vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

FAST_BOOT = True

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
            MODEL_NAME,
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
            "--trust-remote-code"
        ]

        cmd += [
            "--enforce-eager" if FAST_BOOT else "--no-enforce-eager"
        ]

        cmd += [
            "--tensor-parallel-size",  str(N_GPU)
        ]

        cmd += [
            "--limit-mm-per-prompt" , json.dumps({"image" : 0, "video": 0, "audio": 0}),
            "--enable-auto-tool-choice",
            "--reasoning-parser",
            "qwen3",
            "--tool-call-parser",
            "qwen3_coder"
        ]

        print(*cmd)

        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self):
        self.process.terminate()