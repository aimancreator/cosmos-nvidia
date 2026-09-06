from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import modal

from cosmos_studio.config import MODEL_ID


APP_NAME = "cosmos3-image-to-video-studio"
MODEL_VOLUME_NAME = "cosmos3-super-models"
MODEL_MOUNT = "/models"
MODEL_DIR = f"{MODEL_MOUNT}/Cosmos3-Super-Image2Video"
MODEL_COMPLETE_MARKER = f"{MODEL_DIR}/.download-complete"
GUARDRAIL_MODEL_ID = "nvidia/Cosmos-1.0-Guardrail"
HF_HUB_CACHE = f"{MODEL_MOUNT}/.huggingface/hub"
VLLM_PORT = 8000

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

# NVIDIA calls this the easiest all-in-one Cosmos 3 serving build. Modal adds a
# Python 3.11 runtime for our worker wrapper while preserving its CUDA stack.
gpu_image = (
    modal.Image.from_registry("vllm/vllm-omni:cosmos3", add_python="3.11")
    .entrypoint([])
    .env({"HF_HOME": f"{MODEL_MOUNT}/.huggingface"})
    .uv_pip_install(
        "huggingface_hub[hf_xet]>=0.34",
        "modal>=1.3,<2",
        "pillow>=11,<13",
        "requests>=2.32,<3",
    )
    .add_local_python_source("cosmos_studio")
)

download_image_base = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("huggingface_hub[hf_xet]>=0.34")
)
download_image = download_image_base.add_local_python_source("cosmos_studio")

# The guardrail repository currently contains metadata that hf-xet 1.5 cannot
# parse. Hugging Face supports disabling Xet so its regular download path can
# populate the same persistent Hub cache instead.
guardrail_download_image = download_image_base.env(
    {"HF_HUB_DISABLE_XET": "1"}
).add_local_python_source("cosmos_studio")


@app.function(
    image=download_image,
    volumes={MODEL_MOUNT: model_volume},
    secrets=[hf_secret],
    cpu=4,
    memory=16_384,
    timeout=60 * 60 * 4,
)
def download_model(force: bool = False) -> str:
    """Download the large checkpoint once into a persistent Modal Volume."""
    from huggingface_hub import snapshot_download

    marker = Path(MODEL_COMPLETE_MARKER)
    if marker.exists() and not force:
        return f"Model is already cached at {MODEL_DIR}"

    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_DIR,
        max_workers=8,
        token=os.environ.get("HF_TOKEN"),
    )
    marker.write_text(f"{MODEL_ID}\n", encoding="utf-8")
    model_volume.commit()
    return f"Downloaded {MODEL_ID} to Modal Volume {MODEL_VOLUME_NAME}"


@app.function(
    image=guardrail_download_image,
    volumes={MODEL_MOUNT: model_volume},
    secrets=[hf_secret],
    cpu=4,
    memory=16_384,
    timeout=60 * 60 * 2,
)
def download_guardrail() -> str:
    """Cache the gated Cosmos guardrail checkpoint for GPU cold starts."""
    from huggingface_hub import snapshot_download

    checkpoint_dir = snapshot_download(
        repo_id=GUARDRAIL_MODEL_ID,
        cache_dir=HF_HUB_CACHE,
        max_workers=8,
        token=os.environ.get("HF_TOKEN"),
    )
    model_volume.commit()
    return f"Downloaded {GUARDRAIL_MODEL_ID} to {checkpoint_dir}"


@app.function(
    image=download_image,
    volumes={MODEL_MOUNT: model_volume},
    secrets=[hf_secret],
    cpu=4,
    memory=16_384,
    timeout=60 * 60 * 2,
)
def diagnose_guardrail_download() -> str:
    """Download guardrail files serially and identify the first Xet failure."""
    from huggingface_hub import HfApi, hf_hub_download

    token = os.environ.get("HF_TOKEN")
    filenames = HfApi().list_repo_files(GUARDRAIL_MODEL_ID, token=token)
    total = len(filenames)
    for index, filename in enumerate(filenames, start=1):
        print(f"DIAGNOSTIC_FILE {index}/{total}: {filename}", flush=True)
        try:
            hf_hub_download(
                repo_id=GUARDRAIL_MODEL_ID,
                filename=filename,
                cache_dir=HF_HUB_CACHE,
                token=token,
            )
        except Exception:
            print(f"DIAGNOSTIC_FAILED_FILE: {filename}", flush=True)
            raise

    return f"All {total} files downloaded without an Xet failure"


@app.cls(
    image=gpu_image,
    gpu="H100:8",
    cpu=16,
    memory=131_072,
    volumes={MODEL_MOUNT: model_volume},
    secrets=[hf_secret],
    timeout=15 * 60,
    startup_timeout=15 * 60,
    retries=0,
    # Release the eight-GPU container almost immediately after each call. The
    # serving platform enforces two seconds as its minimum idle window.
    scaledown_window=2,
    max_containers=1,
)
@modal.concurrent(max_inputs=1)
class CosmosWorker:
    def _terminate_server(self) -> None:
        server = getattr(self, "server", None)
        if server is None or server.poll() is not None:
            return
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)

    @modal.enter()
    def start_server(self) -> None:
        import requests

        marker = Path(MODEL_COMPLETE_MARKER)
        if not marker.exists():
            raise RuntimeError(
                "Model is not downloaded yet. Run: modal run modal_app.py::download_model"
            )

        command = [
            "vllm",
            "serve",
            MODEL_DIR,
            "--omni",
            "--host",
            "127.0.0.1",
            "--port",
            str(VLLM_PORT),
            "--cfg-parallel-size",
            "2",
            "--ulysses-degree",
            "4",
            "--use-hsdp",
            "--hsdp-shard-size",
            "8",
            "--allowed-local-media-path",
            "/tmp/cosmos-inputs",
            # This image's vLLM-Omni source supports the Cosmos3 server gate
            # but its CLI accidentally omits the documented shortcut flag.
            # Pass the underlying OmniDiffusionConfig value to stage 0.
            "--stage-overrides",
            '{"0":{"model_config":{"guardrails":false}}}',
            "--init-timeout",
            "1800",
        ]
        self.server = subprocess.Popen(command)

        deadline = time.monotonic() + 50 * 60
        health_url = f"http://127.0.0.1:{VLLM_PORT}/v1/models"
        while time.monotonic() < deadline:
            if self.server.poll() is not None:
                raise RuntimeError(
                    f"vLLM-Omni exited during startup with code {self.server.returncode}."
                )
            try:
                response = requests.get(health_url, timeout=5)
                if response.ok:
                    # Persist guardrail downloads from HF_HOME so later cold
                    # starts do not fetch the gated checkpoint again.
                    model_volume.commit()
                    return
            except requests.RequestException:
                pass
            time.sleep(5)
        self.server.terminate()
        raise TimeoutError("vLLM-Omni did not become ready within 50 minutes.")

    @modal.exit()
    def stop_server(self) -> None:
        self._terminate_server()

    @modal.method()
    def generate(self, image_bytes: bytes, filename: str, request_data: dict) -> bytes:
        import requests
        from PIL import Image

        # Each expensive GPU container is one-shot. Once this input begins, do
        # not fetch another input; exit after success, failure, or cancellation.
        modal.experimental.stop_fetching_inputs()

        try:
            input_dir = Path("/tmp/cosmos-inputs")
            input_dir.mkdir(parents=True, exist_ok=True)
            input_path = input_dir / "input.png"

            # Decode and normalize here as a final server-side safety check. RGB
            # is required, and PNG avoids relying on user filenames.
            from io import BytesIO

            with Image.open(BytesIO(image_bytes)) as source:
                source.convert("RGB").save(input_path, format="PNG")

            with input_path.open("rb") as image_file:
                response = requests.post(
                    f"http://127.0.0.1:{VLLM_PORT}/v1/videos/sync",
                    data=request_data,
                    files={
                        "input_reference": (
                            filename or "input.png",
                            image_file,
                            "image/png",
                        )
                    },
                    headers={"Accept": "video/mp4"},
                    timeout=14 * 60,
                )

            if not response.ok:
                detail = response.text[:2_000]
                raise RuntimeError(
                    f"Cosmos generation failed ({response.status_code}): {detail}"
                )
            return response.content
        except BaseException:
            # Modal cancellation interrupts the wrapper, but vLLM may otherwise
            # keep denoising the already-submitted HTTP request. Kill it here.
            self._terminate_server()
            raise
