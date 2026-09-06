from __future__ import annotations

import subprocess
import time
from io import BytesIO
from pathlib import Path

import modal

from cosmos_studio.config import MODEL_ID, build_generation_request


MODEL_VOLUME_NAME = "cosmos3-super-models"
MODEL_MOUNT = "/models"
MODEL_DIR = f"{MODEL_MOUNT}/Cosmos3-Super-Image2Video"
MODEL_COMPLETE_MARKER = f"{MODEL_DIR}/.download-complete"
VLLM_PORT = 8000

app = modal.App("cosmos3-single-h100-test")
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

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


@app.cls(
    image=gpu_image,
    gpu="H100:1",
    cpu=16,
    memory=131_072,
    volumes={MODEL_MOUNT: model_volume},
    secrets=[hf_secret],
    timeout=20 * 60,
    startup_timeout=15 * 60,
    scaledown_window=60,
    max_containers=1,
    retries=0,
)
@modal.concurrent(max_inputs=1)
class CosmosSingleH100Test:
    def start_server(self) -> None:
        import requests

        if not Path(MODEL_COMPLETE_MARKER).exists():
            raise RuntimeError("The model completion marker is missing.")

        command = [
            "vllm",
            "serve",
            MODEL_DIR,
            "--omni",
            "--host",
            "127.0.0.1",
            "--port",
            str(VLLM_PORT),
            "--allowed-local-media-path",
            "/tmp/cosmos-inputs",
            "--stage-overrides",
            '{"0":{"model_config":{"guardrails":false}}}',
            "--init-timeout",
            "600",
        ]
        self.server = subprocess.Popen(command)

        deadline = time.monotonic() + 12 * 60
        health_url = f"http://127.0.0.1:{VLLM_PORT}/v1/models"
        while time.monotonic() < deadline:
            if self.server.poll() is not None:
                raise RuntimeError(
                    f"Single-H100 vLLM-Omni exited during startup with code "
                    f"{self.server.returncode}."
                )
            try:
                response = requests.get(health_url, timeout=5)
                if response.ok:
                    return
            except requests.RequestException:
                pass
            time.sleep(5)

        self.server.terminate()
        raise TimeoutError("Single-H100 vLLM-Omni was not ready within 12 minutes.")

    @modal.exit()
    def stop_server(self) -> None:
        if getattr(self, "server", None) and self.server.poll() is None:
            self.server.terminate()

    @modal.method()
    def generate(self, image_bytes: bytes, filename: str, request_data: dict) -> bytes:
        import requests
        from PIL import Image

        self.start_server()

        input_dir = Path("/tmp/cosmos-inputs")
        input_dir.mkdir(parents=True, exist_ok=True)
        input_path = input_dir / "input.png"
        with Image.open(BytesIO(image_bytes)) as source:
            source.convert("RGB").save(input_path, format="PNG")

        with input_path.open("rb") as image_file:
            response = requests.post(
                f"http://127.0.0.1:{VLLM_PORT}/v1/videos/sync",
                data=request_data,
                files={"input_reference": (filename, image_file, "image/png")},
                headers={"Accept": "video/mp4"},
                timeout=15 * 60,
            )
        if not response.ok:
            raise RuntimeError(
                f"Cosmos generation failed ({response.status_code}): "
                f"{response.text[:2_000]}"
            )
        return response.content


@app.local_entrypoint()
def main() -> None:
    input_path = Path("images.jpeg")
    request_data = build_generation_request(
        prompt=(
            "The humanoid robot slowly turns its head toward the camera, blinks "
            "naturally, then lowers its hand slightly while subtle mechanical joints "
            "move smoothly. The camera remains steady and the white studio background "
            "stays unchanged."
        ),
        preset_name="draft",
        size_name="portrait",
        seed=42,
    )
    video = CosmosSingleH100Test().generate.remote(
        input_path.read_bytes(), input_path.name, request_data
    )
    if not isinstance(video, bytes) or b"ftyp" not in video[:64]:
        raise RuntimeError("Modal returned data that is not a valid-looking MP4 file.")
    output = Path("cosmos3-robot-1xh100.mp4")
    output.write_bytes(video)
    print(f"Saved {output.resolve()} ({len(video)} bytes)")
