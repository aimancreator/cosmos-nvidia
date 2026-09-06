# Cosmos 3 Motion Studio — Local Flask UI + Modal GPU

A local Flask browser UI for `nvidia/Cosmos3-Super-Image2Video`. You open the studio on `localhost`, upload one image and a motion prompt, Flask submits an asynchronous job to Modal, and the finished MP4 comes back to localhost for preview and download.

I'm using [Modal.com](https://modal.com) for GPU compute to run the NVIDIA Cosmos model. The Flask UI runs locally, while video generation runs on Modal's cloud GPUs.

## Start the local studio

Double-click `start_local.command`, or run:

```bash
source .venv/bin/activate
python flask_app.py
```

Then open:

```text
http://127.0.0.1:7860
```

The UI itself runs only on your Mac. Clicking **Generate video** sends the image and prompt to the deployed Modal GPU worker, then the page polls Modal and shows the returned video locally.

## What this project creates

- A Python 3.11 local environment managed by `uv`
- A persistent Modal Volume containing the model weights
- NVIDIA's official `vllm/vllm-omni:cosmos3` serving image
- The NVIDIA-recommended 8-GPU parallelism settings for the Super model
- A local Flask server with a responsive upload, prompt, progress, cancel, preview, and download UI
- Scale-to-zero GPU infrastructure with one generation at a time

## One-time setup

```bash
source .venv/bin/activate
modal setup
```

`modal setup` opens a browser so you can connect this terminal to your Modal account.

The checkpoint is approximately 65B BF16 parameters and is very large. Download it directly into the persistent Modal Volume:

```bash
modal run modal_app.py::download_model
```

Then deploy only the GPU worker:

```bash
modal deploy modal_app.py
```

There is no hosted web UI. The browser interface is available only from `http://127.0.0.1:7860`. Its Generation Log records submission, Modal acceptance, processing, success, download, cancellation, and errors. Flask writes matching job results to the terminal.

## Development and checks

```bash
uv sync
uv run pytest
python flask_app.py
```

## Important cost and model notes

- NVIDIA recommends 8×H100, 8×H200, or 8×A100 for this 64B Image2Video checkpoint. An H100 worker is expensive even for a short run. Set a Modal workspace budget before testing.
- The GPU container scales to zero approximately two seconds after a generation call finishes, minimizing idle GPU charges.
- The first startup loads the large checkpoint and may take many minutes; generation itself also takes several minutes.
- This app keeps Cosmos guardrails disabled because NVIDIA's separate Cosmos guardrail model is gated. Only submit content you have the right to use and review generated output responsibly.
- Modal retains Function inputs and outputs temporarily according to its platform data-retention policy. Do not upload sensitive imagery unless that policy fits your use case.

## Useful operations

```bash
# See deployed apps
modal app list

# View logs
modal app logs cosmos3-image-to-video-studio

# Stop the deployment
modal app stop cosmos3-image-to-video-studio

# Check or re-run the model download
modal run modal_app.py::download_model

# Force a clean checkpoint download if the cache is incomplete
modal run modal_app.py::download_model --force
```
