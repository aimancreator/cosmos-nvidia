from __future__ import annotations

import os
import json
import logging
import re
import select
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import modal
from flask import (
    Flask,
    Response,
    jsonify,
    request,
    send_from_directory,
    stream_with_context,
)
from PIL import Image

from cosmos_studio.config import MODEL_ID, build_generation_request


MODAL_APP_NAME = "cosmos3-image-to-video-studio"
MODAL_CLASS_NAME = "CosmosWorker"
ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
FUNCTION_CALL_ID = re.compile(r"^fc-[A-Za-z0-9]+$")
ANSI_ESCAPE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES + 1024 * 1024
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("cosmos-studio")

_worker = None


def clean_remote_log_line(line: str) -> str:
    """Make a remote worker line safe and readable in the public demo UI."""
    line = ANSI_ESCAPE.sub("", line).replace("\r", "").strip()
    if not line or line.startswith("Following logs for "):
        return ""

    # Keep the localhost demo provider-neutral without discarding useful
    # engine, loading, denoising, decoding, or error details.
    line = re.sub(r"(?i)modal(?:\.com)?", "render service", line)
    return line[:1_000]


def remote_log_level(line: str) -> str:
    lowered = line.lower()
    if any(word in lowered for word in ("error", "failed", "exception", "traceback", "out of memory")):
        return "error"
    if "warning" in lowered or "warn" in lowered:
        return "warning"
    if any(word in lowered for word in ("complete", "completed", "generated", "decoded", "finished")):
        return "success"
    return "info"


def stop_log_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=1)


def user_friendly_generation_error(exc: Exception) -> str:
    """Keep infrastructure details in the terminal and out of the public UI."""
    detail = str(exc)
    lowered = detail.lower()
    if "gatedrepoerror" in lowered or "cannot access gated repo" in lowered:
        return (
            "The motion engine could not load a required model component. "
            "Check the local terminal for details."
        )
    if "vllm-omni exited during startup" in lowered:
        return "The motion engine failed while loading. Check the local terminal for details."
    if "out of memory" in lowered or "cuda oom" in lowered:
        return "The motion engine ran out of memory while generating the video."
    if "notfounderror" in lowered or "app" in lowered and "not found" in lowered:
        return "The rendering service is currently offline."
    return "The render could not be completed. Check the local terminal for technical details."


def get_modal_worker():
    """Connect to the already-deployed Modal class using local credentials."""
    global _worker
    if _worker is None:
        worker_class = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)
        _worker = worker_class()
    return _worker


@app.get("/")
def home():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/api/health")
def health():
    return jsonify(
        {
            "ok": True,
            "mode": "local-flask",
            "model": MODEL_ID,
            "engine": "cosmos3-image-to-video",
        }
    )


@app.post("/api/generate")
def generate():
    uploaded = request.files.get("image")
    if uploaded is None:
        return jsonify(detail="Choose a starting image first."), 400
    if uploaded.mimetype not in {"image/jpeg", "image/png", "image/webp"}:
        return jsonify(detail="Please upload a JPG, PNG, or WebP image."), 400

    image_bytes = uploaded.read()
    if not image_bytes:
        return jsonify(detail="The uploaded image is empty."), 400
    if len(image_bytes) > MAX_IMAGE_BYTES:
        return jsonify(detail="Image must be 20 MB or smaller."), 400

    try:
        with Image.open(BytesIO(image_bytes)) as decoded:
            decoded.verify()
        request_data = build_generation_request(
            prompt=request.form.get("prompt", ""),
            negative_prompt=request.form.get("negative_prompt", ""),
            preset_name=request.form.get("preset", "balanced"),
            size_name=request.form.get("aspect", "landscape"),
            seed=int(request.form.get("seed", "42")),
        )
        logger.info(
            "Submitting generation | file=%s | preset=%s | aspect=%s",
            uploaded.filename or "input.png",
            request.form.get("preset", "balanced"),
            request.form.get("aspect", "landscape"),
        )
        call = get_modal_worker().generate.spawn(
            image_bytes,
            uploaded.filename or "input.png",
            request_data,
        )
    except (ValueError, OSError) as exc:
        logger.warning("Generation validation error | %s", exc)
        return jsonify(detail=str(exc)), 400
    except Exception as exc:
        logger.exception("Generation submission failed")
        return jsonify(detail="Could not start the render. Check the local terminal for details."), 502

    logger.info("Generation queued successfully | job_id=%s", call.object_id)
    return jsonify(job_id=call.object_id, status="queued"), 202


@app.get("/api/jobs/<job_id>")
def job_result(job_id: str):
    try:
        call = modal.FunctionCall.from_id(job_id)
        video_bytes = call.get(timeout=0)
    except TimeoutError:
        return (
            jsonify(
                status="processing",
                message="The motion engine is loading or generating your video.",
            ),
            202,
        )
    except Exception as exc:
        logger.exception("Generation failed | job_id=%s", job_id)
        try:
            call.cancel(terminate_containers=True)
        except Exception:
            logger.exception(
                "Could not terminate failed generation container | job_id=%s", job_id
            )
        return jsonify(status="failed", message=user_friendly_generation_error(exc)), 500

    logger.info(
        "Video generated successfully | job_id=%s | bytes=%s", job_id, len(video_bytes)
    )
    return Response(
        video_bytes,
        mimetype="video/mp4",
        headers={
            "Content-Disposition": f'attachment; filename="cosmos3-{job_id[-8:]}.mp4"',
            "Cache-Control": "private, max-age=86400",
        },
    )


@app.get("/api/jobs/<job_id>/events")
def job_events(job_id: str):
    """Stream provider logs for one function call to the local browser."""
    if not FUNCTION_CALL_ID.fullmatch(job_id):
        return jsonify(detail="Invalid generation identifier."), 400

    environment = os.environ.copy()
    environment.update(PYTHONUNBUFFERED="1", NO_COLOR="1", TERM="dumb")
    command = [
        sys.executable,
        "-m",
        "modal",
        "app",
        "logs",
        MODAL_APP_NAME,
        "--follow",
        "--function-call",
        job_id,
    ]

    def stream():
        process = None
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=environment,
            )
            assert process.stdout is not None
            yield ": connected\n\n"

            while process.poll() is None:
                readable, _, _ = select.select([process.stdout], [], [], 5)
                if not readable:
                    yield ": keepalive\n\n"
                    continue

                line = process.stdout.readline()
                if not line:
                    continue
                message = clean_remote_log_line(line)
                if not message:
                    continue
                payload = json.dumps(
                    {"level": remote_log_level(message), "message": message}
                )
                yield f"data: {payload}\n\n"

            # Flush any final buffered lines if the log command exits.
            for line in process.stdout:
                message = clean_remote_log_line(line)
                if message:
                    payload = json.dumps(
                        {"level": remote_log_level(message), "message": message}
                    )
                    yield f"data: {payload}\n\n"
        except GeneratorExit:
            return
        except Exception:
            logger.exception("Remote log stream failed | job_id=%s", job_id)
            payload = json.dumps(
                {
                    "level": "warning",
                    "message": "The live worker log became unavailable; generation status is still monitored.",
                }
            )
            yield f"data: {payload}\n\n"
        finally:
            if process is not None:
                stop_log_process(process)

    return Response(
        stream_with_context(stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store",
            "X-Accel-Buffering": "no",
        },
    )


@app.delete("/api/jobs/<job_id>")
def cancel_job(job_id: str):
    try:
        # A soft cancel can leave the blocking vLLM HTTP request running. Force
        # termination so the eight-GPU container cannot continue in the background.
        modal.FunctionCall.from_id(job_id).cancel(terminate_containers=True)
    except Exception as exc:
        logger.exception("Could not cancel generation | job_id=%s", job_id)
        return jsonify(detail="Could not cancel this generation."), 400
    logger.info("Modal job cancelled | job_id=%s", job_id)
    return jsonify(status="cancelled")


@app.errorhandler(413)
def image_too_large(_error):
    return jsonify(detail="Image must be 20 MB or smaller."), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    print(f"\nCosmos Motion Studio: http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
