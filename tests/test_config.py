import pytest

import json
from pathlib import Path

from cosmos_studio.config import SIZES, build_generation_request


def test_builds_cosmos_json_prompt():
    request = build_generation_request(
        prompt="The robot slowly lifts the blue box while the camera remains still.",
        preset_name="quality",
        size_name="portrait",
        seed=7,
    )

    prompt = json.loads(request["prompt"])
    assert prompt["resolution"] == {"H": 832, "W": 480}
    assert prompt["aspect_ratio"] == "9,16"
    assert request["size"] == "480x832"
    assert request["num_frames"] == 189
    assert request["num_inference_steps"] == 50
    assert json.loads(request["extra_params"])["guardrails"] is False
    assert request["seed"] == 7


def test_rejects_short_prompt():
    with pytest.raises(ValueError, match="more detail"):
        build_generation_request(prompt="move")


def test_rejects_invalid_options():
    with pytest.raises(ValueError, match="quality"):
        build_generation_request(prompt="A detailed motion prompt", preset_name="turbo")


@pytest.mark.parametrize(
    ("preset", "expected_frames"),
    [("draft", 85), ("balanced", 125), ("quality", 189)],
)
@pytest.mark.parametrize("size", ["landscape", "portrait"])
def test_frames_are_compatible_with_four_way_ulysses(preset, expected_frames, size):
    request = build_generation_request(
        prompt="The robot moves smoothly while the camera remains completely steady.",
        preset_name=preset,
        size_name=size,
    )

    width, height, _ = SIZES[size]
    temporal_tokens = (request["num_frames"] - 1) // 4 + 1
    spatial_tokens = ((height + 31) // 32) * ((width + 31) // 32)
    assert request["num_frames"] == expected_frames
    assert (temporal_tokens * spatial_tokens) % 4 == 0


def test_square_draft_uses_nearest_compatible_frame_count():
    request = build_generation_request(
        prompt="The robot moves smoothly while the camera remains completely steady.",
        preset_name="draft",
        size_name="square",
    )
    assert request["num_frames"] == 93


def test_local_flask_health_and_home():
    from flask_app import app

    client = app.test_client()
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json["mode"] == "local-flask"
    assert health.json["model"] == "nvidia/Cosmos3-Super-Image2Video"
    assert "modal" not in json.dumps(health.json).lower()

    home = client.get("/")
    assert home.status_code == 200
    assert b"modal" not in home.data.lower()


def test_public_web_assets_do_not_name_infrastructure_provider():
    web_dir = Path(__file__).parents[1] / "web"
    for filename in ("index.html", "app.js", "styles.css"):
        assert "modal" not in (web_dir / filename).read_text(encoding="utf-8").lower()


def test_generation_errors_do_not_leak_infrastructure_details():
    from flask_app import user_friendly_generation_error

    message = user_friendly_generation_error(
        RuntimeError("Modal NotFoundError: app cosmos3-image-to-video-studio not found")
    )
    assert "modal" not in message.lower()
    assert "cosmos3-image-to-video-studio" not in message


def test_local_flask_requires_image():
    from flask_app import app

    response = app.test_client().post("/api/generate", data={})
    assert response.status_code == 400
    assert "starting image" in response.json["detail"]


def test_remote_log_lines_are_cleaned_and_classified():
    from flask_app import clean_remote_log_line, remote_log_level

    assert clean_remote_log_line("\x1b[32mLoading checkpoint 4/12\x1b[0m\n") == "Loading checkpoint 4/12"
    assert "modal" not in clean_remote_log_line("Modal.com worker ready").lower()
    assert clean_remote_log_line("Following logs for ap-example...") == ""
    assert remote_log_level("Video decoding completed") == "success"
    assert remote_log_level("CUDA out of memory error") == "error"
    assert remote_log_level("Warning: slow startup") == "warning"


def test_remote_log_stream_rejects_invalid_job_id_without_starting_process():
    from flask_app import app

    response = app.test_client().get("/api/jobs/not-a-call/events")
    assert response.status_code == 400
