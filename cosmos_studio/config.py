from __future__ import annotations

import json
import math
from dataclasses import dataclass


MODEL_ID = "nvidia/Cosmos3-Super-Image2Video"


@dataclass(frozen=True)
class Preset:
    label: str
    frames: int
    steps: int
    guidance: float


PRESETS = {
    "draft": Preset("Draft", frames=81, steps=25, guidance=4.0),
    "balanced": Preset("Balanced", frames=121, steps=35, guidance=5.0),
    "quality": Preset("Quality", frames=189, steps=50, guidance=6.0),
}

SIZES = {
    "landscape": (832, 480, "16,9"),
    "portrait": (480, 832, "9,16"),
    "square": (480, 480, "1,1"),
}

DEFAULT_NEGATIVE_PROMPT = (
    "macroblocking artifacts, chromatic aberration, high-frequency noise, rolling "
    "shutter distortion, static with no motion, excessive motion blur, over-saturation, "
    "shaky footage, low resolution, grainy texture, pixelated images, poor lighting, "
    "underexposure, overexposure, poor color balance, washed out colors, choppy motion, "
    "jerky movement, low frame rate, compression artifacts, color banding, unnatural "
    "transitions, jump cuts, visual noise, flickering, object morphing, duplicate subjects"
)


def ulysses_compatible_frames(
    *, frames: int, width: int, height: int, ulysses_degree: int = 4
) -> int:
    """Return the nearest valid frame count for Cosmos sequence parallelism.

    Cosmos3 compresses time by 4 and patches latent spatial dimensions by 32.
    vLLM-Omni requires the resulting generation sequence to divide evenly
    across the configured Ulysses workers. Valid video lengths are 4n+1.
    """
    if frames < 1:
        raise ValueError("Frame count must be positive.")
    if (frames - 1) % 4:
        frames += 4 - ((frames - 1) % 4)

    spatial_tokens = math.ceil(height / 32) * math.ceil(width / 32)
    while True:
        temporal_tokens = (frames - 1) // 4 + 1
        if (temporal_tokens * spatial_tokens) % ulysses_degree == 0:
            return frames
        frames += 4


def build_generation_request(
    *,
    prompt: str,
    negative_prompt: str = "",
    preset_name: str = "balanced",
    size_name: str = "landscape",
    seed: int = 42,
) -> dict:
    """Validate UI input and build the exact vLLM-Omni request fields."""
    cleaned_prompt = " ".join(prompt.split())
    if len(cleaned_prompt) < 8:
        raise ValueError("Describe the motion in a little more detail.")
    if len(cleaned_prompt) > 4_000:
        raise ValueError("Prompt must be 4,000 characters or fewer.")
    if preset_name not in PRESETS:
        raise ValueError("Unknown quality preset.")
    if size_name not in SIZES:
        raise ValueError("Unknown aspect ratio.")
    if not 0 <= seed <= 2_147_483_647:
        raise ValueError("Seed must be between 0 and 2,147,483,647.")

    preset = PRESETS[preset_name]
    width, height, aspect_ratio = SIZES[size_name]
    frames = ulysses_compatible_frames(
        frames=preset.frames,
        width=width,
        height=height,
    )
    fps = 24
    duration = max(1, round(frames / fps))

    # Cosmos 3 performs best with a JSON temporal-caption envelope. This keeps the
    # UI self-contained and does not silently send the user's image to a second VLM.
    prompt_envelope = {
        "temporal_caption": cleaned_prompt,
        "duration": f"{duration}s",
        "fps": float(fps),
        "resolution": {"H": height, "W": width},
        "aspect_ratio": aspect_ratio,
    }

    return {
        "prompt": json.dumps(prompt_envelope, ensure_ascii=False),
        "negative_prompt": negative_prompt.strip() or DEFAULT_NEGATIVE_PROMPT,
        "size": f"{width}x{height}",
        "num_frames": frames,
        "fps": fps,
        "num_inference_steps": preset.steps,
        "guidance_scale": preset.guidance,
        "flow_shift": 5.0,
        "seed": seed,
        "extra_params": json.dumps(
            {
                "guardrails": False,
                "use_resolution_template": False,
                "use_duration_template": False,
            }
        ),
    }
