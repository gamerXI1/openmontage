"""Repo-owned ComfyUI workflow profiles for first-class custom paths."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_COMFYUI_SERVER_URL = "http://localhost:8188"
_WORKFLOW_DIR = Path(__file__).resolve().parent / "workflows"

VIDEO_WORKFLOW_PROFILES: dict[str, dict[str, Any]] = {
    "ltx25_i2v_explicit_vae": {
        "operation": "image_to_video",
        "workflow_path": _WORKFLOW_DIR / "ltx25-i2v-explicit-vae.json",
        "output_node": "12",
        "workflow_name": "ltx25-i2v-explicit-vae.json",
        "workflow_model": "ltx-2.5-i2v-explicit-vae",
        "frame_rate": 8,
        "default_width": 512,
        "default_height": 320,
        "default_num_frames": 9,
        "required_models": [
            "ltx-2.5-22b-distilled-transformer-nvfp4.safetensors",
            "ltx-2.5-video-vae-bf16.safetensors",
            "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot/model.safetensors",
        ],
        "required_model_bindings": [
            {
                "role": "checkpoint",
                "node_class": "CheckpointLoaderSimple",
                "field": "ckpt_name",
                "model": "ltx-2.5-22b-distilled-transformer-nvfp4.safetensors",
            },
            {
                "role": "vae",
                "node_class": "VAELoader",
                "field": "vae_name",
                "model": "ltx-2.5-video-vae-bf16.safetensors",
            },
            {
                "role": "text_encoder",
                "node_class": "LTXVGemmaCLIPModelLoader",
                "field": "gemma_path",
                "model": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot/model.safetensors",
            },
        ],
        "required_node_classes": [
            "CheckpointLoaderSimple",
            "VAELoader",
            "LTXVGemmaCLIPModelLoader",
            "LoadImage",
            "LTXVPreprocess",
            "CLIPTextEncode",
            "LTXVConditioning",
            "LTXVImgToVideo",
            "KSampler",
            "VAEDecode",
            "VHS_VideoCombine",
        ],
        "input_bindings": {
            "reference_image": {"node": "4", "field": "image"},
            "prompt": {"node": "6", "field": "text"},
            "negative_prompt": {"node": "7", "field": "text"},
            "dimensions": {
                "node": "9",
                "width_field": "width",
                "height_field": "height",
                "frames_field": "length",
            },
            "seed": {"node": "10", "field": "seed"},
            "output": {"node": "12", "field": "filename_prefix"},
        },
        "default_negative_prompt": (
            "blurry, warped face, extra limbs, aggressive motion, "
            "cinematic camera move, text, watermark"
        ),
        "model_stack": [
            {
                "role": "checkpoint",
                "name": "ltx-2.5-22b-distilled-transformer-nvfp4.safetensors",
                "quantization": "NVFP4",
                "destination_hint": "ComfyUI/models/checkpoints/",
            },
            {
                "role": "vae",
                "name": "ltx-2.5-video-vae-bf16.safetensors",
                "quantization": "BF16",
                "destination_hint": "ComfyUI/models/vae/",
            },
            {
                "role": "text_encoder",
                "name": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot/model.safetensors",
                "quantization": "INT8 convrot",
                "destination_hint": "ComfyUI/models/text_encoders/ or custom-node-managed path",
            },
            {
                "role": "sampler",
                "name": "KSampler",
                "steps": 2,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
            },
        ],
    }
}


def resolve_comfyui_server_url(explicit_url: str | None = None) -> str:
    """Resolve the ComfyUI base URL from explicit input, env, then repo default."""
    return (explicit_url or os.environ.get("COMFYUI_SERVER_URL") or DEFAULT_COMFYUI_SERVER_URL).rstrip("/")


def has_video_workflow_profile(name: str | None) -> bool:
    return bool(name and name in VIDEO_WORKFLOW_PROFILES)


def get_video_workflow_profile(name: str) -> dict[str, Any]:
    try:
        profile = VIDEO_WORKFLOW_PROFILES[name]
    except KeyError as exc:
        known = ", ".join(sorted(VIDEO_WORKFLOW_PROFILES)) or "<none>"
        raise KeyError(f"Unknown ComfyUI video workflow_profile {name!r}. Known profiles: {known}") from exc
    return {
        **profile,
        "workflow_path": Path(profile["workflow_path"]),
        "model_stack": [dict(item) for item in profile.get("model_stack", [])],
        "required_models": list(profile.get("required_models", [])),
        "required_model_bindings": [dict(item) for item in profile.get("required_model_bindings", [])],
        "required_node_classes": list(profile.get("required_node_classes", [])),
        "input_bindings": dict(profile.get("input_bindings", {})),
    }
