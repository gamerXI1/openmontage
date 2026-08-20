"""ComfyUI video generation via a local or remote ComfyUI server.

Supports text-to-video and image-to-video using WAN 2.2 14B with
4-step LightX2V LoRA acceleration.  Custom workflows are accepted
via the ``workflow_json`` input.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import requests

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)
from tools._comfyui.client import ComfyUIClient, ComfyUIError
from tools._comfyui.metadata import (
    BUNDLED_MODEL_STACKS,
    COMFYUI_SETUP_OFFER,
    missing_models_payload,
    model_stack,
    workflow_hash,
)
from tools._comfyui.workflow_profiles import (
    DEFAULT_COMFYUI_SERVER_URL,
    get_video_workflow_profile,
    has_video_workflow_profile,
)

_WORKFLOWS = Path(__file__).resolve().parent.parent / "_comfyui" / "workflows"

# Output node IDs in the bundled workflows
_T2V_OUTPUT_NODE = "16"
_I2V_OUTPUT_NODE = "108"

# Models required by the bundled WAN 2.2 workflows
_REQUIRED_MODELS_COMMON = [
    "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
]
_REQUIRED_MODELS_I2V = [
    *_REQUIRED_MODELS_COMMON,
    "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
    "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
    "wan_2.1_vae.safetensors",
    "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
    "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
]
_REQUIRED_MODELS_T2V = [
    *_REQUIRED_MODELS_COMMON,
    "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
    "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
    "wan_2.1_vae.safetensors",
    "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
    "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors",
]

_RESOURCE_PROFILES = {
    "provider_floor": {
        "vram_mb": 8000,
        "ram_mb": 16000,
        "applies_to": (
            "ComfyUI provider availability and low-VRAM custom workflows. "
            "Actual requirements depend on workflow_json/workflow_path."
        ),
    },
    "bundled_wan22_14b_fp8": {
        "vram_mb": 16000,
        "ram_mb": 32000,
        "applies_to": (
            "Bundled WAN 2.2 14B FP8 T2V/I2V workflows. This is not a "
            "ComfyUI provider-wide requirement."
        ),
    },
    "low_vram_custom_workflows": {
        "vram_mb": "8000-12000",
        "ram_mb": "16000-32000",
        "examples": [
            "Wan 2.1 1.3B",
            "LTX-Video / LTXV FP8 or quantized workflows",
            "Wan 2.2 GGUF / quantized community workflows",
        ],
    },
}


class ComfyUIVideo(BaseTool):
    name = "comfyui_video"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "video_generation"
    provider = "comfyui"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.SEEDED
    runtime = ToolRuntime.LOCAL_GPU

    dependencies = []
    setup_offer = COMFYUI_SETUP_OFFER
    install_instructions = (
        "Start a ComfyUI server and set COMFYUI_SERVER_URL "
        f"(default {DEFAULT_COMFYUI_SERVER_URL}).\n"
        "Bundled WAN workflows require WAN 2.2 models and LightX2V LoRAs. "
        "The repo-owned ltx25_i2v_explicit_vae profile additionally requires the LTX 2.5 custom node stack and matching model files."
    )
    agent_skills = ["comfyui", "ai-video-gen", "ltx2"]

    capabilities = ["text_to_video", "image_to_video"]
    supports = {
        "seed": True,
        "reference_image": True,
        "custom_workflow": True,
        "custom_output_node": True,
        "offline": True,
    }
    best_for = [
        "local GPU video generation without API costs",
        "Blackwell / DGX Spark hardware where diffusers is unsupported",
        "image-to-video with WAN 2.2 14B (4-step accelerated)",
        "text-to-video with WAN 2.2 14B (4-step accelerated)",
        "custom low-VRAM ComfyUI workflows on 8GB-12GB GPUs",
    ]
    not_good_for = [
        "setups without a running ComfyUI server",
        "CPU-only machines",
        "running the bundled WAN 2.2 14B FP8 workflows on GPUs below 16GB VRAM",
    ]
    fallback = "wan_video"
    fallback_tools = ["wan_video", "hunyuan_video", "ltx_video_local", "kling_video"]

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string", "description": "Text prompt for video generation"},
            "operation": {
                "type": "string",
                "enum": ["text_to_video", "image_to_video"],
                "default": "text_to_video",
            },
            "reference_image_path": {
                "type": "string",
                "description": "Local path to reference image (for image_to_video)",
            },
            "reference_image_url": {
                "type": "string",
                "description": "URL of reference image (for image_to_video, downloaded first)",
            },
            "width": {"type": "integer", "default": 832, "description": "T2V default 832, I2V default 640"},
            "height": {"type": "integer", "default": 480, "description": "T2V default 480, I2V default 640"},
            "num_frames": {"type": "integer", "default": 81, "description": "81 frames = 5s at 16fps"},
            "seed": {"type": "integer", "description": "Random if omitted"},
            "output_path": {"type": "string", "description": "Where to save the video"},
            "negative_prompt": {
                "type": "string",
                "description": "Optional negative prompt; supported by bundled and repo-profile ComfyUI video workflows.",
            },
            "workflow_json": {
                "type": "string",
                "description": "Optional full ComfyUI workflow JSON. Requires output_node.",
            },
            "workflow_profile": {
                "type": "string",
                "description": (
                    "Optional repo-owned custom workflow profile. "
                    "Use ltx25_i2v_explicit_vae for the verified local LTX 2.5 I2V stack."
                ),
            },
            "workflow_path": {
                "type": "string",
                "description": "Optional path to a ComfyUI workflow JSON file. Requires output_node.",
            },
            "output_node": {
                "type": "string",
                "description": "ComfyUI output node ID for custom workflow_json/workflow_path.",
            },
            "workflow_name": {
                "type": "string",
                "description": "Optional human-readable provenance label for a custom workflow.",
            },
            "workflow_model": {
                "type": "string",
                "description": "Optional model/provenance label for a custom workflow.",
            },
            "workflow_model_stack": {
                "type": "array",
                "description": (
                    "Optional provenance metadata for custom workflow dependencies. "
                    "Items should include name, role, quantization, scheduler, "
                    "and LoRA strengths when known."
                ),
                "items": {"type": "object"},
            },
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=2, ram_mb=16000, vram_mb=8000, disk_mb=2000, network_required=False,
    )
    retry_policy = RetryPolicy(max_retries=1, retryable_errors=["timeout"])
    idempotency_key_fields = ["prompt", "operation", "width", "height", "num_frames", "seed"]
    side_effects = ["writes video file to output_path"]
    user_visible_verification = ["Watch generated clip for motion coherence and artifacts"]

    def __init__(self) -> None:
        self._client = ComfyUIClient()

    def get_status(self) -> ToolStatus:
        if not self._client.is_available():
            return ToolStatus.UNAVAILABLE
        statuses = self.operation_statuses()
        if any(status == "available" for status in statuses.values()):
            return ToolStatus.AVAILABLE
        if statuses:
            return ToolStatus.DEGRADED
        return ToolStatus.UNAVAILABLE

    def operation_statuses(self) -> dict[str, str]:
        """Return per-operation readiness for selector routing and preflight."""
        if not self._client.is_available():
            return {
                "text_to_video": "unavailable",
                "image_to_video": "unavailable",
            }

        _, missing_t2v = self._client.check_models(_REQUIRED_MODELS_T2V)
        _, missing_i2v = self._client.check_models(_REQUIRED_MODELS_I2V)
        return {
            "text_to_video": "available" if not missing_t2v else "degraded",
            "image_to_video": "available" if not missing_i2v else "degraded",
        }

    def is_operation_available(self, operation: str) -> bool:
        if operation not in {"text_to_video", "image_to_video"}:
            return False
        return self.operation_statuses().get(operation) == "available"

    def get_info(self) -> dict[str, Any]:
        info = super().get_info()
        info["operation_statuses"] = self.operation_statuses()
        info["resource_profiles"] = _RESOURCE_PROFILES
        info["setup_offer"] = self.setup_offer
        info["bundled_model_stacks"] = {
            "text_to_video": BUNDLED_MODEL_STACKS["wan22-t2v-4step"],
            "image_to_video": BUNDLED_MODEL_STACKS["wan22-i2v-4step"],
        }
        info["resource_profile_note"] = (
            "The top-level resource_profile is a ComfyUI provider floor, not a "
            "promise that every workflow fits 8GB VRAM. Bundled WAN 2.2 14B FP8 "
            "workflows recommend 16GB VRAM; custom low-VRAM workflows can target "
            "8GB-12GB depending on model, quantization, resolution, and frame count."
        )
        return info

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        operation = inputs.get("operation", "text_to_video")
        if operation == "image_to_video":
            return 210.0  # ~3.5 min
        return 240.0  # ~4 min

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        profile_name = inputs.get("workflow_profile")
        custom_workflow = bool(
            inputs.get("workflow_json")
            or inputs.get("workflow_path")
            or profile_name
        )
        if profile_name and not has_video_workflow_profile(str(profile_name)):
            return ToolResult(
                success=False,
                error=(
                    f"Unknown ComfyUI workflow_profile {profile_name!r}. "
                    "Known profiles: ltx25_i2v_explicit_vae"
                ),
            )
        if custom_workflow and not (inputs.get("output_node") or profile_name):
            return ToolResult(
                success=False,
                error=(
                    "Custom ComfyUI workflows require output_node so OpenMontage "
                    "knows which ComfyUI node to download artifacts from."
                ),
            )

        if not self._client.is_available():
            return ToolResult(
                success=False,
                error=self._client.unavailable_reason(),
            )

        operation = inputs.get("operation", "text_to_video")
        if profile_name:
            profile_meta = get_video_workflow_profile(str(profile_name))
            expected_operation = str(profile_meta["operation"])
            if "operation" in inputs and operation != expected_operation:
                return ToolResult(
                    success=False,
                    error=(
                        f"workflow_profile {profile_name!r} only supports operation "
                        f"{expected_operation!r}, got {operation!r}."
                    ),
                )
            profile_ready_error = self._profile_readiness_error(profile_meta)
            if profile_ready_error:
                return ToolResult(success=False, error=profile_ready_error)
            operation = expected_operation

        if not custom_workflow:
            required = _REQUIRED_MODELS_I2V if operation == "image_to_video" else _REQUIRED_MODELS_T2V
            _, missing = self._client.check_models(required)
            if missing:
                workflow_key = (
                    "wan22-i2v-4step"
                    if operation == "image_to_video"
                    else "wan22-t2v-4step"
                )
                return ToolResult(
                    success=False,
                    data=missing_models_payload(
                        missing,
                        workflow_key=workflow_key,
                        workflow_name=f"{workflow_key}.json",
                        operation=operation,
                    ),
                    error=(
                        f"ComfyUI server is running but missing models for {operation}: "
                        f"{', '.join(missing)}.\n"
                        f"See data.missing_models for destination hints and download URLs."
                    ),
                )
        start = time.time()
        seed = inputs.get("seed") or ComfyUIClient.random_seed()
        output_path = Path(
            inputs.get("output_path", f"comfyui_video_{operation}_{seed}.mp4")
        )
        fps = 16
        video_timeout = int(os.environ.get("COMFYUI_VIDEO_TIMEOUT", "900"))
        poll_interval = int(os.environ.get("COMFYUI_POLL_INTERVAL", "10"))

        try:
            if profile_name:
                workflow, output_node, profile_meta = self._build_profile_workflow(
                    str(profile_name),
                    inputs,
                    seed,
                    output_path,
                )
                operation = str(profile_meta["operation"])
                fps = int(profile_meta.get("frame_rate", fps))
            elif custom_workflow:
                workflow = self._load_custom_workflow(inputs)
                output_node = str(inputs["output_node"])
            elif operation == "image_to_video":
                workflow, output_node = self._build_i2v(inputs, seed, output_path)
            else:
                workflow, output_node = self._build_t2v(inputs, seed, output_path)

            paths = self._client.generate(
                workflow,
                output_node=output_node,
                dest=output_path,
                timeout=video_timeout,
                interval=poll_interval,
            )
            self._validate_video_artifacts(paths)
            provenance = self._workflow_provenance(
                inputs,
                custom_workflow,
                output_node,
                operation,
                workflow,
                runtime_metadata=getattr(self._client, "last_run_metadata", {}),
            )

        except ComfyUIError as exc:
            return ToolResult(success=False, error=str(exc))
        except Exception as exc:
            return ToolResult(success=False, error=f"ComfyUI video generation failed: {exc}")

        width = inputs.get("width", 832 if operation == "text_to_video" else 640)
        height = inputs.get("height", 480 if operation == "text_to_video" else 640)
        num_frames = inputs.get("num_frames", 81)
        if profile_name:
            profile_meta = get_video_workflow_profile(str(profile_name))
            width = inputs.get("width", int(profile_meta.get("default_width", width)))
            height = inputs.get("height", int(profile_meta.get("default_height", height)))
            num_frames = inputs.get("num_frames", int(profile_meta.get("default_num_frames", num_frames)))

        model_name = self._model_name(inputs, custom_workflow)
        return ToolResult(
            success=True,
            data={
                "provider": "comfyui",
                "model": model_name,
                "prompt": inputs["prompt"],
                "operation": operation,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "fps": fps,
                "duration_seconds": round(num_frames / fps, 2),
                "output": str(paths[0]),
                "format": "mp4",
                "workflow_provenance": provenance,
            },
            artifacts=[str(p) for p in paths],
            cost_usd=0.0,
            duration_seconds=round(time.time() - start, 2),
            seed=seed,
            model=model_name,
        )

    # ------------------------------------------------------------------
    # Workflow builders
    # ------------------------------------------------------------------

    def _build_t2v(
        self, inputs: dict[str, Any], seed: int, output_path: Path
    ) -> tuple[dict, str]:
        width = inputs.get("width", 832)
        height = inputs.get("height", 480)
        num_frames = inputs.get("num_frames", 81)

        workflow = ComfyUIClient.load_workflow(_WORKFLOWS / "wan22-t2v-4step.json")
        workflow = ComfyUIClient.patch_workflow(workflow, {
            "2": {"text": inputs["prompt"]},
            "11": {"width": width, "height": height, "batch_size": num_frames},
            "12": {"noise_seed": seed},
            "16": {"filename_prefix": output_path.stem},
        })
        return workflow, _T2V_OUTPUT_NODE

    def _build_i2v(
        self, inputs: dict[str, Any], seed: int, output_path: Path
    ) -> tuple[dict, str]:
        width = inputs.get("width", 640)
        height = inputs.get("height", 640)
        num_frames = inputs.get("num_frames", 81)

        # Resolve reference image
        ref_path = inputs.get("reference_image_path")
        ref_url = inputs.get("reference_image_url")

        if ref_url and not ref_path:
            # Download to a temp location
            resp = requests.get(ref_url, timeout=60)
            resp.raise_for_status()
            ref_path = str(output_path.with_suffix(".ref.png"))
            Path(ref_path).parent.mkdir(parents=True, exist_ok=True)
            Path(ref_path).write_bytes(resp.content)

        if not ref_path:
            raise ComfyUIError(
                "image_to_video requires reference_image_path or reference_image_url"
            )

        # Upload to ComfyUI
        upload_name = f"om_{output_path.stem}.png"
        server_name = self._client.upload_image(Path(ref_path), upload_name)

        workflow = ComfyUIClient.load_workflow(_WORKFLOWS / "wan22-i2v-4step.json")
        workflow = ComfyUIClient.patch_workflow(workflow, {
            "93": {"text": inputs["prompt"]},
            "97": {"image": server_name},
            "98": {"width": width, "height": height, "length": num_frames},
            "86": {"noise_seed": seed},
            "108": {"filename_prefix": output_path.stem},
        })
        return workflow, _I2V_OUTPUT_NODE

    def _build_profile_workflow(
        self,
        profile_name: str,
        inputs: dict[str, Any],
        seed: int,
        output_path: Path,
    ) -> tuple[dict[str, Any], str, dict[str, Any]]:
        profile = get_video_workflow_profile(profile_name)
        operation = str(profile["operation"])
        if operation != "image_to_video":
            raise ComfyUIError(
                f"workflow_profile {profile_name!r} currently supports only image_to_video"
            )

        ref_path = inputs.get("reference_image_path")
        ref_url = inputs.get("reference_image_url")
        if ref_url and not ref_path:
            resp = requests.get(ref_url, timeout=60)
            resp.raise_for_status()
            ref_path = str(output_path.with_suffix(".ref.png"))
            Path(ref_path).parent.mkdir(parents=True, exist_ok=True)
            Path(ref_path).write_bytes(resp.content)

        if not ref_path:
            raise ComfyUIError(
                f"workflow_profile {profile_name!r} requires reference_image_path or reference_image_url"
            )

        upload_name = ComfyUIClient.unique_upload_name(Path(ref_path), output_path.stem)
        server_name = self._client.upload_image(Path(ref_path), upload_name)
        workflow = ComfyUIClient.load_workflow(Path(profile["workflow_path"]))
        bindings = dict(profile.get("input_bindings", {}))
        image_binding = bindings["reference_image"]
        prompt_binding = bindings["prompt"]
        neg_binding = bindings["negative_prompt"]
        dims_binding = bindings["dimensions"]
        seed_binding = bindings["seed"]
        out_binding = bindings["output"]
        workflow = ComfyUIClient.patch_workflow(workflow, {
            str(image_binding["node"]): {str(image_binding["field"]): server_name},
            str(prompt_binding["node"]): {str(prompt_binding["field"]): inputs["prompt"]},
            str(neg_binding["node"]): {str(neg_binding["field"]): inputs.get("negative_prompt", profile["default_negative_prompt"])},
            str(dims_binding["node"]): {
                str(dims_binding["width_field"]): inputs.get("width", profile["default_width"]),
                str(dims_binding["height_field"]): inputs.get("height", profile["default_height"]),
                str(dims_binding["frames_field"]): inputs.get("num_frames", profile["default_num_frames"]),
            },
            str(seed_binding["node"]): {str(seed_binding["field"]): seed},
            str(out_binding["node"]): {str(out_binding["field"]): output_path.stem},
        })
        self._client.last_run_metadata = {
            "upload_name": upload_name,
            "server_upload_name": server_name,
            "server_url": self._client.server_url,
        }
        return workflow, str(profile["output_node"]), profile

    def _validate_video_artifacts(self, paths: list[Path]) -> None:
        for path in paths:
            if not path.exists() or path.stat().st_size <= 0:
                raise ComfyUIError(f"Generated video artifact missing or empty: {path}")
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "stream=codec_name,nb_frames",
                    "-show_entries",
                    "format=duration,size",
                    "-of",
                    "json",
                    str(path),
                ],
                capture_output=True,
                text=True,
            )
            if probe.returncode != 0:
                raise ComfyUIError(
                    f"Generated video artifact is not a valid playable media file: {path} ({probe.stderr.strip()})"
                )
            try:
                payload = json.loads(probe.stdout or "{}")
            except json.JSONDecodeError as exc:
                raise ComfyUIError(
                    f"Generated video artifact probe returned invalid JSON for {path}: {exc}"
                ) from exc
            streams = payload.get("streams", [])
            fmt = payload.get("format", {})
            if not streams:
                raise ComfyUIError(f"Generated video artifact has no media streams: {path}")
            if float(fmt.get("duration", 0) or 0) <= 0:
                raise ComfyUIError(f"Generated video artifact has non-positive duration: {path}")

    def workflow_profile_ready(self, profile_name: str, *, operation: str | None = None) -> bool:
        profile = get_video_workflow_profile(profile_name)
        if operation and operation != str(profile.get("operation")):
            return False
        return self._profile_readiness_error(profile) is None

    def workflow_profile_readiness_error(self, profile_name: str, *, operation: str | None = None) -> str | None:
        if not self._client.is_available():
            return self._client.unavailable_reason()
        profile = get_video_workflow_profile(profile_name)
        if operation and operation != str(profile.get("operation")):
            return (
                f"workflow_profile {profile_name!r} only supports operation "
                f"{str(profile.get('operation'))!r}, got {operation!r}."
            )
        return self._profile_readiness_error(profile)

    def _profile_readiness_error(self, profile: dict[str, Any]) -> str | None:
        probe_errors: list[str] = []
        missing_nodes: list[str] = []
        for node_class in profile.get("required_node_classes", []):
            try:
                data = self._client.object_info(str(node_class))
            except Exception as exc:
                probe_errors.append(str(exc))
                continue
            if str(node_class) not in data:
                missing_nodes.append(str(node_class))
        found_models: set[str] = set()
        missing_models: list[str] = []
        for binding in profile.get("required_model_bindings", []):
            model_name = str(binding.get("model"))
            try:
                options = self._client.node_options(
                    str(binding.get("node_class")), str(binding.get("field"))
                )
            except Exception as exc:
                probe_errors.append(str(exc))
                continue
            if model_name in options:
                found_models.add(model_name)
            else:
                missing_models.append(model_name)
        if probe_errors:
            unique_probe_errors = []
            for err in probe_errors:
                if err not in unique_probe_errors:
                    unique_probe_errors.append(err)
            return (
                f"ComfyUI readiness probe failed for workflow_profile {profile['workflow_name']!r}: "
                + " | ".join(unique_probe_errors)
            )
        if not missing_models:
            _, generic_missing = self._client.check_models(
                [m for m in profile.get("required_models", []) if m not in found_models]
            )
            missing_models.extend(generic_missing)
        if not missing_nodes and not missing_models:
            return None

        parts: list[str] = []
        if missing_nodes:
            parts.append(f"missing node classes: {', '.join(missing_nodes)}")
        if missing_models:
            parts.append(f"missing models: {', '.join(missing_models)}")
        return (
            f"ComfyUI server is reachable but workflow_profile {profile['workflow_name']!r} is not runnable: "
            + "; ".join(parts)
        )

    @staticmethod
    def _load_custom_workflow(inputs: dict[str, Any]) -> dict:
        if inputs.get("workflow_json"):
            return json.loads(inputs["workflow_json"])
        return ComfyUIClient.load_workflow(Path(inputs["workflow_path"]))

    @staticmethod
    def _model_name(inputs: dict[str, Any], custom_workflow: bool) -> str:
        if not custom_workflow:
            return "wan2.2-14b-fp8-4step"
        if inputs.get("workflow_profile"):
            return str(inputs["workflow_profile"])
        return (
            inputs.get("workflow_model")
            or inputs.get("model")
            or inputs.get("workflow_name")
            or "custom-comfyui-workflow"
        )

    @staticmethod
    def _workflow_provenance(
        inputs: dict[str, Any],
        custom_workflow: bool,
        output_node: str,
        operation: str,
        workflow: dict[str, Any],
        runtime_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not custom_workflow:
            workflow_key = (
                "wan22-i2v-4step"
                if operation == "image_to_video"
                else "wan22-t2v-4step"
            )
            return {
                "source": "bundled",
                "workflow": (
                    "wan22-i2v-4step.json"
                    if operation == "image_to_video"
                    else "wan22-t2v-4step.json"
                ),
                "workflow_hash_sha256": workflow_hash(workflow),
                "model_stack": model_stack(workflow_key, inputs),
                "output_node": output_node,
            }
        if inputs.get("workflow_profile"):
            profile = get_video_workflow_profile(str(inputs["workflow_profile"]))
            metadata = runtime_metadata or {}
            artifact_meta = metadata.get("artifacts", [])
            return {
                "source": "repo_profile",
                "workflow_profile": inputs.get("workflow_profile"),
                "workflow_name": profile.get("workflow_name"),
                "workflow_path": str(Path("tools") / "_comfyui" / "workflows" / str(profile.get("workflow_name"))),
                "model": profile.get("workflow_model"),
                "workflow_hash_sha256": workflow_hash(workflow),
                "model_stack": profile.get("model_stack", []),
                "required_node_classes": profile.get("required_node_classes", []),
                "required_models": profile.get("required_models", []),
                "model_stack_source": "repo_profile",
                "output_node": output_node,
                "server_url": metadata.get("server_url"),
                "prompt_id": metadata.get("prompt_id"),
                "upload_name": metadata.get("upload_name"),
                "server_upload_name": metadata.get("server_upload_name"),
                "artifact_records": artifact_meta,
            }
        return {
            "source": "user_supplied",
            "workflow_name": inputs.get("workflow_name"),
            "workflow_path": inputs.get("workflow_path"),
            "model": inputs.get("workflow_model") or inputs.get("model"),
            "workflow_hash_sha256": workflow_hash(workflow),
            "model_stack": model_stack(None, inputs),
            "model_stack_source": (
                "caller_supplied"
                if inputs.get("workflow_model_stack")
                else "unknown_custom_workflow"
            ),
            "output_node": output_node,
        }
