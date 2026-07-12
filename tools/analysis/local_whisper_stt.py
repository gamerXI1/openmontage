"""Local faster-whisper STT provider via spark-41db gateway."""

from __future__ import annotations

import json
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
    ToolStability,
    ToolStatus,
    ToolTier,
)


class LocalWhisperSTT(BaseTool):
    name = "local_whisper_stt"
    version = "0.1.0"
    tier = ToolTier.CORE
    capability = "analysis"
    provider = "local_whisper"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC

    dependencies = []
    install_instructions = (
        "Requires the local spark-41db whisper gateway on http://127.0.0.1:30008 "
        "serving /v1/models and /v1/transcribe."
    )
    agent_skills = ["speech-to-text"]

    capabilities = [
        "transcribe",
        "word_timestamps",
        "language_detection",
        "segments",
    ]
    supports = {
        "diarization": False,
        "offline": False,
        "local_network": True,
        "subtitle_safe": True,
    }
    best_for = [
        "local spark transcription",
        "subtitle timing with local gateway",
        "low-friction narration verification",
    ]
    not_good_for = [
        "speaker diarization",
        "fully offline no-network workflows",
    ]

    input_schema = {
        "type": "object",
        "required": ["input_path"],
        "properties": {
            "input_path": {"type": "string", "description": "Path to audio or video file"},
            "language": {"type": "string", "description": "Optional ISO 639-1 hint. Current gateway auto-detects if omitted."},
            "output_dir": {"type": "string", "description": "Directory for output transcript JSON"},
            "base_url": {"type": "string", "default": "http://127.0.0.1:30008"},
        },
    }

    output_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "segments": {"type": "array"},
            "word_timestamps": {"type": "array"},
            "language": {"type": "string"},
            "duration_seconds": {"type": "number"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1,
        ram_mb=256,
        vram_mb=0,
        disk_mb=50,
        network_required=True,
    )
    retry_policy = RetryPolicy(max_retries=1, retryable_errors=["timeout", "connection"])
    idempotency_key_fields = ["input_path", "language"]
    side_effects = ["writes transcript JSON to output_dir", "calls local whisper gateway"]
    user_visible_verification = [
        "Check transcript text against source audio",
        "Verify word timestamps align with speech",
    ]

    def get_status(self) -> ToolStatus:
        try:
            r = requests.get("http://127.0.0.1:30008/v1/models", timeout=3)
            if r.ok:
                return ToolStatus.AVAILABLE
        except Exception:
            pass
        return ToolStatus.UNAVAILABLE

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 30.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        if self.get_status() != ToolStatus.AVAILABLE:
            return ToolResult(success=False, error="Local whisper gateway not available. " + self.install_instructions)

        input_path = Path(inputs["input_path"]).expanduser()
        if not input_path.exists():
            return ToolResult(success=False, error=f"Input file not found: {input_path}")

        output_dir = Path(inputs.get("output_dir", input_path.parent)).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        base_url = str(inputs.get("base_url", "http://127.0.0.1:30008")).rstrip("/")

        start = time.time()
        try:
            with input_path.open("rb") as fh:
                response = requests.post(
                    f"{base_url}/v1/transcribe",
                    files={"file": (input_path.name, fh)},
                    timeout=300,
                )
        except Exception as exc:
            return ToolResult(success=False, error=f"Local whisper request failed: {exc}")

        if not response.ok:
            return ToolResult(success=False, error=f"Local whisper gateway failed ({response.status_code}): {response.text[:1000]}")

        try:
            payload = response.json()
        except Exception as exc:
            return ToolResult(success=False, error=f"Local whisper returned non-JSON response: {exc}")

        result_data = {
            "text": payload.get("text", ""),
            "segments": payload.get("segments", []),
            "word_timestamps": payload.get("word_timestamps", []),
            "language": payload.get("language"),
            "duration_seconds": payload.get("duration"),
            "provider": self.provider,
            "base_url": base_url,
            "supports_diarization": False,
        }

        output_path = output_dir / f"{input_path.stem}_local_whisper_transcript.json"
        output_path.write_text(json.dumps(result_data, indent=2), encoding="utf-8")

        return ToolResult(
            success=True,
            data=result_data,
            artifacts=[str(output_path)],
            duration_seconds=round(time.time() - start, 2),
        )
