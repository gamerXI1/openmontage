"""Local Edge TTS provider via spark-41db gateway."""

from __future__ import annotations

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


class LocalEdgeTTS(BaseTool):
    name = "local_edge_tts"
    version = "0.1.0"
    tier = ToolTier.VOICE
    capability = "tts"
    provider = "local_edge"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.API

    dependencies = []
    install_instructions = (
        "Requires the local spark-41db voice gateway on http://127.0.0.1:30003 "
        "serving /v1/audio/speech or /synthesize."
    )
    fallback = "piper_tts"
    fallback_tools = ["piper_tts"]
    agent_skills = ["text-to-speech"]

    capabilities = [
        "text_to_speech",
        "local_gateway",
        "openai_compatible_speech",
    ]
    supports = {
        "voice_cloning": False,
        "multilingual": True,
        "offline": False,
        "native_audio": True,
        "local_network": True,
    }
    best_for = [
        "local spark voice pipeline",
        "low-friction narration on spark-41db",
    ]
    not_good_for = [
        "strict offline-only workflows",
        "voice clone matching",
    ]

    input_schema = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {"type": "string"},
            "voice": {
                "type": "string",
                "default": "en-US-AriaNeural",
                "description": "Voice name exposed by the local Edge TTS gateway.",
            },
            "model": {
                "type": "string",
                "default": "edge-tts",
                "description": "Logical model name for the local gateway.",
            },
            "format": {
                "type": "string",
                "default": "mp3",
                "enum": ["mp3"],
            },
            "response_format": {
                "type": "string",
                "default": "mp3",
                "enum": ["mp3"],
            },
            "speed": {
                "type": "number",
                "default": 1.0,
                "minimum": 0.25,
                "maximum": 4.0,
            },
            "output_path": {"type": "string"},
            "base_url": {
                "type": "string",
                "default": "http://127.0.0.1:30003",
            },
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=128, vram_mb=0, disk_mb=50, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=1, retryable_errors=["timeout", "connection"])
    idempotency_key_fields = ["text", "voice", "model", "response_format", "speed"]
    side_effects = ["writes audio file to output_path", "calls local voice gateway"]
    user_visible_verification = ["Listen to generated audio for intelligibility and timing"]

    def get_status(self) -> ToolStatus:
        base_url = "http://127.0.0.1:30003"
        try:
            r = requests.get(f"{base_url}/v1/models", timeout=3)
            if r.ok:
                return ToolStatus.AVAILABLE
        except Exception:
            pass
        return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        if self.get_status() != ToolStatus.AVAILABLE:
            return ToolResult(success=False, error="Local Edge TTS not available. " + self.install_instructions)

        start = time.time()
        try:
            result = self._generate(inputs)
        except Exception as exc:
            return ToolResult(success=False, error=f"Local Edge TTS failed: {exc}")

        result.duration_seconds = round(time.time() - start, 2)
        return result

    def _generate(self, inputs: dict[str, Any]) -> ToolResult:
        from tools.analysis.audio_probe import probe_duration

        text = inputs["text"]
        model = inputs.get("model", "edge-tts")
        voice = inputs.get("voice", "en-US-AriaNeural")
        fmt = inputs.get("response_format") or inputs.get("format", "mp3")
        speed = inputs.get("speed", 1.0)
        base_url = inputs.get("base_url", "http://127.0.0.1:30003").rstrip("/")

        output_path = Path(inputs.get("output_path", f"local_edge_tts.{fmt}"))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": fmt,
            "speed": speed,
        }
        response = requests.post(f"{base_url}/v1/audio/speech", json=payload, timeout=180)
        if not response.ok:
            return ToolResult(success=False, error=f"Local Edge gateway failed ({response.status_code}): {response.text[:1000]}")

        output_path.write_bytes(response.content)
        audio_duration = probe_duration(output_path)

        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "model": model,
                "voice": voice,
                "format": fmt,
                "response_format": fmt,
                "speed": speed,
                "text_length": len(text),
                "audio_duration_seconds": round(audio_duration, 2) if audio_duration else None,
                "output": str(output_path),
                "base_url": base_url,
            },
            artifacts=[str(output_path)],
            model=model,
        )
