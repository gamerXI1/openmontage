"""Local Kokoro TTS provider with repo-relative runtime discovery."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from lib.paths import REPO_ROOT
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


class LocalKokoroTTS(BaseTool):
    name = "local_kokoro_tts"
    version = "0.1.0"
    tier = ToolTier.VOICE
    capability = "tts"
    provider = "kokoro"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL_GPU

    dependencies = []
    DEFAULT_BAKEOFF_PYTHON = REPO_ROOT.parent / "tts-bakeoff" / ".venv" / "bin" / "python"
    install_instructions = (
        "Requires a Python environment with kokoro + soundfile installed. "
        "Set KOKORO_TTS_PYTHON, pass python_path explicitly, or install Kokoro in the repo sibling runtime "
        f"at {DEFAULT_BAKEOFF_PYTHON}."
    )
    fallback = "local_edge_tts"
    fallback_tools = ["local_edge_tts", "piper_tts"]
    agent_skills = ["text-to-speech"]

    capabilities = [
        "text_to_speech",
        "local_generation",
        "natural_voice_local",
    ]
    supports = {
        "voice_cloning": False,
        "multilingual": True,
        "offline": True,
        "native_audio": True,
        "local_network": False,
    }
    best_for = [
        "natural local narration default",
        "high-quality offline-ish TTS on spark",
    ]
    not_good_for = [
        "voice clone matching",
        "strict minimal-dependency environments",
    ]

    input_schema = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {"type": "string"},
            "voice": {
                "type": "string",
                "default": "af_heart",
                "description": "Kokoro voice ID.",
            },
            "lang_code": {
                "type": "string",
                "default": "a",
                "description": "Kokoro language code; 'a' is English.",
            },
            "repo_id": {
                "type": "string",
                "default": "hexgrad/Kokoro-82M",
            },
            "output_path": {"type": "string"},
            "python_path": {
                "type": "string",
                "default": str(DEFAULT_BAKEOFF_PYTHON),
            },
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=2, ram_mb=2048, vram_mb=2048, disk_mb=200, network_required=False
    )
    retry_policy = RetryPolicy(max_retries=1, retryable_errors=[])
    idempotency_key_fields = ["text", "voice", "lang_code", "repo_id"]
    side_effects = ["writes audio file to output_path"]
    user_visible_verification = ["Listen to generated audio for naturalness and pacing"]

    @classmethod
    def _candidate_python_paths(cls, inputs: dict[str, Any]) -> list[str]:
        candidates: list[str] = []

        explicit = inputs.get("python_path")
        if explicit:
            candidates.append(str(explicit))

        env_python = os.environ.get("KOKORO_TTS_PYTHON")
        if env_python:
            candidates.append(env_python)

        candidates.append(str(cls.DEFAULT_BAKEOFF_PYTHON))

        current_python = shutil.which("python3") or shutil.which("python")
        if current_python:
            candidates.append(current_python)

        deduped: list[str] = []
        for candidate in candidates:
            if candidate and candidate not in deduped:
                deduped.append(candidate)
        return deduped

    def _python_path(self, inputs: dict[str, Any]) -> str:
        for candidate in self._candidate_python_paths(inputs):
            if Path(candidate).exists():
                return candidate
        return self._candidate_python_paths(inputs)[0]

    def get_status(self) -> ToolStatus:
        for candidate in self._candidate_python_paths({}):
            if Path(candidate).exists():
                return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        if self.get_status() != ToolStatus.AVAILABLE:
            return ToolResult(success=False, error="Local Kokoro TTS not available. " + self.install_instructions)

        start = time.time()
        try:
            result = self._generate(inputs)
        except Exception as exc:
            return ToolResult(success=False, error=f"Local Kokoro TTS failed: {exc}")

        result.duration_seconds = round(time.time() - start, 2)
        return result

    def _generate(self, inputs: dict[str, Any]) -> ToolResult:
        from tools.analysis.audio_probe import probe_duration

        output_path = Path(inputs.get("output_path", "local_kokoro_tts.wav"))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "text": inputs["text"],
            "voice": inputs.get("voice", "af_heart"),
            "lang_code": inputs.get("lang_code", "a"),
            "repo_id": inputs.get("repo_id", "hexgrad/Kokoro-82M"),
            "output_path": str(output_path),
        }

        script = r'''
import json, sys
from pathlib import Path
import numpy as np
import soundfile as sf
from kokoro import KPipeline

payload = json.loads(sys.stdin.read())
pipeline = KPipeline(lang_code=payload.get("lang_code", "a"), repo_id=payload.get("repo_id", "hexgrad/Kokoro-82M"))
chunks = []
for _, _, audio in pipeline(payload["text"], voice=payload.get("voice", "af_heart")):
    chunks.append(audio)
if not chunks:
    raise RuntimeError("Kokoro returned no audio chunks")
out = Path(payload["output_path"])
out.parent.mkdir(parents=True, exist_ok=True)
wav = np.concatenate(chunks)
sf.write(out, wav, 24000)
print(json.dumps({"sample_rate": 24000, "samples": int(len(wav)), "output": str(out)}))
'''
        proc = subprocess.run(
            [self._python_path(inputs), "-c", script],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=600,
        )
        if proc.returncode != 0:
            return ToolResult(success=False, error=f"Kokoro failed (exit {proc.returncode}): {proc.stderr[:2000]}")
        if not output_path.exists():
            return ToolResult(success=False, error=f"Kokoro output file missing: {output_path}")

        meta = {}
        stdout = (proc.stdout or "").strip().splitlines()
        if stdout:
            try:
                meta = json.loads(stdout[-1])
            except Exception:
                meta = {"stdout": proc.stdout[-500:]}

        audio_duration = probe_duration(output_path)
        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "model": inputs.get("repo_id", "hexgrad/Kokoro-82M"),
                "voice": inputs.get("voice", "af_heart"),
                "lang_code": inputs.get("lang_code", "a"),
                "text_length": len(inputs["text"]),
                "audio_duration_seconds": round(audio_duration, 2) if audio_duration else None,
                "output": str(output_path),
                **meta,
            },
            artifacts=[str(output_path)],
            model=inputs.get("repo_id", "hexgrad/Kokoro-82M"),
        )
