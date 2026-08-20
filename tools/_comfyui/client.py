"""Thin REST client for a running ComfyUI server.

Handles the full generation cycle: submit workflow, poll for completion,
download artifacts.  Used by comfyui_image, comfyui_video, and comfyui_music.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import requests

from tools._comfyui.workflow_profiles import (
    DEFAULT_COMFYUI_SERVER_URL,
    resolve_comfyui_server_url,
)


class ComfyUIError(Exception):
    """Raised when ComfyUI returns an error or times out."""


class ComfyUIProbeError(ComfyUIError):
    """Raised when metadata/probe endpoints fail transiently or unexpectedly."""


class ComfyUIClient:
    """Client for the ComfyUI REST API.

    The protocol is simple and battle-tested:
      1. POST /prompt           → queue a workflow, get a prompt_id
      2. GET  /history/{id}     → poll until outputs appear
      3. GET  /view?filename=…  → download the generated artifact
      4. POST /upload/image     → stage a local image for I2V workflows
    """

    def __init__(self, server_url: str | None = None) -> None:
        self.server_url = resolve_comfyui_server_url(server_url)
        if server_url:
            self._url_source = "explicit"
        elif os.environ.get("COMFYUI_SERVER_URL"):
            self._url_source = "env"
        else:
            self._url_source = "fallback"

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @property
    def is_default_url(self) -> bool:
        """True if using the fallback URL with no explicit override set."""
        return self._url_source == "fallback" and self.server_url == DEFAULT_COMFYUI_SERVER_URL

    def is_available(self) -> bool:
        """Return True if the ComfyUI server is reachable."""
        try:
            resp = requests.get(
                f"{self.server_url}/system_stats", timeout=5
            )
            return resp.status_code == 200
        except Exception:
            return False

    def unavailable_reason(self) -> str:
        """Human-readable explanation of why the server can't be reached."""
        if self.is_default_url:
            return (
                f"No ComfyUI server found at {self.server_url} "
                f"(default — no COMFYUI_SERVER_URL configured).\n"
                f"Set COMFYUI_SERVER_URL in your .env file to the address of "
                f"your ComfyUI server (fallback default: {self.server_url})."
            )
        return (
            f"ComfyUI server not reachable at {self.server_url}.\n"
            f"Check that ComfyUI is running and the URL is correct."
        )

    def has_node_class(self, node_class: str) -> bool:
        try:
            data = self.object_info(node_class)
            return node_class in data
        except Exception:
            return False

    def object_info(self, node_class: str) -> dict[str, Any]:
        try:
            resp = requests.get(f"{self.server_url}/object_info/{node_class}", timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            raise ComfyUIProbeError(
                f"Failed to query ComfyUI object_info for {node_class}: {exc}"
            ) from exc

    def node_options(self, node_class: str, field: str) -> list[str]:
        """Return enumerated options for a node input field, or [] if unavailable."""
        try:
            resp = requests.get(f"{self.server_url}/object_info/{node_class}", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            options = (
                data.get(node_class, {})
                .get("input", {})
                .get("required", {})
                .get(field, [[]])[0]
            )
            return options if isinstance(options, list) else []
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Model discovery
    # ------------------------------------------------------------------

    def list_models(self) -> dict[str, list[str]]:
        """Query ComfyUI for available models, grouped by type.

        Returns a dict like::

            {
                "checkpoints": ["sd_xl_base.safetensors", ...],
                "diffusion_models": ["flux2-dev-nvfp4.safetensors", ...],
                "vae": ["ae.safetensors", ...],
                "clip": ["clip_l.safetensors", ...],
                "loras": ["my_lora.safetensors", ...],
            }
        """
        node_to_key = {
            "CheckpointLoaderSimple": ("ckpt_name", "checkpoints"),
            "UNETLoader": ("unet_name", "diffusion_models"),
            "VAELoader": ("vae_name", "vae"),
            "CLIPLoader": ("clip_name", "clip"),
            "LoraLoaderModelOnly": ("lora_name", "loras"),
        }
        result: dict[str, list[str]] = {}
        for node_class, (field, group) in node_to_key.items():
            try:
                result[group] = self.node_options(node_class, field)
            except Exception:
                result[group] = []
        return result

    def check_models(
        self, required: list[str]
    ) -> tuple[list[str], list[str]]:
        """Check which of *required* model filenames are available.

        Returns ``(found, missing)`` — two lists of filenames.
        """
        all_models: set[str] = set()
        for names in self.list_models().values():
            all_models.update(names)

        found = [m for m in required if m in all_models]
        missing = [m for m in required if m not in all_models]
        return found, missing

    # ------------------------------------------------------------------
    # Core cycle
    # ------------------------------------------------------------------

    def submit(self, workflow: dict) -> str:
        """Queue a workflow for execution.  Returns the ``prompt_id``."""
        resp = requests.post(
            f"{self.server_url}/prompt",
            json={"prompt": workflow},
            timeout=30,
        )
        try:
            data = resp.json()
        except ValueError:
            data = {}
        if data.get("node_errors"):
            raise ComfyUIError(f"Node errors: {json.dumps(data['node_errors'])}")
        if data.get("error"):
            raise ComfyUIError(f"Prompt error: {json.dumps(data['error'])}")
        resp.raise_for_status()
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise ComfyUIError(f"No prompt_id in response: {data}")
        return prompt_id

    def poll(
        self,
        prompt_id: str,
        *,
        timeout: int = 600,
        interval: int = 5,
    ) -> dict:
        """Block until *prompt_id* finishes.  Returns the history entry."""
        timeout = int(os.environ.get("COMFYUI_POLL_TIMEOUT", str(timeout)))
        interval = int(os.environ.get("COMFYUI_POLL_INTERVAL", str(interval)))
        deadline = time.time() + timeout
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                resp = requests.get(
                    f"{self.server_url}/history/{prompt_id}", timeout=10
                )
                resp.raise_for_status()
                history = resp.json()
                last_error = None
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(interval)
                continue
            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    msgs = status.get("messages", [])
                    raise ComfyUIError(f"Execution error: {msgs}")
                return entry
            time.sleep(interval)
        if last_error is not None:
            raise ComfyUIError(
                f"Prompt {prompt_id} did not complete within {timeout}s after poll errors: {last_error}"
            )
        raise ComfyUIError(
            f"Prompt {prompt_id} did not complete within {timeout}s"
        )

    def download(
        self,
        filename: str,
        subfolder: str,
        dest: Path,
        folder_type: str = "output",
    ) -> Path:
        """Download an output artifact from the ComfyUI server."""
        resp = requests.get(
            f"{self.server_url}/view",
            params={
                "filename": filename,
                "subfolder": subfolder,
                "type": folder_type,
            },
            timeout=120,
        )
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        if dest.stat().st_size <= 0:
            raise ComfyUIError(f"Downloaded empty artifact for {filename!r}")
        return dest

    def upload_image(self, local_path: Path, name: str) -> str:
        """Upload a local image so it can be referenced by LoadImage nodes.

        Returns the server-side filename.
        """
        with open(local_path, "rb") as f:
            resp = requests.post(
                f"{self.server_url}/upload/image",
                files={"image": (name, f, "image/png")},
                timeout=30,
            )
        resp.raise_for_status()
        return resp.json()["name"]

    @staticmethod
    def unique_upload_name(local_path: Path, stem_hint: str, *, suffix: str | None = None) -> str:
        ext = suffix or local_path.suffix or ".png"
        digest = hashlib.sha256(
            f"{local_path}:{time.time_ns()}:{random.random()}".encode()
        ).hexdigest()[:12]
        safe_stem = "".join(
            ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem_hint
        )[:48] or "upload"
        return f"{safe_stem}_{digest}{ext}"

    # ------------------------------------------------------------------
    # High-level helper
    # ------------------------------------------------------------------

    def generate(
        self,
        workflow: dict,
        output_node: str,
        dest: Path,
        *,
        timeout: int = 600,
        interval: int = 5,
    ) -> list[Path]:
        """Submit → poll → download.  Returns list of artifact paths."""
        existing_metadata = dict(getattr(self, "last_run_metadata", {}) or {})
        self.last_run_metadata = existing_metadata
        self.last_run_metadata.setdefault("server_url", self.server_url)
        prompt_id = self.submit(workflow)
        entry = self.poll(prompt_id, timeout=timeout, interval=interval)
        self.last_run_metadata["prompt_id"] = prompt_id

        outputs = entry.get("outputs", {})
        node_output = outputs.get(output_node, {})

        # ComfyUI stores images and videos under the "images" key
        items = node_output.get("images", []) or node_output.get("gifs", [])
        if not items:
            raise ComfyUIError(
                f"No output artifacts on node {output_node}. "
                f"Available nodes: {list(outputs.keys())}"
            )

        paths: list[Path] = []
        for i, item in enumerate(items):
            suffix = Path(item["filename"]).suffix
            if len(items) == 1:
                target = dest
            else:
                target = dest.with_stem(f"{dest.stem}_{i:03d}").with_suffix(suffix)
            self.download(
                item["filename"],
                item.get("subfolder", ""),
                target,
                item.get("type", "output"),
            )
            paths.append(target)
        self.last_run_metadata["artifacts"] = [
            {
                "filename": item.get("filename"),
                "subfolder": item.get("subfolder", ""),
                "type": item.get("type", "output"),
            }
            for item in items
        ]
        return paths

    # ------------------------------------------------------------------
    # Workflow helpers
    # ------------------------------------------------------------------

    @staticmethod
    def load_workflow(path: Path) -> dict:
        """Load a workflow JSON template from disk."""
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def patch_workflow(
        workflow: dict, patches: dict[str, dict[str, Any]]
    ) -> dict:
        """Deep-copy *workflow* and apply *patches*.

        *patches* maps ``node_id`` → ``{input_name: value, ...}``.
        """
        w = copy.deepcopy(workflow)
        for node_id, values in patches.items():
            if node_id not in w:
                raise ComfyUIError(
                    f"Node {node_id!r} not found in workflow. "
                    f"Available: {list(w.keys())}"
                )
            for key, val in values.items():
                w[node_id]["inputs"][key] = val
        return w

    @staticmethod
    def random_seed() -> int:
        """Return a random seed suitable for ComfyUI noise nodes."""
        return random.randint(0, 2**32 - 1)
