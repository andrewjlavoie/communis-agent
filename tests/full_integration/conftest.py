"""Shared fixtures for full integration tests.

Provides:
- ensure_temporal: session-scoped — connects to localhost:7233 or starts the dev server
- temporal_client: per-test async Temporal client against the shared server
- model_backend: parametrized across Anthropic Haiku and local Qwen
- configure_llm_backend: patches llm_activities module globals per backend
- isolate_workspace: routes workspace I/O to a temp dir
- load_config: factory that loads a YAML config and builds a CommunisConfig
"""
from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path

import urllib.request
import urllib.error

import pytest
from temporalio.client import Client

from cli.main import _load_config_file
from models.data_types import CommunisConfig

CONFIGS_DIR = Path(__file__).parent / "configs"
TEMPORAL_ADDRESS = "localhost:7233"


# ── Temporal server management (session-scoped) ─────────────────────


def _temporal_is_reachable(host: str = "localhost", port: int = 7233) -> bool:
    """Check if a Temporal server is accepting TCP connections."""
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def ensure_temporal():
    """Ensure a Temporal dev server is running for the entire test session.

    If localhost:7233 is already reachable (user started it themselves or it's
    already running), use it as-is. Otherwise, start `temporal server start-dev`
    as a subprocess and tear it down when the session ends.
    """
    if _temporal_is_reachable():
        print("\n  [temporal] Using existing server at localhost:7233")
        yield
        return

    print("\n  [temporal] No server at localhost:7233 — starting temporal server start-dev ...")
    proc = subprocess.Popen(
        ["temporal", "server", "start-dev", "--headless"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for the server to become reachable
    for attempt in range(30):
        if _temporal_is_reachable():
            print(f"  [temporal] Dev server ready (took ~{attempt}s)")
            break
        time.sleep(1)
    else:
        proc.kill()
        pytest.fail("Temporal dev server failed to start within 30s")

    yield

    print("\n  [temporal] Shutting down dev server (pid={proc.pid})")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


# ── Temporal client (per-test, async) ────────────────────────────────


@pytest.fixture
async def temporal_client(ensure_temporal):
    """Async Temporal client connected to the shared server."""
    client = await Client.connect(TEMPORAL_ADDRESS)
    return client


# ── Backend definitions ──────────────────────────────────────────────


BACKENDS = {
    "haiku": {
        "provider": "anthropic",
        "model": "claude-haiku-4-5-20251001",
        "fast_model": "claude-haiku-4-5-20251001",
        "fast_max_tokens": 0,
        "base_url": "",
        "env_overrides": {},
        "skip_check": lambda: (
            not os.getenv("ANTHROPIC_API_KEY"),
            "ANTHROPIC_API_KEY not set",
        ),
    },
    "qwen_local": {
        "provider": "openai",
        "model": "qwen/qwen3.5-9b",
        "fast_model": "qwen/qwen3.5-9b",
        "fast_max_tokens": 4096,
        "base_url": "http://192.168.5.71:1234/v1",
        "env_overrides": {
            "OPENAI_BASE_URL": "http://192.168.5.71:1234/v1",
            "OPENAI_API_KEY": "lm-studio",
        },
        "skip_check": lambda: _check_openai_endpoint("http://192.168.5.71:1234/v1"),
    },
}


def _check_openai_endpoint(base_url: str) -> tuple[bool, str]:
    """Return (should_skip, reason) for an OpenAI-compatible endpoint."""
    try:
        req = urllib.request.Request(f"{base_url}/models", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                return (False, "")
            return (True, f"Endpoint returned {resp.status}")
    except urllib.error.URLError as e:
        return (True, f"Cannot connect to {base_url}: {e.reason}")
    except Exception as e:
        return (True, str(e))


# ── Parametrized backend fixture ─────────────────────────────────────


@pytest.fixture(params=list(BACKENDS.keys()), ids=list(BACKENDS.keys()))
def model_backend(request):
    """Yield a backend config dict, skipping if the backend is unavailable."""
    backend = BACKENDS[request.param]
    should_skip, reason = backend["skip_check"]()
    if should_skip:
        pytest.skip(f"Backend '{request.param}' unavailable: {reason}")
    return backend


# ── LLM module patching ─────────────────────────────────────────────


@pytest.fixture(autouse=True)
def configure_llm_backend(model_backend):
    """Patch llm_activities module globals to route to the selected backend."""
    import activities.llm_activities as mod

    saved = {
        "provider": mod.LLM_PROVIDER,
        "default_model": mod.DEFAULT_MODEL,
        "fast_model": mod.FAST_MODEL,
        "fast_max_tokens": mod.FAST_MAX_TOKENS,
        "anthropic_client": mod._anthropic_client,
        "openai_clients": mod._openai_clients.copy(),
    }
    saved_env: dict[str, str | None] = {}

    mod.LLM_PROVIDER = model_backend["provider"]
    mod.DEFAULT_MODEL = model_backend["model"]
    mod.FAST_MODEL = model_backend["fast_model"]
    mod.FAST_MAX_TOKENS = model_backend["fast_max_tokens"]
    mod._anthropic_client = None
    mod._openai_clients.clear()

    for key, val in model_backend["env_overrides"].items():
        saved_env[key] = os.environ.get(key)
        os.environ[key] = val

    yield

    mod.LLM_PROVIDER = saved["provider"]
    mod.DEFAULT_MODEL = saved["default_model"]
    mod.FAST_MODEL = saved["fast_model"]
    mod.FAST_MAX_TOKENS = saved["fast_max_tokens"]
    mod._anthropic_client = saved["anthropic_client"]
    mod._openai_clients = saved["openai_clients"]

    for key, original in saved_env.items():
        if original is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original


# ── Workspace isolation ──────────────────────────────────────────────


@pytest.fixture(autouse=True)
def isolate_workspace(tmp_path):
    """Route workspace I/O to a temp dir so tests don't pollute .communis/."""
    import activities.workspace_activities as ws_mod

    saved = ws_mod.WORKSPACE_BASE
    ws_mod.WORKSPACE_BASE = str(tmp_path / "workspaces")
    yield tmp_path / "workspaces"
    ws_mod.WORKSPACE_BASE = saved


# ── Config loading factory ───────────────────────────────────────────


@pytest.fixture
def load_config(model_backend):
    """Factory fixture: load_config("quick_sanity.yaml") -> CommunisConfig.

    Model/provider/base_url from the YAML are overridden by the parametrized
    model_backend so the same scenario runs against both backends.
    """
    def _load(filename: str) -> CommunisConfig:
        cfg = _load_config_file(str(CONFIGS_DIR / filename))

        no_goal_detect = cfg.get("no_goal_detect", False)

        return CommunisConfig(
            idea=cfg.get("prompt", ""),
            max_turns=cfg.get("turns", 0),
            model=model_backend["model"],
            auto=cfg.get("auto", False),
            provider=model_backend["provider"],
            base_url=model_backend["base_url"],
            dangerous=cfg.get("dangerous", False),
            goal_complete_detection=not no_goal_detect,
            max_subcommunis=max(0, min(5, cfg.get("max_subcommunis", 3))),
        )

    return _load
