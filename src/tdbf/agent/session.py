"""Persistent per-instance Docker container for an agent run.

The standalone harness helpers (`run_snippet_in_container`,
`run_test_in_container`) each do a fresh `docker run --rm` — ~10-40s of startup.
An agent makes many tool calls, so that startup would dominate the run. This
keeps **one** container alive (`docker run -d ... sleep infinity`) and `docker
exec`s into it for every tool call, so startup is paid once.

GOLD-FREE: this session operates only on the buggy checkout (base_commit). It
has no method that applies the gold patch or runs against the fixed tree. The
gold patch is offline scoring only and never reaches here.

Colima/macOS: ensure `DOCKER_HOST=unix://${HOME}/.colima/default/docker.sock` is
set in the shell (this module will set it as a fallback if it's unset and the
socket exists). File transfer uses `docker cp`/stdin via the Docker API, so the
`~/`-only mount limitation that affects bind mounts does not apply here.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

# Activation prefix shared by every exec: SWE-bench images put the testbed env in
# a conda env named "testbed"; the repo is checked out at /testbed.
_ACTIVATE = "source /opt/miniconda3/bin/activate && conda activate testbed && cd /testbed && "


def _ensure_docker_host() -> None:
    """Default DOCKER_HOST to the Colima socket on macOS if unset."""
    if os.environ.get("DOCKER_HOST"):
        return
    sock = Path.home() / ".colima" / "default" / "docker.sock"
    if sock.exists():
        os.environ["DOCKER_HOST"] = f"unix://{sock}"


@dataclass
class ExecResult:
    returncode: int
    output: str  # stdout + stderr, combined (what the model/parser sees)


class ContainerSession:
    """A long-lived container for one instance. Use as a context manager."""

    def __init__(self, instance: object, *, keep: bool = False) -> None:
        self.instance = instance
        self.instance_id: str = instance.instance_id  # type: ignore[attr-defined]
        self.repo: str = getattr(instance, "repo", "")
        self.image = f"sweb.eval.x86_64.{self.instance_id}:latest"
        self.keep = keep
        self.container: str | None = None
        self._pytest_ready = False

    # -- lifecycle ---------------------------------------------------------- #

    def start(self) -> "ContainerSession":
        if self.container is not None:
            return self
        _ensure_docker_host()
        name = f"tdbf-{self.instance_id}-{uuid.uuid4().hex[:8]}"
        proc = subprocess.run(
            ["docker", "run", "-d", "--name", name, self.image, "sleep", "infinity"],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            err = proc.stderr.strip()
            if "Unable to find image" in err or "No such image" in err:
                raise RuntimeError(
                    f"Docker image {self.image!r} not found. "
                    "Run scripts/01_validate_gold.py first to build it."
                )
            raise RuntimeError(f"Failed to start container for {self.instance_id}: {err}")
        self.container = proc.stdout.strip() or name
        return self

    def close(self) -> None:
        if self.container is None or self.keep:
            return
        subprocess.run(
            ["docker", "rm", "-f", self.container],
            capture_output=True, text=True,
        )
        self.container = None

    def __enter__(self) -> "ContainerSession":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    # -- primitives --------------------------------------------------------- #

    def exec(
        self,
        command: str,
        *,
        timeout: int = 120,
        activate: bool = True,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Run a shell command inside the live container.

        With activate=True the conda testbed env is sourced and cwd is /testbed
        (the normal case). Combined stdout+stderr is returned so callers see
        tracebacks the same way the model will.

        Untrusted (model-supplied) strings must be passed via `env` and
        referenced as "$VAR" in `command`, never interpolated into the command
        text — that keeps grep patterns / paths from being shell-injected.
        """
        if self.container is None:
            raise RuntimeError("Session not started; call start() or use as a context manager.")
        script = (_ACTIVATE + command) if activate else command
        cmd = ["docker", "exec"]
        for key, value in (env or {}).items():
            cmd += ["-e", f"{key}={value}"]
        cmd += [self.container, "bash", "-c", script]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return ExecResult(returncode=124, output=f"timeout after {timeout}s")
        return ExecResult(returncode=proc.returncode, output=proc.stdout + proc.stderr)

    def write_file(self, path: str, content: str) -> None:
        """Write a file inside the container at an absolute path.

        Streams the content over stdin (`cat >`) so no host bind mount is
        needed. Creates the parent directory first.
        """
        if self.container is None:
            raise RuntimeError("Session not started.")
        parent = os.path.dirname(path)
        if parent:
            self.exec(f"mkdir -p {parent}", activate=False)
        proc = subprocess.run(
            ["docker", "exec", "-i", self.container, "bash", "-c", f"cat > {path}"],
            input=content, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"write_file({path}) failed: {proc.stderr.strip()}")

    def read_file(self, path: str, *, max_bytes: int = 0) -> ExecResult:
        """Read a file from the buggy checkout. `path` is relative to /testbed
        unless absolute. Optionally cap the byte count read (head -c)."""
        target = path if path.startswith("/") else f"/testbed/{path}"
        reader = f'head -c {int(max_bytes)} -- "$TARGET"' if max_bytes else 'cat -- "$TARGET"'
        return self.exec(reader, activate=False, env={"TARGET": target})

    def ensure_pytest(self) -> None:
        """Install pytest once if the testbed env lacks it (some Django envs do)."""
        if self._pytest_ready:
            return
        self.exec("python -m pytest --version > /dev/null 2>&1 || pip install pytest -q", timeout=180)
        self._pytest_ready = True

    def reset_tree(self) -> None:
        """Restore the checkout to a pristine base_commit state.

        Not needed by the read-only BRT agent (Phase A) but available for the
        fix agent (Phase B), which mutates source. Still gold-free: it resets to
        base_commit, never to the fixed tree.
        """
        self.exec("git checkout -- . && git clean -fdq", activate=True)
