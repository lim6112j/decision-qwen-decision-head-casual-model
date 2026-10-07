"""llama-server process manager + HTTP client for chat/embedding API."""

import json
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests


class LlamaServer:
    """Context manager that starts/stops llama-server as a subprocess."""

    def __init__(self, gguf_path: Path, port: int = 8080, context_length: int = 2048):
        self.gguf_path = Path(gguf_path).expanduser().resolve()
        self.port = port
        self.context_length = context_length
        self._proc: Optional[subprocess.Popen] = None
        self._log_file = None
        self.base_url = f"http://127.0.0.1:{port}"

    @property
    def running(self) -> bool:
        if self._proc is None:
            return False
        poll = self._proc.poll()
        return poll is None

    def start(self) -> None:
        """Launch llama-server subprocess."""
        if self.running:
            return
        if not self.gguf_path.exists():
            raise FileNotFoundError(f"GGUF model not found: {self.gguf_path}")

        cmd = [
            "llama-server",
            "-m", str(self.gguf_path),
            "--port", str(self.port),
            "--ctx-size", str(self.context_length),
            "--embeddings",
            "--pooling", "last",
            "--no-webui",
        ]
        # Log to a file rather than pipes: unbounded llama-server output on
        # PIPE buffers deadlocks the subprocess once the buffer fills.
        self._log_file = open(f"/tmp/llama-server-{self.port}.log", "w")
        self._proc = subprocess.Popen(
            cmd, stdout=self._log_file, stderr=subprocess.STDOUT,
        )

        self._wait_ready()

    def stop(self) -> None:
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._proc = None
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def _wait_ready(self, timeout: float = 60.0) -> None:
        """Poll /health until server responds."""
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if self._proc and self._proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited early (code {self._proc.returncode}); "
                    f"see /tmp/llama-server-{self.port}.log"
                )
            try:
                r = requests.get(f"{self.base_url}/health", timeout=2)
                if r.status_code == 200:
                    return
            except (requests.ConnectionError, requests.Timeout):
                pass
            time.sleep(0.5)
        raise TimeoutError(f"llama-server not ready after {timeout}s")

    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 128) -> str:
        """POST /v1/chat/completions, return content text."""
        body = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            # Qwen3.5 is a thinking model; without this it burns all max_tokens
            # inside reasoning_content and content comes back empty.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        r = requests.post(f"{self.base_url}/v1/chat/completions", json=body, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """POST /v1/embeddings, return list of embedding vectors."""
        body = {"input": texts}
        r = requests.post(f"{self.base_url}/v1/embeddings", json=body, timeout=120)
        r.raise_for_status()
        data = r.json()["data"]
        # sort by index to preserve input order
        data.sort(key=lambda d: d["index"])
        return [d["embedding"] for d in data]

    def __enter__(self) -> "LlamaServer":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()