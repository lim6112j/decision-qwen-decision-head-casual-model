"""llama-server process manager + HTTP client for chat/embedding API."""

import json
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests


class LlamaServer:
    """Context manager that starts/stops llama-server as a subprocess.

    Backend-agnostic: the same process manager serves a causal LM (Qwen, for
    chat + embeddings) or a dedicated embedding GGUF (e.g. bge-m3, embeddings
    only). Only ``embed()`` is assumed to work — ``chat()`` needs a model with
    a chat template.
    """

    def __init__(
        self,
        gguf_path: Path,
        port: int = 8080,
        context_length: int = 2048,
        pooling: str = "last",
        embed_dim: int = 1024,
        batch_size: int = 0,
    ):
        self.gguf_path = Path(gguf_path).expanduser().resolve()
        self.port = port
        self.context_length = context_length
        # Physical/ubatch size. llama.cpp defaults to 512 tokens and rejects a
        # single input longer than that with a 500 ("input is too large to
        # process"). Long fields (e.g. real JSON states) exceed it, so allow
        # raising it. Does not change the embeddings, so it is deliberately NOT
        # part of `backend_id`.
        self.batch_size = batch_size
        # Pooling must match the model family: causal LMs pool the last token,
        # XLM-R/BERT-family embedding models pool the CLS token.
        self.pooling = pooling
        self.embed_dim = embed_dim
        self._proc: Optional[subprocess.Popen] = None
        self._log_file = None
        self._adopted = False
        self._served_model_path: Optional[str] = None
        self.base_url = f"http://127.0.0.1:{port}"

    @property
    def backend_id(self) -> str:
        """Identity of the embedding backend behind this server.

        Cache fingerprints hash this, so embeddings extracted from one
        backend can never be silently reused for another. Prefer what the
        server reports — ``start()`` adopts any healthy server on the port,
        so ``gguf_path`` is not authoritative — and fall back to local file
        identity when ``/props`` is unavailable.
        """
        source = (
            f"served={self._served_model_path}"
            if self._served_model_path
            else f"file={self._local_file_id()}"
        )
        return "|".join([
            source,
            f"pooling={self.pooling}",
            f"ctx={self.context_length}",
            f"dim={self.embed_dim}",
        ])

    def _local_file_id(self) -> str:
        try:
            st = self.gguf_path.stat()
        except OSError:
            return f"{self.gguf_path.name}|missing"
        return f"{self.gguf_path.name}|size={st.st_size}|mtime={int(st.st_mtime)}"

    def _probe_served_model(self) -> None:
        """Best-effort: ask the server which weights it actually loaded.

        Never raises — an older build without /props, or a server that
        answers it oddly, must not break startup.
        """
        try:
            r = requests.get(f"{self.base_url}/props", timeout=5)
            if r.status_code != 200:
                return
            path = r.json().get("model_path")
        except (requests.RequestException, ValueError):
            return
        if isinstance(path, str) and path:
            self._served_model_path = path

    @property
    def running(self) -> bool:
        if self._proc is None:
            return False
        poll = self._proc.poll()
        return poll is None

    def _health_ok(self, timeout: float = 2.0) -> bool:
        """True if a server already answers /health on the port."""
        try:
            r = requests.get(f"{self.base_url}/health", timeout=timeout)
        except requests.RequestException:
            return False
        return r.status_code == 200

    def start(self) -> None:
        """Launch llama-server subprocess, or adopt a healthy one already up.

        If something is already answering /health on the port (e.g. a remote
        llama-server reached over an ssh tunnel), reuse it and do NOT spawn a
        local process — stop() then leaves it untouched.

        Consistency caveat: embedding vectors depend on the served backend
        (model file, quantization, pooling). Feature caches extracted from
        one backend and option/question embeddings from another land in
        different spaces and silently corrupt the head — extract and train
        against the SAME server. ``backend_id`` is folded into the feature
        cache fingerprint so a stale cache is now detected instead of
        silently reused; option/question embeddings are still embedded live
        at train time, so the port discipline still matters.
        """
        if self.running:
            return
        if self._health_ok():
            self._adopted = True
            self._probe_served_model()
            return
        if not self.gguf_path.exists():
            raise FileNotFoundError(f"GGUF model not found: {self.gguf_path}")

        cmd = [
            "llama-server",
            "-m", str(self.gguf_path),
            "--port", str(self.port),
            "--ctx-size", str(self.context_length),
            "--embeddings",
            "--pooling", self.pooling,
            "--no-webui",
            # Errors only. At the default verbosity the server logs one
            # "srv update" line per prompt in every batch, which grew the
            # log file past 30 GB during a single feature extraction.
            # Errors (E ...) still land in the log, so a startup failure
            # stays diagnosable.
            "--verbosity", "1",
        ]
        if self.batch_size:
            cmd += ["--batch-size", str(self.batch_size),
                    "--ubatch-size", str(self.batch_size)]
        # Log to a file rather than pipes: unbounded llama-server output on
        # PIPE buffers deadlocks the subprocess once the buffer fills.
        self._log_file = open(f"/tmp/llama-server-{self.port}.log", "w")
        self._proc = subprocess.Popen(
            cmd, stdout=self._log_file, stderr=subprocess.STDOUT,
        )

        self._wait_ready()
        self._probe_served_model()

    def stop(self) -> None:
        if self._proc is None:
            return
        if getattr(self, "_adopted", False):
            # We didn't spawn this server (a remote/tunneled one) — leave it up.
            self._proc = None
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


def server_from_config(cfg) -> LlamaServer:
    """Build a LlamaServer from a Config's ``model`` section.

    Single construction point: call sites used to pass only path/port/ctx,
    which silently dropped pooling and dimension — the settings that decide
    whether an embedding cache is still valid.
    """
    return LlamaServer(
        cfg.model.gguf_path,
        port=cfg.model.server_port,
        context_length=cfg.model.context_length,
        pooling=cfg.model.pooling,
        embed_dim=cfg.model.embed_dim,
        batch_size=getattr(cfg.model, "batch_size", 0),
    )