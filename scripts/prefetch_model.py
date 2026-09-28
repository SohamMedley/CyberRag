"""Download the embedding model during the build (Render ``buildCommand``).

Running this from ``render.yaml``'s build step means the sentence-transformer
weights (~90 MB) are already on disk before the web process boots, so the
first ``/index`` or ``/ask`` request does not pay the download and the
background warmup thread (``WARMUP_EMBEDDINGS=true``) finishes instantly.

Only ``huggingface_hub`` is used (no torch import) so the build stays well
inside the 512 MB memory limit of the Render free plan. The cache location
follows ``HF_HOME``, which is set in ``render.yaml`` to a directory inside the
project checkout - the same filesystem the runtime reads.

Exit codes:
    0 - model is cached and ready
    1 - download failed (fails the build loudly rather than deferring the
        download to the first user request)
"""

from __future__ import annotations

import os
import sys

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Keep the download to what CPU runtime actually loads (safetensors/pytorch +
# configs + tokenizer); TF/Flax/ONNX duplicates would triple the build size.
IGNORE_PATTERNS = ["*.h5", "*.msgpack", "*.tflite", "*.pb", "*.onnx", "*.ot"]


def main() -> int:
    model_name = os.environ.get("EMBEDDING_MODEL_NAME") or DEFAULT_MODEL
    cache_dir = os.environ.get("HF_HOME")
    print(f"Prefetching embedding model {model_name!r} ...", flush=True)
    if cache_dir:
        print(f"HF_HOME={cache_dir}", flush=True)
        os.makedirs(cache_dir, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(repo_id=model_name, ignore_patterns=IGNORE_PATTERNS)
    except Exception as exc:
        print(f"ERROR: could not prefetch {model_name}: {exc}", file=sys.stderr)
        return 1

    print(f"Model ready at {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
