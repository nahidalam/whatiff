"""Atomic local checkpoints and immutable, content-addressed image artifacts."""

import hashlib
import json
import re
from pathlib import Path
from uuid import uuid4


class RunStore:
    def __init__(self, root):
        self.root = Path(root)

    def folder(self, run_id):
        if not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise ValueError("Invalid run ID.")
        return self.root / run_id

    def save(self, run):
        folder = self.folder(run["id"])
        folder.mkdir(parents=True, exist_ok=True)
        temporary = folder / (uuid4().hex + ".tmp")
        temporary.write_text(json.dumps(run, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(folder / "run.json")

    def load(self, run_id):
        return json.loads((self.folder(run_id) / "run.json").read_text(encoding="utf-8"))

    def image(self, run_id, content, media_type):
        if not content or len(content) > 20 * 1024 * 1024:
            raise ValueError("Images must contain 1 byte to 20 MB.")
        extensions = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}
        if media_type not in extensions:
            raise ValueError("Unsupported image media type.")
        digest = hashlib.sha256(content).hexdigest()
        relative = "images/" + digest + "." + extensions[media_type]
        path = self.folder(run_id) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_name(uuid4().hex + ".tmp")
            temporary.write_bytes(content)
            temporary.replace(path)
        return {"path": relative, "sha256": digest, "media_type": media_type}

    def read_image(self, run_id, artifact):
        # Reconstruct the path rather than trusting a saved relative path.
        digest = artifact["sha256"]
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Invalid artifact digest.")
        suffix = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}[artifact["media_type"]]
        content = (self.folder(run_id) / "images" / (digest + "." + suffix)).read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("Image integrity check failed.")
        return content
