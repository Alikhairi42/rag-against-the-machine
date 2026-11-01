import hashlib
import json
from pathlib import Path
from typing import Dict


class IncrementalTracker:
    """Track file modifications via MD5 hash manifest."""

    def __init__(
        self,
        manifest_path: str = "data/processed/manifest.json",
    ) -> None:
        """Initialise tracker with a manifest file path.

        Args:
            manifest_path: Path to the JSON manifest file.
        """
        self.manifest_path = Path(manifest_path)
        self.manifest: Dict[str, str] = self.load_manifest()

    def compute_hash(self, file_path: str) -> str:
        """Compute the MD5 hash of a file.

        Args:
            file_path: Path to the file to hash.

        Returns:
            Hex-digest string of the file's MD5 hash.
        """
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def load_manifest(self) -> Dict[str, str]:
        """Load the hash manifest from disk.

        Returns:
            Dictionary mapping file paths to their hashes.
        """
        if self.manifest_path.exists():
            try:
                with open(
                    self.manifest_path, "r", encoding="utf-8"
                ) as f:
                    content = f.read().strip()
                    if content:
                        return json.loads(content)
            except Exception:
                return {}
        return {}

    def save_manifest(self) -> None:
        """Persist the current hash manifest to disk."""
        self.manifest_path.parent.mkdir(
            parents=True, exist_ok=True
        )
        with open(
            self.manifest_path, "w", encoding="utf-8"
        ) as f:
            json.dump(self.manifest, f, indent=4)

    def is_file_modified(self, file_path: str) -> bool:
        """Check whether a file has been modified.

        Compares the current hash against the stored hash
        and updates the manifest when a change is detected.

        Args:
            file_path: Path to the file to check.

        Returns:
            True if the file is new or modified.
        """
        str_path = str(file_path)
        current_hash = self.compute_hash(str_path)
        old_hash = self.manifest.get(str_path)

        if old_hash != current_hash:
            self.manifest[file_path] = current_hash
            return True
        return False
