import hashlib
import json
from pathlib import Path
from typing import Dict

class IncrementalTracker:
    def __init__(self, manifest_path: str = "data/processed/manifest.json"):
         self.manifest_path = Path(manifest_path)
         self.manifest: Dict[str, str] = self.load_manifest()

    def compute_hash(self, file_path: str) -> str:
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    
    def load_manifest(self) -> Dict[str, str]:
        if self.manifest_path.exists():
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    
    def save_manifest(self):
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=4)

    def is_file_modified(self, file_path: str) -> bool:
        current_hash = self.compute_hash(file_path)
        old_hash = self.manifest.get(file_path)
        
        if old_hash != current_hash:
            self.manifest[file_path] = current_hash
            return True
            
        return False