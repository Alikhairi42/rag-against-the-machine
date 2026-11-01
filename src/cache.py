import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


class QueryCache:
    """Cache for storing and retrieving query results."""

    def __init__(
        self,
        cache_file: str = "data/processed/query_cache.json",
    ) -> None:
        """Initialise the query cache.

        Args:
            cache_file: Path to the JSON file used for
                persistent cache storage.
        """
        self.cache_file: Path = Path(cache_file)
        self.cache: Dict[str, Any] = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        """Load cached data from the JSON file on disk.

        Returns:
            Dictionary of cached query results, or an empty
            dict if the file is missing or corrupt.
        """
        if self.cache_file.exists():
            try:
                with open(
                    self.cache_file, "r", encoding="utf-8"
                ) as f:
                    content = f.read().strip()
                    if content:
                        return json.loads(content)
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        """Persist the current cache dictionary to disk."""
        self.cache_file.parent.mkdir(
            parents=True, exist_ok=True
        )
        with open(
            self.cache_file, "w", encoding="utf-8"
        ) as f:
            json.dump(self.cache, f, indent=2)

    def _make_key(
        self, query: str, k: int, task: str
    ) -> str:
        """Generate a unique cache key for a query.

        Args:
            query: The search query string.
            k: Number of results requested.
            task: The task type identifier.

        Returns:
            An MD5 hex-digest string used as the cache key.
        """
        raw = f"{task}:{query.strip().lower()}:{k}"
        return hashlib.md5(
            raw.encode("utf-8")
        ).hexdigest()

    def get(
        self,
        query: str,
        k: int,
        task: str = "search",
    ) -> Optional[Any]:
        """Retrieve a cached result for the given query.

        Args:
            query: The search query string.
            k: Number of results requested.
            task: The task type identifier.

        Returns:
            The cached result, or ``None`` if not found.
        """
        key = self._make_key(query, k, task)
        return self.cache.get(key)

    def set(
        self,
        query: str,
        k: int,
        result: Any,
        task: str = "search",
    ) -> None:
        """Store a query result in the cache.

        Args:
            query: The search query string.
            k: Number of results requested.
            result: The result object to cache.
            task: The task type identifier.
        """
        key = self._make_key(query, k, task)
        self.cache[key] = result
        self._save_cache()
