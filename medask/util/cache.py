import json
from typing import Any, Dict, Tuple

from medask.util.log import get_logger

logger = get_logger("medask.util.cache")


class FileCache:
    """
    Filecache dumping self._cache dict as JSON to <abs_path> with backup to .backup.
    """

    def __init__(self, abs_path: str) -> None:
        self._path = abs_path
        self._path_to_backup = f"{abs_path}.backup"
        self._cache: Dict[str, Any] = self._load() or {}
        self._tainted = False

    def _load(self) -> Dict[str, Any]:
        """Load cache from disk."""
        try:
            with open(self._path) as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"FileCache read error {e}")
            return {}
        except FileNotFoundError:
            return {}

    def _dump(self) -> None:
        """
        Dump cache to disk after each write.
        If the write fails the cache gets tainted and no longer dumped to disk. This
            preserves the backup pristine.
        """
        if self._tainted:
            logger.warning(f"Tainted so skipping dumping {len(self._cache)}")
            return
        try:
            for path in (self._path, self._path_to_backup):
                with open(path, "w") as f:
                    json.dump(self._cache, f)
        except Exception as e:
            logger.exception(f"Error dumping to disk: {e}")
            self._tainted = True

    def add(self, items: Dict[str, Any], overwrite: bool = False) -> None:
        """Add <items> to the cache."""
        for k, v in items.items():
            k = str(k)
            if overwrite or k not in self._cache:
                self._cache[k] = v
        self._dump()

    def has_key(self, key: str) -> Tuple[bool, Any]:
        """If the cache contains <key>, return True and the corresponding value."""
        key = str(key)
        if key in self._cache:
            return True, self._cache[key]
        else:
            return False, None
