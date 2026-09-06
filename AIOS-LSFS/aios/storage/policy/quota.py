import hashlib
import json
import threading
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path


DEFAULT_LIMITS = {
    "source_code": 100 * 1024 * 1024,
    "report": 50 * 1024 * 1024,
    "personal": 10 * 1024 * 1024,
}


class SemanticQuotaExceeded(ValueError):
    """A proposed file replacement would exceed its category quota."""


@lru_cache(maxsize=None)
def _storage_lock(namespace: str):
    # One kernel process may have multiple LSFS instances for the same root.
    return threading.RLock()


class SemanticQuota:
    """Track LSFS-managed file bytes; never scan or reset usage at startup."""

    def __init__(self, redis_client, root_dir: str, limits: dict[str, int] | None = None):
        self.redis = redis_client
        self.limits = dict(DEFAULT_LIMITS if limits is None else limits)
        for category, limit in self.limits.items():
            if not isinstance(category, str) or not category.strip():
                raise ValueError("Quota categories must be non-empty strings")
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
                raise ValueError("Quota limits must be non-negative integer byte counts")

        root = str(Path(root_dir).resolve())
        root_id = hashlib.sha256(root.encode("utf-8")).hexdigest()[:16]
        self.namespace = f"semantic_quota:{root_id}"
        self.files_key = f"{self.namespace}:files"
        self._lock = _storage_lock(self.namespace)

    def usage_key(self, agent_name: str) -> str:
        return f"{self.namespace}:usage:{agent_name}"

    def get_usage(self, agent_name: str) -> dict[str, int]:
        with self._lock:
            return {
                category: int(size)
                for category, size in self.redis.hgetall(self.usage_key(agent_name)).items()
            }

    def get_file_metadata(self, file_path: str) -> dict:
        with self._lock:
            value = self.redis.hget(self.files_key, str(Path(file_path).resolve()))
            return json.loads(value) if value else {}

    def can_replace(
        self, agent_name: str, old_metadata: dict, new_category: str, new_size: int
    ) -> bool:
        with self._lock:
            current = int(self.redis.hget(self.usage_key(agent_name), new_category) or 0)
            if (
                old_metadata.get("owner") == agent_name
                and old_metadata.get("category") == new_category
            ):
                current -= old_metadata["size_bytes"]
            limit = self.limits.get(new_category)
            return limit is None or current + new_size <= limit

    def _replace_record(self, file_path: str, old_metadata: dict, new_metadata: dict):
        # Used under the storage lock, including compensation after failed I/O.
        deltas = {}
        for record, sign in ((old_metadata, -1), (new_metadata, 1)):
            if record:
                key = (self.usage_key(record["owner"]), record["category"])
                deltas[key] = deltas.get(key, 0) + sign * record["size_bytes"]

        with self.redis.pipeline(transaction=True) as pipe:
            for (key, category), delta in deltas.items():
                if delta:
                    pipe.hincrby(key, category, delta)
            if new_metadata:
                pipe.hset(self.files_key, file_path, json.dumps(new_metadata))
            else:
                pipe.hdel(self.files_key, file_path)
            pipe.execute()

    @contextmanager
    def reserve_replace(
        self, agent_name: str, file_path: str, new_category: str, new_size: int
    ):
        """Reserve space before I/O and refund it if the write fails."""
        if not isinstance(agent_name, str) or not agent_name:
            raise ValueError("Quota requires an agent name")
        if not isinstance(new_category, str) or not new_category:
            raise ValueError("Quota requires a category")
        if isinstance(new_size, bool) or not isinstance(new_size, int) or new_size < 0:
            raise ValueError("Quota requires a non-negative byte count")

        file_path = str(Path(file_path).resolve())
        with self._lock:
            old_metadata = self.get_file_metadata(file_path)
            if not self.can_replace(agent_name, old_metadata, new_category, new_size):
                raise SemanticQuotaExceeded(
                    f"Semantic quota exceeded: {new_category} "
                    f"(limit={self.limits[new_category]} bytes)"
                )

            new_metadata = {
                "owner": agent_name,
                "category": new_category,
                "size_bytes": new_size,
            }
            self._replace_record(file_path, old_metadata, new_metadata)
            try:
                yield
            except BaseException:
                self._replace_record(file_path, new_metadata, old_metadata)
                raise

    @contextmanager
    def reserve_delete(self, file_path: str):
        """Remove a tracked allocation, restoring it when deletion fails."""
        file_path = str(Path(file_path).resolve())
        with self._lock:
            old_metadata = self.get_file_metadata(file_path)
            if old_metadata:
                self._replace_record(file_path, old_metadata, {})
            try:
                yield
            except BaseException:
                if old_metadata:
                    self._replace_record(file_path, {}, old_metadata)
                raise
