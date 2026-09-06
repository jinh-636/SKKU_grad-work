import hashlib
import json
import os
import stat
import threading
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from .analyzer import SemanticAnalyzer


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
    """Track file bytes and rebuild persisted usage from the storage root."""

    def __init__(self, redis_client, root_dir: str, limits: dict[str, int] | None = None):
        self.redis = redis_client
        self.limits = dict(DEFAULT_LIMITS if limits is None else limits)
        for category, limit in self.limits.items():
            if not isinstance(category, str) or not category.strip():
                raise ValueError("Quota categories must be non-empty strings")
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
                raise ValueError("Quota limits must be non-negative integer byte counts")

        self.root_dir = Path(root_dir).resolve()
        root = str(self.root_dir)
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

    def get_status(self, agent_name: str = "terminal", categories=()) -> dict:
        """Read current usage and active limits without scanning or inference."""
        if not isinstance(agent_name, str) or not agent_name.strip():
            raise ValueError("Quota status requires an agent name")
        with self._lock:
            usage = self.get_usage(agent_name)
            rows = []
            for category in sorted(set(categories) | set(self.limits) | set(usage)):
                used = usage.get(category, 0)
                limit = self.limits.get(category)
                if limit is None:
                    status = "unlimited"
                elif used > limit:
                    status = "exceeded"
                elif used == limit:
                    status = "full"
                else:
                    status = "available"
                rows.append({
                    "category": category,
                    "used_bytes": used,
                    "limit_bytes": limit,
                    "remaining_bytes": None if limit is None else max(0, limit - used),
                    "status": status,
                })
            return {
                "enabled": True,
                "root_dir": str(self.root_dir),
                "agent_name": agent_name,
                "total_used_bytes": sum(usage.values()),
                "categories": rows,
            }

    def get_file_metadata(self, file_path: str) -> dict:
        with self._lock:
            value = self.redis.hget(self.files_key, str(Path(file_path).resolve()))
            return json.loads(value) if value else {}

    def _scan_files(self, excluded_dirs: set[Path]):
        """Yield regular files only; never follow links or count internal data."""
        def raise_walk_error(error):
            raise error

        for directory, subdirs, names in os.walk(
            self.root_dir, topdown=True, onerror=raise_walk_error, followlinks=False
        ):
            parent = Path(directory)
            subdirs[:] = sorted(
                name for name in subdirs
                if not (parent / name).is_symlink()
                and parent / name not in excluded_dirs
            )
            for name in sorted(names):
                if name.startswith(".lsfs-quota-"):
                    continue
                path = parent / name
                info = path.lstat()
                if stat.S_ISREG(info.st_mode):
                    yield path, info

    @staticmethod
    def _file_signature(info):
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    def rebuild_usage(
        self, analyzer: SemanticAnalyzer, default_owner: str = "terminal", excluded_dirs=()
    ) -> dict[str, int]:
        """Rebuild bytes, inferring a category only when no stored category exists."""
        if not isinstance(default_owner, str) or not default_owner.strip():
            raise ValueError("Quota scan requires a default owner")
        if not self.root_dir.is_dir():
            raise NotADirectoryError(str(self.root_dir))
        excluded = {Path(directory).resolve() for directory in excluded_dirs}
        if any(directory == self.root_dir or directory in self.root_dir.parents
               for directory in excluded):
            raise ValueError("Quota scan cannot exclude the storage root itself")

        # Startup runs before this instance accepts requests. The shared lock
        # also prevents other instances in this process from changing usage.
        with self._lock:
            previous = self.redis.hgetall(self.files_key)
            records, usage, signatures = {}, {}, {}
            files_analyzed = metadata_reused = other_fallback_files = total_bytes = 0
            for path, info in self._scan_files(excluded):
                key = str(path)
                old_record = json.loads(previous[key]) if key in previous else {}
                owner = old_record.get("owner", default_owner)
                if not isinstance(owner, str) or not owner.strip():
                    raise ValueError(f"Stored quota owner is invalid: {path}")
                if "category" in old_record:
                    category = old_record["category"]
                    if not isinstance(category, str) or not category.strip():
                        raise ValueError(f"Stored quota category is invalid: {path}")
                    metadata_reused += 1
                else:
                    try:
                        raw = path.read_bytes()
                        if len(raw) != info.st_size or (
                            self._file_signature(info) != self._file_signature(path.lstat())
                        ):
                            raise RuntimeError("File changed while being read")
                        try:
                            content = raw.decode("utf-8")
                        except UnicodeDecodeError:
                            content = None
                        if not content or "\x00" in content:
                            category = "other"
                            other_fallback_files += 1
                        else:
                            category = analyzer.analyze(content).category
                            files_analyzed += 1
                    except Exception as error:
                        raise RuntimeError(
                            f"Failed to classify quota file '{path}' ({type(error).__name__})"
                        ) from error
                size = info.st_size
                signatures[path] = self._file_signature(info)
                records[key] = json.dumps({
                    "owner": owner, "category": category, "size_bytes": size,
                })
                totals = usage.setdefault(owner, {})
                totals[category] = totals.get(category, 0) + size
                total_bytes += size

            # Reject a stale snapshot if files changed before publication.
            current = {
                path: self._file_signature(info)
                for path, info in self._scan_files(excluded)
            }
            if current != signatures:
                raise RuntimeError("Storage files changed during quota scan; restart to retry")

            usage_keys = set(self.redis.scan_iter(match=f"{self.namespace}:usage:*"))
            with self.redis.pipeline(transaction=True) as pipe:
                pipe.delete(self.files_key, *usage_keys)
                if records:
                    pipe.hset(self.files_key, mapping=records)
                for owner, totals in usage.items():
                    pipe.hset(self.usage_key(owner), mapping=totals)
                pipe.execute()

            return {
                "files_scanned": len(records),
                "files_analyzed": files_analyzed,
                "metadata_reused": metadata_reused,
                "other_fallback_files": other_fallback_files,
                "total_bytes": total_bytes,
            }

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
