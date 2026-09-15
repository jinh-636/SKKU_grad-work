"""Find existing semantic duplicates and apply a user-confirmed deletion plan."""
import hashlib
import os
import stat
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import numpy as np
from llama_index.core import SimpleDirectoryReader

from .document_embedding import EMBEDDING_VERSION


SIMILARITY_THRESHOLD = 0.95
DOCUMENT_SUFFIXES = {".hwp", ".pdf", ".docx", ".pptx", ".ppt", ".pptm", ".epub"}


class SemanticDeduplicator:
    def __init__(self, filesystem):
        self.filesystem = filesystem
        self.root = Path(filesystem.root_dir).resolve()
        self.database_dir = Path(filesystem.db_dir).resolve()

    def _snapshot(self, file_path):
        path = Path(file_path)
        if (not path.is_absolute() or path != path.resolve()
                or not path.is_relative_to(self.root)
                or path.is_relative_to(self.database_dir)
                or path.name.startswith(".lsfs-quota-")):
            raise ValueError(f"File is outside the cleanup scope or is a link: {path}")
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Not a regular file: {path}")
        raw = path.read_bytes()
        after = path.lstat()
        def signature(info):
            return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]
        if signature(before) != signature(after) or len(raw) != before.st_size:
            raise ValueError(f"File changed while being read: {path}")
        return raw, {"file_path": str(path), "size_bytes": len(raw),
                     "signature": signature(after), "sha256": hashlib.sha256(raw).hexdigest(),
                     "modified_at": datetime.fromtimestamp(after.st_mtime).isoformat()}

    def _owners(self):
        # Quota is authoritative when enabled. Without it, use collection ownership.
        if self.filesystem.semantic_quota is not None:
            return None
        owners = {}
        for collection in self.filesystem.vector_db.client.list_collections():
            collection = self.filesystem.vector_db.client.get_collection(collection.name)
            for metadata in collection.get(include=["metadatas"])["metadatas"]:
                if metadata and metadata.get("file_path"):
                    key = str(Path(metadata["file_path"]).resolve())
                    owners.setdefault(key, set()).add(collection.name)
        return owners

    def _is_owned(self, path, agent_name, owners):
        quota = self.filesystem.semantic_quota
        if quota is not None:
            return quota.get_file_metadata(str(path)).get("owner", "terminal") == agent_name
        return owners.get(str(path), {"terminal"}) == {agent_name}

    def scan(self, agent_name):
        if not self.filesystem.use_vector_db:
            raise ValueError("Semantic deduplication requires the vector database")
        database = self.filesystem.vector_db
        collection = database.add_or_get_collection(agent_name)
        stored = collection.get(include=["documents", "metadatas", "embeddings"])
        cached = {key: index for index, key in enumerate(stored["ids"])}
        owners = self._owners()
        files, vectors, skipped = [], [], []
        def walk_error(error):
            raise error
        for directory, subdirs, names in os.walk(self.root, onerror=walk_error, followlinks=False):
            parent = Path(directory)
            subdirs[:] = sorted(name for name in subdirs
                                if not (parent / name).is_symlink()
                                and not (parent / name).is_relative_to(self.database_dir))
            for name in sorted(names):
                path = parent / name
                if path.is_symlink() or name.startswith(".lsfs-quota-"):
                    continue
                if not self._is_owned(path, agent_name, owners):
                    continue
                try:
                    with self.filesystem.get_file_lock(str(path)):
                        raw, record = self._snapshot(path)
                        if path.suffix.lower() in DOCUMENT_SUFFIXES:
                            documents = SimpleDirectoryReader(input_files=[str(path)]).load_data()
                            content = " ".join(doc.text for doc in documents)
                        else:
                            content = raw.decode("utf-8")
                        if not content.strip() or "\x00" in content:
                            raise ValueError("No non-empty text to compare")
                        if self._snapshot(path)[1] != record:
                            raise ValueError("File changed during text extraction")
                        key = hashlib.md5(str(path).encode()).hexdigest()
                        index = cached.get(key)
                        metadata = (stored["metadatas"][index] or {}) if index is not None else {}
                        if (index is not None and metadata.get("embedding_version") == EMBEDDING_VERSION
                                and stored["documents"][index] == content):
                            vector = stored["embeddings"][index]
                        else:
                            if not database.update_document(str(path), content, agent_name):
                                raise ValueError("Could not update the document embedding")
                            vector = collection.get(ids=[key], include=["embeddings"])["embeddings"][0]
                        vector = np.asarray(vector, dtype=np.float64)
                        norm = np.linalg.norm(vector)
                        if not np.isfinite(norm) or norm == 0:
                            raise ValueError("Invalid document embedding")
                        record.update(id=f"f{len(files) + 1}", preview=content[:300])
                        files.append(record)
                        vectors.append(vector / norm)
                except (OSError, ValueError, ImportError) as error:
                    skipped.append({"file_path": str(path), "reason": str(error)})
        # Keep every threshold-matching pair. Connected groups are for display;
        # the user selects the exact files to delete and confirms the resulting list.
        pairs, neighbors = [], {index: set() for index in range(len(files))}
        if vectors:
            matrix = np.asarray(vectors)
            for left, vector in enumerate(matrix):
                for right in np.flatnonzero(matrix[left + 1:] @ vector >= SIMILARITY_THRESHOLD) + left + 1:
                    right = int(right)
                    score = float(np.clip(np.dot(vector, matrix[right]), -1, 1))
                    pairs.append({"left": files[left]["id"], "right": files[right]["id"], "similarity": score})
                    neighbors[left].add(right)
                    neighbors[right].add(left)
        groups, visited = [], set()
        for index in neighbors:
            if index in visited or not neighbors[index]:
                continue
            pending, members = [index], set()
            while pending:
                member = pending.pop()
                if member in members:
                    continue
                members.add(member)
                pending.extend(neighbors[member] - members)
            visited.update(members)
            group_files = [files[member] for member in sorted(members)]
            ids = {item["id"] for item in group_files}
            groups.append({"id": f"g{len(groups) + 1}", "files": group_files,
                           "pairs": [pair for pair in pairs if pair["left"] in ids and pair["right"] in ids]})
        return {"root_dir": str(self.root), "agent_name": agent_name,
                "threshold": SIMILARITY_THRESHOLD, "scanned_count": len(files),
                "groups": groups, "skipped": skipped}

    def apply(self, plan, agent_name):
        if plan["root_dir"] != str(self.root) or plan["agent_name"] != agent_name:
            raise ValueError("The cleanup plan belongs to a different root or user")
        records = {item["file_path"]: item for item in plan["retained"]}
        for deletion in plan["deletions"]:
            records[deletion["file"]["file_path"]] = deletion["file"]
        results = []
        with ExitStack() as locks:
            for path in sorted(records):
                lock = self.filesystem.get_file_lock(path)
                if not lock.acquire(timeout=10):
                    raise ValueError(f"Timeout waiting for lock on {path}")
                locks.callback(lock.release)
            owners = self._owners()
            def validate(record):
                path = record["file_path"]
                _, current = self._snapshot(path)
                if any(current[key] != record[key] for key in ("signature", "sha256")):
                    raise ValueError(f"File changed; run deduplicate_files again: {path}")
                if not self._is_owned(Path(path), agent_name, owners):
                    raise ValueError(f"File ownership changed; run deduplicate_files again: {path}")
            # Validate the whole plan before the first deletion, including kept files.
            for record in records.values():
                validate(record)
            for deletion in plan["deletions"]:
                path = deletion["file"]["file_path"]
                try:
                    validate(deletion["file"])
                    results.append(self.filesystem._delete_file_unlocked(path, agent_name))
                except Exception as error:
                    results.append(f"Cleanup stopped at {path}: {error}")
                    raise ValueError("\n".join(results)) from error
        return "\n".join(results) or "No files selected for deletion."
