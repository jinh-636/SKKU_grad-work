import os
import pickle
import zlib
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import redis
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Set, Optional
import hashlib
import threading
from urllib.parse import urljoin
from pathlib import Path
import uuid
import requests
import stat
import tempfile
from contextlib import nullcontext

from .vector_db import ChromaDB
from ..policy import SemanticAnalysisError, SemanticAnalyzer
from ..policy.quota import SemanticQuota, SemanticQuotaExceeded

import logging

logging.getLogger('watchdog').setLevel(logging.ERROR)

class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, lsfs_instance):
        self.lsfs = lsfs_instance
        
    def on_modified(self, event):
        if not event.is_directory:
            self.lsfs.handle_file_change(event.src_path, "modified")
            
    def on_created(self, event):
        if not event.is_directory:
            self.lsfs.handle_file_change(event.src_path, "created")
            
    def on_deleted(self, event):
        if not event.is_directory:
            self.lsfs.handle_file_change(event.src_path, "deleted")

    def on_moved(self, event):
        # Atomic quota writes rename a temporary file to the final path.
        if not event.is_directory and os.path.basename(event.src_path).startswith(".lsfs-quota-"):
            self.lsfs.handle_file_change(event.dest_path, "modified")

class LSFS:
    def __init__(self, root_dir, db_dir, use_vector_db=True, max_versions=20, semantic_quota=None):
        self.root_dir = root_dir
        self.db_dir = db_dir
        self.use_vector_db = use_vector_db
        self.max_versions = max_versions
        self.vector_db = ChromaDB(db_dir=self.db_dir)
        
        # Initialize Redis connection
        self.redis_client = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=True
        )
        
        # # Test Redis connection
        try:
            self.redis_client.ping()
            print("Successfully connected to Redis")
            self.use_redis = True
            
        except redis.ConnectionError as e:
            print(f"Failed to connect to Redis: {e}")
            self.use_redis = False
        
        # Initialize policy and locks before the observer can receive events.
        self.file_locks = {}
        self.locks_lock = threading.Lock()
        quota_options = {} if semantic_quota is None else semantic_quota
        if not isinstance(quota_options, dict):
            raise ValueError("semantic_quota must be a mapping")
        enabled = quota_options.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("semantic_quota.enabled must be a boolean")
        self.semantic_quota = None
        self.semantic_analyzer = None
        if enabled:
            if not self.use_redis:
                raise RuntimeError("Semantic quota requires a running Redis server")
            self.semantic_quota = SemanticQuota(
                self.redis_client, self.root_dir, quota_options.get("limits")
            )
            self.semantic_analyzer = SemanticAnalyzer.from_config()
        # No directory scan or usage reset is performed here.

        # Initialize file system observer
        self.observer = Observer()
        self.event_handler = FileChangeHandler(self)
        self.observer.schedule(self.event_handler, self.root_dir, recursive=True)
        self.observer.start() # temporarily disabled
        
        
        
    def __del__(self):
        if hasattr(self, 'observer'):
            self.observer.stop()
            self.observer.join()
            
    def get_file_hash(self, file_path: str) -> str:
        return hashlib.sha256(file_path.encode()).hexdigest()
            
    def get_file_lock(self, file_path: str) -> threading.Lock:
        with self.locks_lock:
            if file_path not in self.file_locks:
                self.file_locks[file_path] = threading.Lock()
            return self.file_locks[file_path]

    def resolve_path(self, raw_path: str) -> str:
        if raw_path is None or not raw_path.strip():
            raise ValueError("File path is required")

        root = Path(self.root_dir).resolve()
        relative_path = Path(raw_path.lstrip("/"))

        # Prevent root/root/test.txt
        if relative_path.parts and relative_path.parts[0] == root.name:
            relative_path = Path(*relative_path.parts[1:])

        path = (root / relative_path).resolve()
        return str(path)
    
    def handle_file_change(self, file_path: str, change_type: str):
        if os.path.basename(file_path).startswith(".lsfs-quota-"):
            return
        # """Handle file changes with proper lock management."""
        lock = self.get_file_lock(file_path)
        try:
            if lock.acquire(timeout=5):  # Add timeout to prevent deadlocks
                try:
                    # relative_path = os.path.relpath(file_path, self.root_dir)
                    file_hash = self.get_file_hash(file_path)
                    
                    if change_type in ["modified", "created"]:
                        with open(file_path, 'r') as f:
                            content = f.read()
                        
                        # Update vector DB
                        if self.use_vector_db:
                            owner = "terminal"
                            if self.semantic_quota is not None:
                                owner = self.semantic_quota.get_file_metadata(file_path).get("owner", owner)
                            self.vector_db.update_document(file_path, content, owner)
                            
                        # Update Redis cache with version history
                        timestamp = datetime.now().isoformat()
                        
                        version_info = {
                            'content': content,
                            'timestamp': timestamp,
                            'hash': file_hash,
                            'change_type': change_type
                        }
                        
                        # versions_key = f"file_versions:{relative_path}"
                        versions_key = file_hash
                        versions = self.redis_client.lrange(versions_key, 0, -1)
                        versions = [json.loads(v) for v in versions]
                        
                        # Add new version
                        self.redis_client.lpush(versions_key, json.dumps(version_info))
                        
                        # Trim to max versions
                        if len(versions) >= self.max_versions:
                            self.redis_client.ltrim(versions_key, 0, self.max_versions - 1)
                            
                    elif change_type == "deleted":
                        # Remove from vector DB
                        if self.use_vector_db:
                            self.vector_db.delete_document(file_path)
                        
                        # Add deletion record to Redis
                        # versions_key = f"file_versions:{relative_path}"
                        versions_key = file_hash
                        deletion_info = {
                            'timestamp': datetime.now().isoformat(),
                            'change_type': 'deleted'
                        }
                        self.redis_client.lpush(versions_key, json.dumps(deletion_info))
                        
                finally:
                    lock.release()  # Ensure lock is always released
            else:
                print(f"Timeout waiting for lock on {file_path}")
        except Exception as e:
            print(f"Error handling file change: {str(e)}")
            
    def get_file_history(self, file_path: str, limit: int = None) -> list:
        # relative_path = os.path.relpath(file_path, self.root_dir)
        # versions_key = f"file_versions:{relative_path}"
        
        file_hash = self.get_file_hash(file_path)
        versions_key = file_hash
        
        limit = limit or self.max_versions
        versions = self.redis_client.lrange(versions_key, 0, limit - 1)
        return [json.loads(v) for v in versions]
        
    def restore_version(self, file_path: str, version_index: int, collection_name: str = None) -> bool:
        # file_lock = self.get_file_lock(file_path)
        
        # with file_lock:
        try:
            # relative_path = os.path.relpath(file_path, self.root_dir)
            # versions_key = f"file_versions:{relative_path}"
            
            file_hash = self.get_file_hash(file_path)
            versions_key = file_hash
            
            # Get specified version
            version_data = self.redis_client.lindex(versions_key, version_index)
            if not version_data:
                return False
                
            version_info = json.loads(version_data)
            if 'content' not in version_info:
                return False
            
            self._write_with_quota(file_path, version_info['content'], collection_name)
                
            # Update vector DB
            # if self.use_vector_db:
            #     self.vector_db.update_document(file_path, version_info['content'])
                
            return True
            
        except Exception as e:
            if isinstance(e, (SemanticQuotaExceeded, SemanticAnalysisError)):
                raise
            print(f"Error restoring version: {str(e)}")
            return False

    def address_request(self, agent_request):
        collection_name = agent_request.agent_name
        operation_type = agent_request.query.operation_type
        
        path = None
        if operation_type in ["create_file", "delete_file", "write", "rollback", "share"]:
            path = self.resolve_path(agent_request.query.params.get("file_path", None))
        elif operation_type in ["create_dir", "delete_dir"]:
            path = self.resolve_path(agent_request.query.params.get("dir_path", None))

        try:
            if operation_type == "mount":
                root = agent_request.query.params.get("root", self.root_dir)
                result = self.sto_mount(
                    collection_name=collection_name,
                    root_dir=root
                )
            
            elif operation_type == "create_file":
                result = self.sto_create_file(
                    file_path=path,
                    collection_name=collection_name
                )

            elif operation_type == "create_dir":
                result = self.sto_create_directory(
                    dir_path=path,
                    collection_name=collection_name
                )

            elif operation_type == "delete_file":
                result = self.sto_delete_file(
                    file_path=path,
                    collection_name=collection_name
                )

            elif operation_type == "delete_dir":
                result = self.sto_delete_directory(dir_path=path)
                
            elif operation_type == "write":
                # file_name = agent_request.query.params.get("file_name", None)
                content = agent_request.query.params.get("content", None)
                # breakpoint()
                result = self.sto_write(
                    file_name=None,
                    file_path=path,
                    content=content,
                    collection_name=collection_name
                )

            elif operation_type == "retrieve":
                query_text = agent_request.query.params.get("query_text", None)
                k = agent_request.query.params.get("k", "3")
                keywords = agent_request.query.params.get("keywords", None)
                result = self.sto_retrieve(
                    collection_name=collection_name,
                    query_text=query_text,
                    k=k,
                    keywords=keywords
                )
                
            elif operation_type == "rollback":
                n = agent_request.query.params.get("n", "1")
                time = agent_request.query.params.get("time", None)
                result = self.sto_rollback(
                    file_path=path,
                    n=int(n),
                    time=time,
                    collection_name=collection_name,
                )

            elif operation_type == "share":
                result = self.sto_share(
                    file_path=path,
                    collection_name=collection_name
                )
        
            else:
                result = f"Operation type: {operation_type} not supported"
        
        except Exception as e:
            result = f"Error handling file operation: {str(e)}"
        return result

    def sto_create_file(self, file_path: str, collection_name: str = None) -> str:
        try:
            print(f"Attempting to create file at: {file_path}")
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    pass  # Create empty file
                
                if self.use_vector_db:
                    self.vector_db.update_document(file_path, "", collection_name)
                return "File has been created successfully at: " + file_path
            return "File already exists at: " + file_path
        
        except Exception as e:
            return f"Error creating file: {str(e)}"
            
    def sto_create_directory(self, dir_path: str, collection_name: str = None) -> str:
        try:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                # if self.use_vector_db:
                #     self.vector_db.create_directory(dir_name, collection_name)
                return "Directory has been created successfully at: " + dir_path
            return "Directory already exists at: " + dir_path
        
        except Exception as e:
            return f"Error creating directory: {str(e)}"

    def sto_delete_file(self, file_path: str, collection_name: str = None) -> str:
        file_path = os.path.abspath(file_path)
        lock = self.get_file_lock(file_path)
        try:
            if not lock.acquire(timeout=10):
                return f"Timeout waiting for lock on {file_path}"
            try:
                if not os.path.exists(file_path):
                    return "File does not exist at: " + file_path
                if not os.path.isfile(file_path):
                    return "Path is not a file: " + file_path

                quota = self.semantic_quota
                allocation = (
                    quota.reserve_delete(file_path)
                    if quota is not None and not os.path.islink(file_path)
                    else nullcontext()
                )
                with allocation:
                    os.remove(file_path)
                if self.use_vector_db:
                    self.vector_db.delete_document(file_path, collection_name)

                return "File has been deleted successfully at: " + file_path
            finally:
                lock.release()
        except Exception as e:
            return f"Error deleting file: {str(e)}"

    def sto_delete_directory(self, dir_path: str) -> str:
        try:
            if not os.path.exists(dir_path):
                return "Directory does not exist at: " + dir_path
            if not os.path.isdir(dir_path):
                return "Path is not a directory: " + dir_path

            os.rmdir(dir_path)
            return "Directory has been deleted successfully at: " + dir_path

        except OSError as e:
            return f"Directory is not empty or cannot be deleted: {str(e)}"
        except Exception as e:
            return f"Error deleting directory: {str(e)}"
            
    def sto_mount(self, collection_name: str, root_dir: str) -> str:
        try:
            collection = self.vector_db.add_or_get_collection(collection_name)
            assert collection is not None, f"Collection {collection_name} not found"
            self.vector_db.build_database(root_dir)
            response = f"File system mounted successfully for agent: {collection_name}"
            return response
        
        except Exception as e:
            response = f"Error mounting file system: {str(e)}"
            return response
            
    def _atomic_write(self, file_path: str, content_bytes: bytes) -> None:
        """Replace content only after the complete temporary file is written."""
        descriptor, temporary = tempfile.mkstemp(
            prefix=".lsfs-quota-", dir=str(Path(file_path).parent)
        )
        try:
            with os.fdopen(descriptor, "wb") as file:
                file.write(content_bytes)
            if os.path.exists(file_path):
                os.chmod(temporary, stat.S_IMODE(os.stat(file_path).st_mode))
            os.replace(temporary, file_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _write_with_quota(self, file_path: str, content: str, collection_name=None) -> None:
        if not isinstance(content, str):
            raise TypeError("File content must be a string")
        file_path = str(Path(file_path).resolve())
        agent_name = collection_name or "terminal"
        quota = self.semantic_quota
        # LLM inference does not hold the file or accounting locks.
        profile = self.semantic_analyzer.analyze(content) if quota is not None else None
        content_bytes = content.encode("utf-8")

        lock = self.get_file_lock(file_path)
        if not lock.acquire(timeout=10):
            raise TimeoutError(f"Timeout waiting for lock on {file_path}")
        try:
            if quota is None:
                with open(file_path, "w", encoding="utf-8", newline="") as file:
                    file.write(content)
            else:
                with quota.reserve_replace(
                    agent_name, file_path, profile.category, len(content_bytes)
                ):
                    self._atomic_write(file_path, content_bytes)
        finally:
            lock.release()

    def sto_write(self, file_name: str, file_path: str, content: str, collection_name: str = None) -> str:
        try:
            self._write_with_quota(file_path, content, collection_name)
            return f"Content has been written to file: {file_path}"
        except Exception as e:
            return f"Error writing to file: {str(e)}"

    def sto_retrieve(self, collection_name: str, query_text: str, k: str = "3", keywords: str = None) -> list:
        try:
            collection = self.vector_db.add_or_get_collection(collection_name)
            return self.vector_db.retrieve(collection, query_text, k, keywords)
        
        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            return []
            
    def sto_rollback(self, file_path, n=1, time=None, collection_name=None) -> str:
        try:
            if not self.use_redis:
                return "Redis is not enabled. Please make sure the redis server has been installed and running."
            
            versions = self.get_file_history(file_path)
            
            if time:
                # Find version closest to specified time
                target_version = None
                min_time_diff = float('inf')
                
                for i, version in enumerate(versions):
                    version_time = datetime.fromisoformat(version['timestamp'])
                    target_time = datetime.fromisoformat(time)
                    time_diff = abs((version_time - target_time).total_seconds())
                    
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        target_version = i
                        
                if target_version is not None:
                    result = self.restore_version(file_path, target_version, collection_name)
                    if result:
                        return f"Successfully rolled back the file: {file_path} to its previous {n} version"
                    else:
                        return f"Failed to roll back the file: {file_path} to its previous {n} version"
            else:
                # Rollback n versions
                target_version = int(n)
                
                result = self.restore_version(file_path, target_version, collection_name)
                if result:
                    return f"Successfully rolled back the file: {file_path} to its previous {n} version"
                else:
                    return f"Failed to roll back the file: {file_path} to its previous {n} version"
                
            return result
        
        except Exception as e:
            result = f"Error rolling back file: {str(e)}"
            return result
            
    def generate_share_link(self, file_path: str) -> str:
        """Generate a publicly accessible link for sharing a file.
        
        Args:
            file_path: Path to the file to be shared
            
        Returns:
            str: A public URL where the file can be accessed
        """
        try:
            # First check if we already have a valid share link in Redis
            if not self.use_redis:
                return "Redis is not enabled. Please make sure the redis server has been installed and running."
            
            file_hash = self.get_file_hash(file_path)
            share_key = f"share:link:{file_hash}"
            
            existing_share = self.redis_client.hgetall(share_key)
            if existing_share and datetime.fromisoformat(existing_share['expires_at']) > datetime.now():
                return existing_share['share_link']

            # If no valid existing share, create new one
            with open(file_path, 'rb') as f:
                # Option 1: Using transfer.sh
                response = requests.put(
                    f'https://transfer.sh/{file_hash}',
                    data=f.read(),
                    headers={'Max-Days': '7'}
                )
                share_link = response.text.strip()
                
                # Option 2: Using 0x0.st (alternative)
                # response = requests.post(
                #     'https://0x0.st',
                #     files={'file': f}
                # )
                # share_link = response.text.strip()

            # Store share info in Redis
            share_info = {
                "file_path": file_path,
                "share_link": share_link,
                "created_at": datetime.now().isoformat(),
                "expires_at": (datetime.now() + timedelta(days=7)).isoformat(),
                "file_hash": file_hash
            }
            
            self.redis_client.hmset(share_key, share_info)
            self.redis_client.expire(share_key, 60 * 60 * 24 * 7)  # 7 days TTL

            return share_link

        except Exception as e:
            print(f"Error generating public share link: {str(e)}")
            return None

    def sto_share(self, file_path: str, collection_name: str = None) -> dict:
        """Share file with proper lock management."""
        lock = self.get_file_lock(file_path)
        try:
            if lock.acquire(timeout=10):  # Add timeout to prevent deadlocks
                try:
                    if not os.path.exists(file_path):
                        return {"error": "File not found"}
                        
                    share_link = self.generate_share_link(file_path)
                    if not share_link:
                        return {"error": "Failed to generate share link"}

                    return {
                        "file_name": os.path.basename(file_path),
                        "file_path": file_path,
                        "share_link": share_link,
                        "expires_in": "7 days",
                        "last_modified": datetime.fromtimestamp(
                            os.path.getmtime(file_path)
                        ).isoformat()
                    }
                finally:
                    lock.release()  # Ensure lock is always released
            else:
                return {"error": f"Timeout waiting for lock on {file_path}"}
        except Exception as e:
            return {"error": f"Error sharing file: {str(e)}"}
