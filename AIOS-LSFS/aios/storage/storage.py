import os
from pathlib import Path

import pickle

import zlib

from .filesystem.lsfs import LSFS

from cerebrum.storage.apis import StorageResponse
from .deduplication import ApplyDeduplicationQuery, DeduplicationResponse

class StorageManager:
    def __init__(self, root_dir, db_dir, use_vector_db=True, filesystem_type="lsfs", semantic_quota=None):
        self.use_vector_db = use_vector_db
        self.filesystem_type = filesystem_type
        self.root_dir = root_dir
        self.db_dir = db_dir
        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(self.db_dir, exist_ok=True)
        if filesystem_type == "lsfs":
            self.filesystem = LSFS(root_dir, db_dir, use_vector_db, semantic_quota=semantic_quota)

    def get_quota_status(self, agent_name: str = "terminal") -> dict:
        if not isinstance(agent_name, str) or not agent_name.strip():
            raise ValueError("Quota status requires an agent name")
        quota = self.filesystem.semantic_quota
        if quota is None:
            return {
                "enabled": False,
                "root_dir": str(Path(self.root_dir).resolve()),
                "agent_name": agent_name,
            }
        return quota.get_status(agent_name, self.filesystem.semantic_analyzer.categories)

    def address_request(self, agent_request):
        if isinstance(agent_request.query, ApplyDeduplicationQuery):
            try:
                result = self.filesystem.sto_apply_deduplication(
                    agent_request.query.plan, agent_request.agent_name
                )
            except Exception as error:
                return StorageResponse(response_message=f"Cleanup stopped: {error}",
                                       error=str(error), finished=True)
        else:
            result = self.filesystem.address_request(agent_request)
        if agent_request.query.operation_type == "deduplicate_files" and isinstance(result, dict):
            return DeduplicationResponse(
                response_message=f"Found {len(result['groups'])} duplicate candidate groups.",
                deduplication=result, finished=True,
            )
        return StorageResponse(
            response_message=result,
            finished=True
        )
