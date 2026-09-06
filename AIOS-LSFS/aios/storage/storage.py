import os

import pickle

import zlib

from .filesystem.lsfs import LSFS

from cerebrum.storage.apis import StorageResponse

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

    def address_request(self, agent_request):
        result = self.filesystem.address_request(agent_request)
        return StorageResponse(
            response_message=result,
            finished=True
        )
