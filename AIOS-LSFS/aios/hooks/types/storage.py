from pydantic import BaseModel
from typing import Any, TypeAlias, Callable
from queue import Queue

StorageRequestQueue: TypeAlias = Queue
# StorageRequestQueue: TypeAlias = []

StorageRequestQueueGetMessage: TypeAlias = Callable[[], None]
StorageRequestQueueAddMessage: TypeAlias = Callable[[str], None]
StorageRequestQueueCheckEmpty: TypeAlias = Callable[[], bool]

class StorageManagerParams(BaseModel):
    root_dir: str
    db_dir: str
    use_vector_db: bool = False
    semantic_quota: dict[str, Any] | None = None
