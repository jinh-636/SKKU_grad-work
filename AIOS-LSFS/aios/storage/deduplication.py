"""Internal messages and deletion plans for interactive deduplication."""
from cerebrum.storage.apis import StorageQuery, StorageResponse


class DeduplicationResponse(StorageResponse):
    deduplication: dict


class ApplyDeduplicationQuery(StorageQuery):
    """Created by the executor after confirmation, never from HTTP query params."""
    plan: dict


def build_deletion_plan(scan: dict, delete_ids: dict[str, list[str]]) -> dict:
    """Delete exactly the candidate IDs selected by the user, including a whole group."""
    groups = scan["groups"]
    if set(delete_ids) != {group["id"] for group in groups}:
        raise ValueError("Select files to delete for every duplicate group")
    retained, deletions = [], []
    for group in groups:
        files = {item["id"]: item for item in group["files"]}
        selected = delete_ids[group["id"]]
        if (not isinstance(selected, list) or not selected
                or any(not isinstance(file_id, str) for file_id in selected)
                or len(set(selected)) != len(selected) or not set(selected) <= files.keys()):
            raise ValueError("Each group requires valid, unique file IDs to delete")
        for file_id, item in files.items():
            if file_id in selected:
                deletions.append({"file": item})
            else:
                retained.append(item)
    return {"root_dir": scan["root_dir"], "agent_name": scan["agent_name"],
            "threshold": scan["threshold"], "retained": retained, "deletions": deletions}
