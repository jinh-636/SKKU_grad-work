from collections.abc import Callable
from typing import Any

from .models import SemanticProfile


InferJson = Callable[[dict[str, Any]], dict[str, Any]]


class SemanticAnalyzer:
    def __init__(self, infer_json: InferJson) -> None:
        self.infer_json = infer_json

    def analyze(self, content: str) -> SemanticProfile:
        result = self.infer_json(
            {
                "instruction": (
                    "Classify the content category and sensitivity. "
                    "Find sensitive values that should be masked."
                ),
                "content": content,
                "response_schema": {
                    "category": "string",
                    "sensitivity": "public|internal|confidential",
                    "sensitive_values": ["string"],
                },
            }
        )

        return SemanticProfile(
            category=result["category"],
            sensitivity=result["sensitivity"],
            sensitive_values=result.get("sensitive_values", []),
        )
