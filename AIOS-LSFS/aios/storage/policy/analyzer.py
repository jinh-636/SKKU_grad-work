from collections.abc import Callable, Mapping
from typing import Any

from .models import SemanticAnalysisError, SemanticProfile


InferJson = Callable[[dict[str, Any]], dict[str, Any]]
DEFAULT_CATEGORIES = {
    "source_code": "Program source code, scripts, or software configuration.",
    "report": "Research reports, papers, experiment notes, or technical reports.",
    "personal": "Personal records, contact information, or private financial records.",
    "other": "Content that does not primarily belong to the categories above.",
}
SENSITIVITIES = ("public", "internal", "confidential")


class SemanticAnalyzer:
    def __init__(
        self,
        infer_json: InferJson,
        categories: Mapping[str, str] | None = None,
    ) -> None:
        self.infer_json = infer_json
        self.categories = dict(DEFAULT_CATEGORIES if categories is None else categories)
        if not self.categories or any(
            not isinstance(name, str) or not name.strip()
            or not isinstance(description, str) or not description.strip()
            for name, description in self.categories.items()
        ):
            raise ValueError("categories must map non-empty names to descriptions")

    @classmethod
    def from_config(cls, settings: dict[str, Any] | None = None) -> "SemanticAnalyzer":
        """Create a real LLM analyzer using the existing AIOS configuration."""
        if settings is None:
            from aios.config.config_manager import config
            settings = config.config

        from .llm_client import LiteLLMMetadataClient

        options = settings.get("storage", {}).get("semantic_analyzer", {})
        return cls(
            infer_json=LiteLLMMetadataClient.from_config(settings),
            categories=options.get("categories"),
        )

    def analyze(self, content: str) -> SemanticProfile:
        if not isinstance(content, str):
            raise TypeError("content must be a string")

        result = self.infer_json(
            {
                "instruction": (
                    "Classify the content category and sensitivity. "
                    "Find sensitive values that should be masked."
                ),
                "content": content,
                "categories": dict(self.categories),
                "response_schema": {
                    "category": "string",
                    "sensitivity": "public|internal|confidential",
                    "sensitive_values": ["string"],
                },
            }
        )

        if not isinstance(result, dict):
            raise SemanticAnalysisError("Metadata response must be a JSON object")
        if set(result) - {"category", "sensitivity", "sensitive_values"}:
            raise SemanticAnalysisError("Metadata response contains unexpected fields")

        category = result.get("category")
        sensitivity = result.get("sensitivity")
        values = result.get("sensitive_values", [])

        if not isinstance(category, str) or category not in self.categories:
            raise SemanticAnalysisError("Metadata category is not configured")
        if not isinstance(sensitivity, str) or sensitivity not in SENSITIVITIES:
            raise SemanticAnalysisError("Metadata sensitivity is invalid")
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value or value not in content
            for value in values
        ):
            raise SemanticAnalysisError(
                "sensitive_values must contain non-empty exact substrings of the content"
            )

        return SemanticProfile(
            category=category,
            sensitivity=sensitivity,
            sensitive_values=list(dict.fromkeys(values)),
        )
