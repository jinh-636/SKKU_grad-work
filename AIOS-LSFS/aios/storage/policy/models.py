from dataclasses import dataclass, field


class SemanticAnalysisError(ValueError):
    """Metadata inference or validation failed; no profile is available."""


@dataclass(frozen=True)
class SemanticProfile:
    category: str
    sensitivity: str
    sensitive_values: list[str] = field(default_factory=list)
