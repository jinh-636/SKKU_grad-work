from dataclasses import dataclass, field


@dataclass(frozen=True)
class SemanticProfile:
    category: str
    sensitivity: str
    sensitive_values: list[str] = field(default_factory=list)
