"""Common Fact dataclass for ET1 experiments.

All three data sources (Meridian, FEVER, SQuAD 2.0) produce Fact objects.
The data formatter (C1-C7) consumes Fact objects. This is the bridge.

Invariants enforced at construction time:
- SL opinion: b + d + u = 1.0, all components in [0, 1]
- Provenance: at least one source, non-empty method
- Dataset: must be one of {"meridian", "fever", "squad"}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


_VALID_DATASETS = {"meridian", "fever", "squad"}


@dataclass(frozen=True)
class SLOpinion:
    """A Subjective Logic opinion (b, d, u, a).

    Validated at construction: b + d + u must equal 1.0 (within 1e-6),
    all components must be in [0, 1].
    """

    belief: float
    disbelief: float
    uncertainty: float
    base_rate: float = 0.5

    def __post_init__(self) -> None:
        for name in ("belief", "disbelief", "uncertainty", "base_rate"):
            val = getattr(self, name)
            if val < 0.0 or val > 1.0:
                raise ValueError(
                    f"SLOpinion components must be non-negative and <= 1.0, "
                    f"got {name}={val}"
                )
        total = self.belief + self.disbelief + self.uncertainty
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"SLOpinion b + d + u must sum to 1.0, "
                f"got {total} (b={self.belief}, d={self.disbelief}, "
                f"u={self.uncertainty})"
            )

    @property
    def projected_probability(self) -> float:
        """P(ω) = b + a·u per Subjective Logic."""
        return self.belief + self.base_rate * self.uncertainty

    def to_dict(self) -> dict:
        return {
            "belief": self.belief,
            "disbelief": self.disbelief,
            "uncertainty": self.uncertainty,
            "base_rate": self.base_rate,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SLOpinion:
        return cls(
            belief=d["belief"],
            disbelief=d["disbelief"],
            uncertainty=d["uncertainty"],
            base_rate=d.get("base_rate", 0.5),
        )


@dataclass(frozen=True)
class Source:
    """A provenance source with an identifier and reliability rating."""

    id: str
    name: str
    reliability: float

    def __post_init__(self) -> None:
        if self.reliability < 0.0 or self.reliability > 1.0:
            raise ValueError(
                f"Source reliability must be in [0, 1], got {self.reliability}"
            )

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "reliability": self.reliability}

    @classmethod
    def from_dict(cls, d: dict) -> Source:
        return cls(id=d["id"], name=d["name"], reliability=d["reliability"])


@dataclass(frozen=True)
class Provenance:
    """Provenance metadata: sources and verification method."""

    sources: tuple[Source, ...] | list[Source]
    method: str

    def __post_init__(self) -> None:
        # Normalize list to tuple for immutability
        if isinstance(self.sources, list):
            object.__setattr__(self, "sources", tuple(self.sources))
        if len(self.sources) == 0:
            raise ValueError("Provenance must have at least one source")
        if not self.method or not self.method.strip():
            raise ValueError("Provenance method must be a non-empty string")

    def to_dict(self) -> dict:
        return {
            "sources": [s.to_dict() for s in self.sources],
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Provenance:
        return cls(
            sources=[Source.from_dict(s) for s in d["sources"]],
            method=d["method"],
        )


@dataclass(frozen=True)
class Fact:
    """A single fact: the unit of data across all three datasets.

    This is the common representation that all data sources produce
    and the 7-condition formatter consumes.
    """

    # Identity
    id: str
    dataset: str  # "meridian", "fever", or "squad"

    # Content
    question: str
    answer: str
    entity_name: str
    entity_type: str
    relation: str

    # Uncertainty metadata (the independent variable)
    opinion: SLOpinion
    provenance: Provenance

    # Optional fields
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    context: Optional[str] = None  # SQuAD context paragraph
    tier: Optional[str] = None  # Confidence tier label (for evaluation)

    def __post_init__(self) -> None:
        if self.dataset not in _VALID_DATASETS:
            raise ValueError(
                f"dataset must be one of {_VALID_DATASETS}, got '{self.dataset}'"
            )

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "dataset": self.dataset,
            "question": self.question,
            "answer": self.answer,
            "entity_name": self.entity_name,
            "entity_type": self.entity_type,
            "relation": self.relation,
            "opinion": self.opinion.to_dict(),
            "provenance": self.provenance.to_dict(),
        }
        if self.valid_from is not None:
            d["valid_from"] = self.valid_from
        if self.valid_until is not None:
            d["valid_until"] = self.valid_until
        if self.context is not None:
            d["context"] = self.context
        if self.tier is not None:
            d["tier"] = self.tier
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Fact:
        return cls(
            id=d["id"],
            dataset=d["dataset"],
            question=d["question"],
            answer=d["answer"],
            entity_name=d["entity_name"],
            entity_type=d["entity_type"],
            relation=d["relation"],
            opinion=SLOpinion.from_dict(d["opinion"]),
            provenance=Provenance.from_dict(d["provenance"]),
            valid_from=d.get("valid_from"),
            valid_until=d.get("valid_until"),
            context=d.get("context"),
            tier=d.get("tier"),
        )

    @classmethod
    def from_meridian(cls, raw: dict) -> Fact:
        """Convert a raw Meridian KB fact dict to a Fact."""
        opinion = SLOpinion(
            belief=raw["opinion"]["belief"],
            disbelief=raw["opinion"]["disbelief"],
            uncertainty=raw["opinion"]["uncertainty"],
            base_rate=raw["opinion"].get("base_rate", 0.5),
        )
        sources = [
            Source(id=s["id"], name=s["name"], reliability=s["reliability"])
            for s in raw["provenance"]["sources"]
        ]
        provenance = Provenance(
            sources=sources,
            method=raw["provenance"]["method"],
        )
        # Use first question from the list
        question = raw["questions"][0] if raw.get("questions") else "?"

        return cls(
            id=raw["id"],
            dataset="meridian",
            question=question,
            answer=raw["value"],
            entity_name=raw["entity_name"],
            entity_type=raw["entity_type"],
            relation=raw["relation"],
            opinion=opinion,
            provenance=provenance,
            valid_from=raw.get("valid_from"),
            valid_until=raw.get("valid_until"),
            tier=raw.get("tier"),
        )
