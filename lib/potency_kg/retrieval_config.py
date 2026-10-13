"""Every tunable value of the hybrid (vector + graph) retrieval and answering pipeline.

Why one class
-------------
Choosing the context for the LLM involves many small decisions: how many vector hits,
how far to walk the graph, how to score what the walk finds, how to split the character
budget between vector and graph evidence, and so on. They all live here, in ONE frozen
object, so that:

* nothing is hidden as a magic number inside retrieval code;
* the exact configuration of a run can be written into its trace (`as_dict`) and
  fingerprinted (`config_hash`), which is what makes a vector-vs-graph comparison
  reproducible and auditable;
* the values can be frozen (committed) BEFORE the first real run, so they cannot be
  tuned on the answers afterwards. Any later change must be disclosed and both modes
  re-run.

None of these values refer to a product, a document, or a question ID: the pipeline
stays one generic, scope-driven engine.

Pipeline in one paragraph
-------------------------
1. Embed the question and take the `vector_hits` most similar passages.
2. (graph mode only) Pick up to `max_seeds` graph entities that cite those passages,
   then walk the graph up to `max_hops` hops. At every hop each entity in the frontier
   keeps only its `neighbors_per_entity` best neighbors (a beam). Every neighbor is scored
   (see "Scoring"). Entities that are hubs (more than `hub_degree_cap` relations) are not
   used as seeds and are not expanded.
3. Every entity is represented by the best passage it cites; the passages become
   graph candidates.
4. Merge: vector hits fill `1 - graph_share` of the character budget, graph-only
   candidates fill `graph_share`, in score order. The last passage is truncated to fit.
5. The context is sent to the LLM, which must cite passages with verbatim quotes.

Scoring (graph candidates only)
-------------------------------
    score = similarity_weight * cosine(question, passage)
            + hop_bonus[hop - 1]
            + edge_type_bonus[type of the edge used to reach the entity]

The bonuses are additive, in cosine units. Embedding cosines between a question and
relevant passages are tightly clustered (a spread of about 0.1 across the top hits), so
0.05 is already a strong push. Set the bonus values from the observed score
distribution in the graph-signal measurement step, then freeze them.
Hop weights only reorder candidates INSIDE the graph share; the size of that share is
what controls how much of the context the graph can claim (`graph_share`).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

from .ids import canonical_json

# Edge types that describe the potency story (what was proposed, requested, revised,
# concluded, measured, or used in which role). They are sparse in the current graph,
# so they get a small bonus: they should survive pruning when they exist.
_POTENCY_EDGE_TYPES: tuple[str, ...] = (
    "proposed_in",
    "requested_in",
    "revised_by",
    "concluded_in",
    "measures",
    "used_as",
)


@dataclass(frozen=True)
class HybridRetrievalConfig:
    """Immutable configuration for one retrieval + context + answering run."""

    # ------------------------------------------------------------------ vector
    # Number of passages taken from the vector (embedding) search. They are the
    # candidates of `vector` mode and the seed passages of `graph` mode.
    vector_hits: int = 8

    # ------------------------------------------------------------------- seeds
    # Maximum number of graph entities used as starting points of the walk. Seeds are
    # non-hub, non-document, non-passage entities that cite a vector-hit passage,
    # ordered by the vector score of that passage, then by entity ID (deterministic).
    max_seeds: int = 8

    # --------------------------------------------------------------- traversal
    # "beam": similarity-pruned walk in SQL (preferred). "age": the existing bounded
    # Apache AGE traversal with hub skipping (fallback; hard-capped at 2 hops).
    traversal: Literal["beam", "age"] = "beam"

    # N: how many neighbors each entity keeps per hop (the beam width). Worst case the
    # walk visits N + N^2 + ... + N^max_hops entities (155 for N=5 and 3 hops).
    neighbors_per_entity: int = 5

    # Number of hops from a seed. A hop is one relation (edge), followed in either
    # direction (the walk is undirected).
    max_hops: int = 3

    # An entity with more relations than this is a "hub" (for example a single product
    # node linked to hundreds of assertions). Hubs are never seeds and are not expanded:
    # walking through them would reach arbitrary evidence. Detection is by degree only,
    # never by name.
    hub_degree_cap: int = 20

    # Upper bound on graph candidate passages considered before ranking; a safety
    # limit on SQL work, not a quality knob.
    candidate_cap: int = 500

    # ----------------------------------------------------------------- scoring
    # Multiplier of the question-passage cosine similarity in the graph score.
    similarity_weight: float = 1.0

    # Bonus added by distance from the seed, one value per hop (index 0 = hop 1). Closer
    # entities are usually more directly related, so the defaults decrease with the hop.
    # Must have exactly `max_hops` values. Starting values: to be fixed from the
    # measurement step, then frozen.
    hop_bonus: tuple[float, ...] = (0.05, 0.03, 0.01)

    # Bonus by the type of the relation used to reach an entity, as (edge_type, bonus)
    # pairs (a tuple, so the config stays hashable and immutable). Types not listed get
    # 0. Used both when pruning the beam and when ranking graph candidates.
    edge_type_bonus: tuple[tuple[str, float], ...] = tuple(
        (edge_type, 0.05) for edge_type in _POTENCY_EDGE_TYPES
    )

    # ------------------------------------------------------------------- merge
    # Share of the character budget reserved for graph-only passages (passages that are
    # not among the vector hits) in `graph` mode. The rest goes to vector hits. If the
    # graph supplies less than its share, the remainder returns to the unused vector
    # hits. `vector` mode ignores this and uses the whole budget for vector hits.
    # There is deliberately no similarity floor: the share is always used.
    graph_share: float = 0.4

    # When the last passage does not fit the budget it is cut to a verbatim prefix, but
    # only if at least this many characters remain; otherwise it is dropped.
    min_truncated_chars: int = 300

    # ----------------------------------------------------------------- budgets
    # Source-evidence budget (Unicode code points, as Python `len`) used when a
    # question does not carry its own cap. A question that sets
    # `max_evidence_chars_per_mode` (in questions.json) overrides this value, so the
    # assignment's cap is data-driven and no question ID appears in code.
    default_budget_chars: int = 16000

    # -------------------------------------------------------------- generation
    # Name of the answer prompt, recorded in every trace. Bump it when the text changes.
    prompt_version: str = "answer-prompt-v1"

    # Extra attempts after the first one when a cited quote cannot be found in the
    # evidence that was sent. The feedback sent on retry names the failed citations
    # and adds no new source text. 0 disables the repair.
    citation_repair_attempts: int = 1

    # A quote is first matched exactly. If True, a match after collapsing whitespace on
    # both sides is also accepted and flagged "normalized" (PDF text contains line
    # breaks and form feeds). Matching is never fuzzy beyond whitespace.
    allow_normalized_quote_match: bool = True

    def __post_init__(self) -> None:
        ints = {
            "vector_hits": (self.vector_hits, 1, 100),
            "max_seeds": (self.max_seeds, 0, 100),
            "neighbors_per_entity": (self.neighbors_per_entity, 1, 100),
            "max_hops": (self.max_hops, 0, 6),
            "hub_degree_cap": (self.hub_degree_cap, 1, 100_000),
            "candidate_cap": (self.candidate_cap, 1, 100_000),
            "min_truncated_chars": (self.min_truncated_chars, 0, 100_000),
            "default_budget_chars": (self.default_budget_chars, 1, 1_000_000),
            "citation_repair_attempts": (self.citation_repair_attempts, 0, 5),
        }
        for name, (value, low, high) in ints.items():
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        if self.traversal not in ("beam", "age"):
            raise ValueError("traversal must be 'beam' or 'age'")
        if self.traversal == "age" and self.max_hops > 2:
            raise ValueError("the AGE traversal supports at most 2 hops")
        if not 0.0 <= self.graph_share <= 1.0:
            raise ValueError("graph_share must be between 0 and 1")
        if not self.similarity_weight > 0:
            raise ValueError("similarity_weight must be positive")
        if len(self.hop_bonus) != self.max_hops:
            raise ValueError("hop_bonus needs exactly one value per hop (max_hops values)")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in self.hop_bonus):
            raise ValueError("hop_bonus values must be numbers")
        edge_types = [edge_type for edge_type, _ in self.edge_type_bonus]
        if len(set(edge_types)) != len(edge_types) or not all(
            isinstance(edge_type, str) and edge_type for edge_type in edge_types
        ):
            raise ValueError("edge_type_bonus needs unique, non-empty edge type names")
        if any(isinstance(b, bool) or not isinstance(b, (int, float)) for _, b in self.edge_type_bonus):
            raise ValueError("edge_type_bonus values must be numbers")
        if not isinstance(self.prompt_version, str) or not self.prompt_version:
            raise ValueError("prompt_version must be a non-empty string")

    def edge_bonus(self, edge_type: str) -> float:
        """Bonus for reaching an entity through `edge_type` (0.0 if not listed)."""
        return dict(self.edge_type_bonus).get(edge_type, 0.0)

    def hop_weight(self, hop: int) -> float:
        """Bonus for an entity first reached at `hop` (1-based); 0.0 for hop 0 or beyond."""
        return self.hop_bonus[hop - 1] if 1 <= hop <= len(self.hop_bonus) else 0.0

    def graph_score(self, cosine: float, hop: int, edge_type: str | None) -> float:
        """Score of a graph candidate; the single place the scoring formula lives."""
        return (
            self.similarity_weight * cosine
            + self.hop_weight(hop)
            + (self.edge_bonus(edge_type) if edge_type else 0.0)
        )

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready copy of every field, written into each run trace."""
        return {
            "vector_hits": self.vector_hits,
            "max_seeds": self.max_seeds,
            "traversal": self.traversal,
            "neighbors_per_entity": self.neighbors_per_entity,
            "max_hops": self.max_hops,
            "hub_degree_cap": self.hub_degree_cap,
            "candidate_cap": self.candidate_cap,
            "similarity_weight": self.similarity_weight,
            "hop_bonus": list(self.hop_bonus),
            "edge_type_bonus": {edge_type: bonus for edge_type, bonus in self.edge_type_bonus},
            "graph_share": self.graph_share,
            "min_truncated_chars": self.min_truncated_chars,
            "default_budget_chars": self.default_budget_chars,
            "prompt_version": self.prompt_version,
            "citation_repair_attempts": self.citation_repair_attempts,
            "allow_normalized_quote_match": self.allow_normalized_quote_match,
        }

    def config_hash(self) -> str:
        """Short stable fingerprint of the configuration, stored in every trace."""
        return hashlib.sha256(canonical_json(self.as_dict()).encode("utf-8")).hexdigest()
