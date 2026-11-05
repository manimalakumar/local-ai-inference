"""Retrieve privacy-reviewed schema facts with a local text embedding model."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from clinical_trial_mvp.data.schema import RELATIONSHIPS, TABLE_ORDER, TABLE_SCHEMAS
from clinical_trial_mvp.data.schema_selection import TABLE_CUES, _shortest_path


RETRIEVAL_VERSION = "mvp1-embed-v4"
EMBEDDING_PROFILES = ("nemotron_embed_1b", "qwen3_embed_06b", "bge_m3")
EMBEDDING_ENCODING_VERSION = "mvp1-embedding-profiles-v1"
PRIVACY_NOTE = (
    "Privacy: subject_id is a private join key. Prefer aggregates; the MCP server "
    "pseudonymizes subject_id in displayed results."
)

# Controlled schema-level examples only. Never populate this map from database rows.
SAFE_EXAMPLES = {
    ("trials", "phase"): ("Phase I", "Phase II", "Phase III", "Phase IV"),
    ("subjects", "screening_status"): ("Enrolled", "Randomized", "Screen Failed", "Withdrawn"),
    ("visits", "status"): ("Completed", "Missed", "Pending", "Screen Failed"),
    ("adverse_events", "severity"): ("Mild", "Moderate", "Severe"),
    ("lab_results", "test_code"): ("ALT", "AST", "HGB", "WBC", "CREAT"),
    ("data_queries", "status"): ("Open", "Answered", "Closed"),
    ("site_monitoring", "visit_type"): ("Initiation", "Interim", "For-cause", "Closeout"),
}

BUSINESS_DEFINITIONS = {
    "enrollment": "Enrolled subjects: screening_status IN ('Enrolled', 'Randomized'); count subjects, not child rows.",
    "screen_failure": "A screen-failed subject has screening_status = 'Screen Failed'. Count DISTINCT subject_id because one subject can have several criterion-level screen_failures. State the denominator for a failure rate.",
    "site_ranking": "Site ranking: return each site and its metric, sorted by the requested metric. LIMIT only when a top N or single winner is requested.",
    "overall_grain": "Overall means one aggregate result row. Group by site, arm, trial, or month only when requested.",
    "serious_events": "Serious adverse event means serious = true. Severity = 'Severe' is a different concept.",
    "monitoring": "A monitoring visit is a row of site_monitoring; a protocol subject visit is a row of visits.",
    "site_17": "Site 17 maps to site_id = 'SITE-017'. Do not infer a cause from operational correlation.",
    "rate": "For a rate, show the numerator and denominator. Preserve requested zero-count groups with LEFT JOIN.",
}

DEFINITION_TRIGGERS = {
    "enrollment": r"\benroll\w*\b",
    "screen_failure": r"\bscreen.fail\w*\b|\bscreening fail\w*\b",
    "site_ranking": r"\b(?:top|longest|highest|lowest|rank\w*)\b",
    "overall_grain": r"\boverall\b",
    "serious_events": r"\bserious\b|\bsae\b",
    "monitoring": r"\bmonitor\w*\b",
    "site_17": r"\bsite\s*17\b",
    "rate": r"\brates?\b|\bpercent\w*\b",
}


@dataclass(frozen=True)
class SchemaFact:
    fact_id: str
    kind: str
    text: str
    table: str | None = None
    column: str | None = None


class TextEmbedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class LocalSentenceEmbedder:
    """Load an already downloaded embedding model; never contact a remote service."""

    def __init__(self, model_path: Path, profile: str = "nemotron_embed_1b"):
        if profile not in EMBEDDING_PROFILES:
            raise ValueError(f"Unknown embedding profile: {profile}")
        if not model_path.exists():
            raise FileNotFoundError(f"Local embedding model missing: {model_path}")
        from sentence_transformers import SentenceTransformer

        self.profile = profile
        self.model = SentenceTransformer(str(model_path), local_files_only=True, trust_remote_code=False, device="cpu")

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.profile == "nemotron_embed_1b":
            prepared = texts
            options = {}
        else:
            query = all(text.startswith("query: ") for text in texts)
            passage = all(text.startswith("passage: ") for text in texts)
            if not (query or passage):
                raise ValueError("Embedding inputs must be all queries or all passages")
            prepared = [text.split(": ", 1)[1] for text in texts]
            # Qwen's model card supplies a query prompt. BGE-M3 dense retrieval
            # uses unprefixed query and document strings.
            options = {"prompt_name": "query"} if query and self.profile == "qwen3_embed_06b" else {}
        return self.model.encode(
            prepared, batch_size=8, normalize_embeddings=True,
            show_progress_bar=False, **options,
        ).tolist()


def schema_facts() -> tuple[SchemaFact, ...]:
    """Build the index only from reviewed code definitions and category examples."""
    facts: list[SchemaFact] = []
    for table in TABLE_ORDER:
        facts.append(SchemaFact(f"table:{table}", "table", f"Table {table}: clinical trial records and measures in {table}.", table))
        for column in TABLE_SCHEMAS[table]:
            examples = SAFE_EXAMPLES.get((table, column.name), ())
            value_text = f" Examples: {', '.join(examples)}." if examples else ""
            facts.append(SchemaFact(
                f"column:{table}.{column.name}", "column",
                f"{table}.{column.name} {column.dtype}: {column.description}{value_text}", table, column.name,
            ))
    for index, relationship in enumerate(RELATIONSHIPS):
        facts.append(SchemaFact(f"relationship:{index}", "relationship", relationship))
    for name, definition in BUSINESS_DEFINITIONS.items():
        facts.append(SchemaFact(f"definition:{name}", "definition", definition))
    return tuple(facts)


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm or not math.isfinite(norm):
        raise ValueError("Embedding model produced a zero or invalid vector")
    return [value / norm for value in vector]


@dataclass(frozen=True)
class RetrievalResult:
    prompt: str
    tables: tuple[str, ...]
    columns: dict[str, tuple[str, ...]]
    fact_ids: tuple[str, ...]
    scores: dict[str, float]
    retrieval_seconds: float


class SchemaRetriever:
    def __init__(self, embedder: TextEmbedder, *, top_k: int = 12, max_seed_tables: int = 4):
        if top_k < 1 or max_seed_tables < 1:
            raise ValueError("top_k and max_seed_tables must be positive")
        self.embedder = embedder
        self.top_k = top_k
        self.max_seed_tables = max_seed_tables
        self.facts = schema_facts()
        self.vectors = [_unit(list(row)) for row in embedder.encode(["passage: " + fact.text for fact in self.facts])]
        if len(self.vectors) != len(self.facts):
            raise ValueError("Embedding count does not match schema fact count")

    def retrieve(self, question: str, *, query_vector: list[float] | None = None) -> RetrievalResult:
        started = time.perf_counter()
        query = _unit(list(query_vector if query_vector is not None else self.embedder.encode(["query: " + question])[0]))
        scored = sorted(
            ((sum(a*b for a, b in zip(query, vector)), fact) for vector, fact in zip(self.vectors, self.facts)),
            key=lambda item: item[0], reverse=True,
        )
        hits = [
            (score, fact) for score, fact in scored
            if fact.kind != "definition" or re.search(
                DEFINITION_TRIGGERS[fact.fact_id.removeprefix("definition:")], question, re.I
            )
        ][:self.top_k]
        # Embed scores rank candidates; explicit clinical nouns disambiguate
        # near-synonyms such as protocol visits and site-monitoring visits.
        protocol_question = re.sub(r"\bmonitoring visits?\b", "", question, flags=re.I)
        seeds = [table for table in TABLE_ORDER if re.search(
            TABLE_CUES[table], protocol_question if table == "visits" else question, re.I
        )][:self.max_seed_tables]
        best = max((score for score, fact in hits if fact.table), default=0.0)
        for score, fact in hits:
            if fact.table and fact.table not in seeds and (not seeds or score >= 0.9 * best):
                seeds.append(fact.table)
            if len(seeds) == self.max_seed_tables:
                break
        if not seeds:
            seeds = [next(fact.table for _score, fact in scored if fact.table)]
        selected = set(seeds)
        # Clinical terms can imply a denominator table even when its name is absent.
        if re.search(r"\benroll\w*\b", question, re.I) and "sites" in selected:
            selected.add("subjects")
        for index, left in enumerate(seeds):
            for right in seeds[index + 1:]:
                selected.update(_shortest_path(left, right))
        tables = tuple(table for table in TABLE_ORDER if table in selected)
        columns: dict[str, set[str]] = {table: set() for table in tables}
        for _score, fact in hits:
            if fact.kind == "column" and fact.table in columns and fact.column:
                columns[fact.table].add(fact.column)
        if "subjects" in selected and re.search(r"\benroll\w*\b|\bserious.*rates?\b", question, re.I):
            columns["subjects"].add("screening_status")
            columns["subjects"].add("subject_id")
        if "data_queries" in selected and re.search(r"\bresolution\b", question, re.I):
            columns["data_queries"].add("status")
        relationships = []
        for relationship in RELATIONSHIPS:
            match = re.match(r"(\w+)\.(\w+) -> (\w+)\.(\w+)", relationship)
            if match and match.group(1) in selected and match.group(3) in selected:
                relationships.append(relationship)
                columns[match.group(1)].add(match.group(2))
                columns[match.group(3)].add(match.group(4))
        lines = ["DuckDB clinical trial schema (retrieved facts):"]
        final_columns: dict[str, tuple[str, ...]] = {}
        for table in tables:
            lines.append(f"\nTable: {table}")
            chosen = tuple(column.name for column in TABLE_SCHEMAS[table] if column.name in columns[table])
            final_columns[table] = chosen
            for column in TABLE_SCHEMAS[table]:
                if column.name in columns[table]:
                    examples = SAFE_EXAMPLES.get((table, column.name), ())
                    value_text = f" Examples: {', '.join(examples)}." if examples else ""
                    lines.append(f"- {column.name} {column.dtype}: {column.description}{value_text}")
        lines.append("\nRelationships:")
        lines.extend(f"- {item}" for item in relationships)
        definitions = [fact.text for _score, fact in hits if fact.kind == "definition"]
        if definitions:
            lines.append("\nClinical business definitions:")
            lines.extend(f"- {item}" for item in definitions)
        lines.append(f"\n{PRIVACY_NOTE}")
        return RetrievalResult(
            prompt="\n".join(lines), tables=tables, columns=final_columns,
            fact_ids=tuple(fact.fact_id for _score, fact in hits),
            scores={fact.fact_id: round(score, 5) for score, fact in hits},
            retrieval_seconds=time.perf_counter() - started,
        )


def corpus_sha256() -> str:
    payload = [fact.__dict__ for fact in schema_facts()]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
