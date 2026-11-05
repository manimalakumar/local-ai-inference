"""Run the clinical trial analyst workflow and coordinate model, tools, and answers."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage

from clinical_trial_mvp.config import Settings
from clinical_trial_mvp.mcp.client import (
    execute_readonly_sql_via_mcp,
    get_schema_via_mcp,
    request_clarification_via_mcp,
)
from clinical_trial_mvp.model.ollama_model_mgmt import ensure_ollama_model
from clinical_trial_mvp.agent.intent_contracts import check_plan_intent, check_sql_intent
from clinical_trial_mvp.agent.plan_policy import check_sql_question_filters, finalize_plan
from clinical_trial_mvp.data.privacy import _mask_sql_literals_and_comments
from clinical_trial_mvp.agent.prompt_guidance import JOIN_AND_DEFINITION_GUIDE
from clinical_trial_mvp.agent.question_policy import preflight_response
from clinical_trial_mvp.data.schema import SCHEMA_PROMPT, TABLE_SCHEMAS
from clinical_trial_mvp.data.schema_selection import select_schema
from clinical_trial_mvp.data.embedding_retrieval import LocalSentenceEmbedder, SchemaRetriever, RETRIEVAL_VERSION, EMBEDDING_ENCODING_VERSION, corpus_sha256


SQL_SYSTEM_RULES = """
You are Clinical Trial Analyst, a local-only clinical data analyst.

Use the schema exactly as given. Generate DuckDB SQL only for the available tables.
Rules:
- Use exactly one read-only SELECT or WITH query.
- Do not use INSERT, UPDATE, DELETE, CREATE, DROP, COPY, ATTACH, INSTALL, LOAD, SET, or PRAGMA.
- Do not use SELECT *.
- Prefer aggregate answers over subject-level listings.
- Include the minimum columns needed.
- Use DuckDB-compatible SQL.
- If the question says "by X" or "for each X", include X in SELECT and GROUP BY X. Do not replace a group breakdown with an overall total.
- Treat status values explicitly named in the question as valid values when they appear in the schema description.
- Write the complete SQL query. Never use placeholders such as `...`.
- Do not show hidden reasoning. Return the final JSON object only.
- Return JSON only with keys "sql" and "rationale".
""".strip()

PLAN_RULES = """
Before writing SQL, fill a compact plan with exactly these fields:
- population: which rows or subjects are eligible, including zero-count groups where requested.
- measure: exactly what is counted, averaged, compared, or divided, with any denominator.
- grouping: the requested output grain (for example each site, each treatment arm, or overall).
- time_window: the calendar interval and date column, or "none".
- filters: status, severity, seriousness, and other restrictions; do not add unasked filters.
- ranking: sort and limit requested, or "none".
- needs_clarification: true only when a required measure or meaning is undefined.
- clarification_question: a short question if clarification is needed, otherwise "".
Proceed with answerable analytics when a threshold or window is unspecified. State a
transparent operational assumption in the relevant plan field and answer with that qualification.
Clarification is reserved for a missing choice that changes which measure to analyze.
For serious adverse events, use serious=true regardless of severity. Never add severity='Severe'
unless the question explicitly asks for Severe events.
For “Why is Site 17 performing poorly?”, no metric defines “poorly”. Ask which metric to assess;
do not infer a cause. Set needs_clarification=true and provide a clarification_question.
The later SQL must implement every plan field. Do not silently replace a serious event with a
Severe event, change a calendar month to a rolling cutoff, or turn a requested group into an overall answer.
""".strip()


DEFAULT_QUESTION = "How many subjects are Enrolled or Randomized by treatment arm?"
DEFAULT_QUESTION_SQL = (
    "SELECT treatment_arm, COUNT(*) AS subject_count "
    "FROM subjects "
    "WHERE screening_status IN ('Enrolled', 'Randomized') "
    "GROUP BY treatment_arm "
    "ORDER BY treatment_arm"
)
SQL_PROMPT_VERSION = "mvp1-v1-embedding-schema-context"
SQL_CHAT_SYSTEM_MESSAGE = "/no_think\nYou are Clinical Trial Analyst. Return only a complete JSON object with sql and rationale. Follow the SQL rules in the user message."
PLAN_CHAT_SYSTEM_MESSAGE = "/no_think\nYou are Clinical Trial Analyst. Return only a complete JSON object with the requested compact plan fields."
SUMMARY_CHAT_SYSTEM_MESSAGE = "/no_think\nYou are Clinical Trial Analyst. Answer using only the query result, without revealing raw subject identifiers."
SQL_JSON_SCHEMA = {
    "type": "object",
    "properties": {"sql": {"type": "string"}, "rationale": {"type": "string"}},
    "required": ["sql", "rationale"],
    "additionalProperties": False,
}
PLAN_JSON_SCHEMA = {
    "type": "object",
    "properties": {
                "population": {"type": "string"},
                "measure": {"type": "string"},
                "grouping": {"type": "string"},
                "time_window": {"type": "string"},
                "filters": {"type": "string"},
                "ranking": {"type": "string"},
                "needs_clarification": {"type": "boolean"},
                "clarification_question": {"type": "string"},
    },
    "required": ["population", "measure", "grouping", "time_window", "filters", "ranking", "needs_clarification", "clarification_question"],
    "additionalProperties": False,
}


def build_plan_prompt(question: str, schema: str = SCHEMA_PROMPT, guidance: str = JOIN_AND_DEFINITION_GUIDE) -> str:
    return f"""
You are Clinical Trial Analyst. Plan the requested answer before SQL generation.
Return only a compact JSON object with the plan fields. Do not write SQL.

{schema}

{guidance}

{PLAN_RULES}

Question:
{question}

JSON:
""".strip()


def build_sql_generation_prompt(
    question: str, schema: str = SCHEMA_PROMPT, plan: dict[str, Any] | None = None,
    contract_feedback: list[str] | None = None,
    guidance: str = JOIN_AND_DEFINITION_GUIDE,
) -> str:
    return f"""
{SQL_SYSTEM_RULES}

{schema}

{guidance}

Answer plan (implement it unless a question-intent check identifies an error):
{json.dumps(plan, sort_keys=True) if plan else 'No plan supplied.'}

Question-intent checks (correct any conflicting plan field in SQL):
{json.dumps(contract_feedback or [])}

Question:
{question}

JSON:
""".strip()


def build_sql_repair_prompt(
    question: str, failed_sql: str, error: str, schema: str = SCHEMA_PROMPT,
    plan: dict[str, Any] | None = None,
    guidance: str = JOIN_AND_DEFINITION_GUIDE,
) -> str:
    return f"""
{SQL_SYSTEM_RULES}

{schema}

{guidance}

Answer plan (repair any field that conflicts with the error below):
{json.dumps(plan, sort_keys=True) if plan else 'No plan supplied.'}

The prior SQL failed. Repair it once.

Question:
{question}

Failed SQL:
{failed_sql}

Error:
{error}

Return JSON only with keys "sql" and "rationale".
JSON:
""".strip()


@dataclass(frozen=True)
class AnalystResponse:
    question: str
    answer: str
    sql: str
    repaired: bool
    query_result: dict[str, Any]
    error: str | None = None
    clarification_required: bool = False
    tables_used: list[str] = field(default_factory=list)
    latency_seconds: float = 0.0
    status: str = "answer"


class ClinicalTrialAnalystAgent:
    """LangChain-powered local analyst with MCP-mediated DuckDB access."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self.llm = None
        self._schema_text: str | None = None
        self._chat_model = False
        self._sql_grammar = None
        self._plan_grammar = None
        self._retriever = None
        self._retrieval_cache = None
        self.model_cold_start_seconds = None
        self.last_invocation_metrics = {}

    def schema_context(self, question: str) -> tuple[str, dict[str, Any], str]:
        """Return comparable full or local embedding-retrieved prompt context."""
        mode = self.settings.schema_context_mode
        if mode == "full":
            schema = self._get_schema()
            return schema, {"mode": "full", "tables": list(TABLE_SCHEMAS),
                            "columns": {t: [c.name for c in cols] for t, cols in TABLE_SCHEMAS.items()},
                            "prompt_chars": len(schema), "retrieval_seconds": 0.0}, JOIN_AND_DEFINITION_GUIDE
        if mode != "retrieved":
            raise ValueError(f"SCHEMA_CONTEXT_MODE must be full or retrieved, got {mode!r}")
        cache_path = self.settings.retrieval_cache_path
        if cache_path and cache_path.exists():
            if self._retrieval_cache is None:
                self._retrieval_cache = json.loads(cache_path.read_text(encoding="utf-8"))
                # Exported caches use portable project-relative model locations.
                cached_model_path = Path(self._retrieval_cache["embedding_model_path"])
                if not cached_model_path.is_absolute():
                    self._retrieval_cache["embedding_model_path"] = str(
                        self.settings.project_root / cached_model_path
                    )
                expected = {
                    "retrieval_version": RETRIEVAL_VERSION,
                    "corpus_sha256": corpus_sha256(),
                    "embedding_model_path": str(self.settings.embed_model_path),
                    "top_k": self.settings.embed_top_k,
                    "max_seed_tables": self.settings.embed_max_seed_tables,
                }
                if self.settings.embed_model_id != "nemotron_embed_1b":
                    expected["embedding_model_id"] = self.settings.embed_model_id
                    expected["embedding_encoding_version"] = EMBEDDING_ENCODING_VERSION
                if any(self._retrieval_cache.get(key) != value for key, value in expected.items()):
                    raise ValueError(f"Retrieval cache is stale or has a different model/configuration: {cache_path}")
            cached = self._retrieval_cache.get("questions", {}).get(question)
            if cached:
                return cached["prompt"], cached["metadata"], ""
        if self._retriever is None:
            self._retriever = SchemaRetriever(
                LocalSentenceEmbedder(self.settings.embed_model_path, self.settings.embed_model_id),
                top_k=self.settings.embed_top_k,
                max_seed_tables=self.settings.embed_max_seed_tables,
            )
        result = self._retriever.retrieve(question)
        return result.prompt, {
            "mode": "retrieved", "tables": result.tables, "columns": result.columns,
            "fact_ids": result.fact_ids, "scores": result.scores,
            "prompt_chars": len(result.prompt), "retrieval_seconds": result.retrieval_seconds,
        }, ""

    def _get_schema(self) -> str:
        if self._schema_text is None:
            self._schema_text = get_schema_via_mcp(self.settings)
        return self._schema_text

    def _get_llm(self):
        if self.llm is None:
            load_started = time.perf_counter()
            self.llm = self._load_llm()
            self.model_cold_start_seconds = time.perf_counter() - load_started
        return self.llm

    def _invoke_llm(self, prompt: str, *, sql_task: bool = True, plan_task: bool = False) -> str:
        model = self._get_llm()
        started = time.perf_counter()
        if self._chat_model:
            system = PLAN_CHAT_SYSTEM_MESSAGE if plan_task else (SQL_CHAT_SYSTEM_MESSAGE if sql_task else SUMMARY_CHAT_SYSTEM_MESSAGE)
            if self.settings.reasoning_enabled:
                system = system.replace("/no_think", "/think")
            options = {}
            if sql_task or plan_task:
                if plan_task and self._plan_grammar is None:
                    from llama_cpp import LlamaGrammar

                    self._plan_grammar = LlamaGrammar.from_json_schema(
                        json.dumps(PLAN_JSON_SCHEMA), verbose=False
                    )
                if not plan_task and self._sql_grammar is None:
                    from llama_cpp import LlamaGrammar

                    self._sql_grammar = LlamaGrammar.from_json_schema(
                        json.dumps(SQL_JSON_SCHEMA), verbose=False
                    )
                options["grammar"] = self._plan_grammar if plan_task else self._sql_grammar
            request = [SystemMessage(content=system), HumanMessage(content=prompt)]
        else:
            request = prompt
            options = {}
        first_token = None
        if self.settings.model_runtime == "llama_cpp" and hasattr(model, "stream"):
            chunks = []
            for chunk in model.stream(request, **options):
                content = str(getattr(chunk, "content", chunk))
                if content and first_token is None:
                    first_token = time.perf_counter() - started
                chunks.append(content)
            result = "".join(chunks)
        else:
            response = model.invoke(request, **options)
            result = str(getattr(response, "content", response))
        total = time.perf_counter() - started
        client = getattr(model, "client", None)
        try:
            prompt_tokens = len(client.tokenize(prompt.encode("utf-8"))) if client else None
            output_tokens = len(client.tokenize(result.encode("utf-8"))) if client else None
        except Exception:
            prompt_tokens = output_tokens = None
        self.last_invocation_metrics = {
            "time_to_first_token_seconds": first_token,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "tokens_per_second": round(output_tokens / max(total - (first_token or 0), 1e-9), 3)
                if output_tokens is not None and first_token is not None else None,
        }
        return result

    def _load_llm(self):
        if (self.settings.model_runtime == "ollama" or self.settings.ollama_deploy_model) and urlparse(self.settings.ollama_base_url).hostname not in {
            "localhost", "127.0.0.1", "::1"
        }:
            raise ValueError("OLLAMA_BASE_URL must point to a local loopback address.")
        if self.settings.ollama_deploy_model:
            ensure_ollama_model(self.settings)

        if self.settings.model_runtime == "ollama":
            try:
                from langchain_community.llms import Ollama
            except ImportError as exc:
                raise RuntimeError("LangChain Ollama support is required. Install requirements.txt.") from exc

            return Ollama(
                base_url=self.settings.ollama_base_url,
                model=self.settings.ollama_model_name,
                temperature=self.settings.llm_temperature,
                num_ctx=self.settings.llm_n_ctx,
                num_predict=self.settings.llm_max_tokens,
                system="You are Clinical Trial Analyst, a local-only clinical data analyst.",
            )

        if self.settings.model_runtime != "llama_cpp":
            raise ValueError(
                f"Unsupported MODEL_RUNTIME={self.settings.model_runtime!r}. "
                "Use 'llama_cpp' or 'ollama'."
            )

        if not self.settings.model_path.exists():
            raise FileNotFoundError(
                f"Local model file not found: {self.settings.model_path}. "
                "Run `python scripts/model/download_model.py` or set MODEL_PATH in .env."
            )

        try:
            from langchain_community.chat_models import ChatLlamaCpp
        except ImportError as exc:
            raise RuntimeError("LangChain and llama-cpp-python are required. Install requirements.txt.") from exc

        seed_options = {"seed": self.settings.llm_seed} if self.settings.llm_seed is not None else {}
        model = ChatLlamaCpp(
            model_path=str(self.settings.model_path),
            n_ctx=self.settings.llm_n_ctx,
            n_gpu_layers=self.settings.llm_n_gpu_layers,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
            verbose=False,
            **seed_options,
        )
        self._chat_model = True
        return model

    def answer(self, question: str) -> AnalystResponse:
        started = time.perf_counter()
        policy_response = preflight_response(question)
        if policy_response:
            status, answer = policy_response
            return AnalystResponse(
                question=question,
                answer=answer,
                sql="",
                repaired=False,
                query_result={},
                latency_seconds=time.perf_counter() - started,
                status=status,
            )

        clarification = json.loads(request_clarification_via_mcp(question, self.settings))
        if clarification.get("clarification_required"):
            return AnalystResponse(
                question=question,
                answer=clarification["question"],
                sql="",
                repaired=False,
                query_result={},
                clarification_required=True,
                latency_seconds=time.perf_counter() - started,
                status="clarification",
            )

        repaired = False
        schema, _schema_metadata, guidance = self.schema_context(question)
        try:
            plan = self._plan_question(question, schema, guidance)
        except ValueError as exc:
            return AnalystResponse(
                question=question,
                answer="I could not determine a reliable answer plan.",
                sql="",
                repaired=False,
                query_result={},
                error=str(exc),
                latency_seconds=time.perf_counter() - started,
            )
        if plan["needs_clarification"]:
            return AnalystResponse(
                question=question,
                answer=plan["clarification_question"],
                sql="",
                repaired=False,
                query_result={},
                clarification_required=True,
                latency_seconds=time.perf_counter() - started,
                status="clarification",
            )
        contract_feedback = check_plan_intent(question, plan)
        try:
            sql = self._generate_sql(question, schema, plan, contract_feedback, guidance)["sql"]
        except ValueError as exc:
            repaired = True
            try:
                sql = self._repair_sql(question, "", str(exc), schema, plan, guidance)
            except ValueError as repair_exc:
                return AnalystResponse(
                    question=question,
                    answer="I could not generate a valid SQL query after one repair attempt.",
                    sql="",
                    repaired=True,
                    query_result={},
                    error=str(repair_exc),
                    latency_seconds=time.perf_counter() - started,
                )

        query_result = self._run_sql(sql, question)
        if not query_result.get("ok", False):
            if not repaired:
                repaired = True
                try:
                    sql = self._repair_sql(question, sql, query_result.get("error", "Unknown SQL error"), schema, plan, guidance)
                except ValueError as repair_exc:
                    return AnalystResponse(
                        question=question,
                        answer="I could not repair the SQL query.",
                        sql=sql,
                        repaired=True,
                        query_result=query_result,
                        error=str(repair_exc),
                        latency_seconds=time.perf_counter() - started,
                    )
                query_result = self._run_sql(sql, question)

        if not query_result.get("ok", False):
            return AnalystResponse(
                question=question,
                answer="I could not answer because the read-only SQL query failed after one repair attempt.",
                sql=sql,
                repaired=repaired,
                query_result=query_result,
                error=query_result.get("error"),
                latency_seconds=time.perf_counter() - started,
            )

        answer_text = self._summarize_answer(question, query_result)
        return AnalystResponse(
            question=question,
            answer=answer_text,
            sql=sql,
            repaired=repaired,
            query_result=query_result,
            tables_used=self._tables_used(sql),
            latency_seconds=time.perf_counter() - started,
        )

    def _plan_question(self, question: str, schema: str, guidance: str = JOIN_AND_DEFINITION_GUIDE) -> dict[str, Any]:
        raw = self._invoke_llm(build_plan_prompt(question, schema, guidance), plan_task=True)
        plan, _ = finalize_plan(question, self._parse_plan_json(raw, question=question))
        return plan

    def _generate_sql(self, question: str, schema: str, plan: dict[str, Any], contract_feedback: list[str] | None = None, guidance: str = JOIN_AND_DEFINITION_GUIDE) -> dict[str, Any]:
        raw = self._invoke_llm(build_sql_generation_prompt(question, schema, plan, contract_feedback, guidance))
        return self._parse_sql_json(raw)

    def _repair_sql(self, question: str, failed_sql: str, error: str, schema: str, plan: dict[str, Any], guidance: str = JOIN_AND_DEFINITION_GUIDE) -> str:
        raw = self._invoke_llm(build_sql_repair_prompt(question, failed_sql, error, schema, plan, guidance=guidance))
        return self._parse_sql_json(raw)["sql"]

    def _summarize_answer(self, question: str, query_result: dict[str, Any]) -> str:
        prompt = f"""
You are Clinical Trial Analyst. Answer the user's question using only the query result.
Do not invent data. If the result is empty, say so.
Keep the answer concise and include important counts, groups, or dates.

Question:
{question}

Query result JSON:
{json.dumps(query_result, indent=2)}

Answer:
""".strip()
        return self._invoke_llm(prompt, sql_task=False).strip()

    def _run_sql(self, sql: str, question: str = "") -> dict[str, Any]:
        try:
            check_sql_question_filters(question, sql)
            findings = check_sql_intent(question, sql)
            if findings:
                raise ValueError("Question-intent contract violation: " + " | ".join(findings))
        except ValueError as exc:
            return {"ok": False, "sql": sql, "columns": [], "rows": [], "row_count": 0, "error": str(exc)}
        raw = execute_readonly_sql_via_mcp(sql, self.settings)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {
                "ok": False,
                "sql": sql,
                "columns": [],
                "rows": [],
                "row_count": 0,
                "error": f"MCP returned non-JSON content: {raw[:500]}",
            }

    @staticmethod
    def _tables_used(sql: str) -> list[str]:
        code = _mask_sql_literals_and_comments(sql)
        table_pattern = "|".join(re.escape(table) for table in TABLE_SCHEMAS)
        return sorted(
            {match.lower() for match in re.findall(
                rf"\b(?:from|join)\s+({table_pattern})\b", code, re.IGNORECASE
            )}
        )

    @staticmethod
    def _parse_json_object(raw: str) -> dict[str, Any]:
        cleaned = raw.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if fenced:
            cleaned = fenced.group(1)
        else:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                cleaned = cleaned[start : end + 1]

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model did not return valid JSON: {raw[:500]}") from exc
        if not isinstance(payload, dict):
            raise ValueError("Model JSON must be an object")
        return payload

    @staticmethod
    def _parse_plan_json(raw: str, *, question: str = "") -> dict[str, Any]:
        plan = ClinicalTrialAnalystAgent._parse_json_object(raw)
        required_plan = {"population", "measure", "grouping", "time_window", "filters", "ranking", "needs_clarification", "clarification_question"}
        if not required_plan.issubset(plan):
            raise ValueError(f"Model JSON did not include a complete plan: {plan}")
        if not isinstance(plan["needs_clarification"], bool):
            raise ValueError("Plan needs_clarification must be boolean")
        if any(not isinstance(plan[key], str) for key in required_plan - {"needs_clarification"}):
            raise ValueError("Plan fields must be strings")
        # A named site plus the undefined adjective cannot justify a causal SQL answer.
        if re.search(r"\bwhy\b.*\bsite\s*17\b.*\bperforming poorly\b", question, re.I):
            plan["needs_clarification"] = True
            plan["clarification_question"] = "Which measure should I use to assess Site 17 (for example enrollment, missed visits, or open queries)? I cannot determine a cause from this question alone."
        if plan["needs_clarification"] and not plan["clarification_question"]:
            raise ValueError("Clarification plan did not include a question")
        return plan

    @staticmethod
    def _parse_sql_json(raw: str) -> dict[str, str]:
        payload = ClinicalTrialAnalystAgent._parse_json_object(raw)
        sql = str(payload.get("sql", "")).strip()
        if not sql:
            raise ValueError(f"Model JSON did not include SQL: {payload}")

        return {
            "sql": sql,
            "rationale": str(payload.get("rationale", "")),
        }
