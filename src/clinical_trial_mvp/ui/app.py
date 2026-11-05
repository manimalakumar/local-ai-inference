"""Render the Streamlit interface for asking clinical operations questions."""

from __future__ import annotations

import json
import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from clinical_trial_mvp.agent.analyst import ClinicalTrialAnalystAgent, DEFAULT_QUESTION
from clinical_trial_mvp.config import Settings
from clinical_trial_mvp.data.db import database_status
from clinical_trial_mvp.data.schema import SCHEMA_PROMPT


def main() -> None:
    st.set_page_config(page_title="Clinical Trial Analyst", layout="wide")
    st.title("Clinical Trial Analyst")

    settings = Settings.from_env()
    status = database_status(settings)

    with st.sidebar:
        st.subheader("Local Runtime")
        st.write(f"Database: `{status['db_path']}`")
        st.write("DB ready" if status["db_exists"] else "DB missing")
        st.write(f"Runtime: `{status['model_runtime']}`")
        st.write(f"Model: `{status['model_path']}`")
        st.write("Model ready" if status["model_exists"] else "Model missing")
        st.write(f"Ollama model: `{status['ollama_model_name']}`")
        st.write(f"Ollama deploy on start: `{status['ollama_deploy_model']}`")
        st.write(f"Row limit: `{settings.sql_result_row_limit}`")

        with st.expander("Schema prompt"):
            st.code(SCHEMA_PROMPT)

    question = st.text_area("Question", value=DEFAULT_QUESTION, height=100)
    run = st.button("Ask analyst", type="primary")

    if not run:
        return

    if not status["db_exists"]:
        st.error("DuckDB database is missing. Run `python scripts/data/build_db.py`.")
        return

    if status["model_runtime"] == "llama_cpp" and not status["model_exists"]:
        st.error("Local GGUF model is missing. Run `python scripts/model/download_model.py` or set MODEL_PATH.")
        return

    try:
        with st.spinner("Running local analyst agent..."):
            agent = ClinicalTrialAnalystAgent(settings)
            response = agent.answer(question)
    except Exception as exc:
        st.exception(exc)
        return

    st.subheader("Answer")
    st.write(response.answer)
    if response.error:
        st.error(response.error)

    if response.status != "answer":
        if response.clarification_required:
            st.caption("Please add your definition to the question and ask again.")
        return

    st.subheader("SQL executed" if response.query_result.get("ok") else "SQL attempted")
    if response.repaired:
        st.caption("The first query failed; one repair attempt was used.")
    st.code(response.query_result.get("sql", response.sql), language="sql")
    st.write(f"Tables used: {', '.join(response.tables_used) if response.tables_used else 'None'}")
    st.write(f"Repair occurred: {'Yes' if response.repaired else 'No'}")
    st.write(f"Latency: {response.latency_seconds:.2f} seconds")

    st.subheader("Query Result")
    result = response.query_result
    if result.get("rows"):
        st.dataframe(result["rows"], use_container_width=True)
    else:
        st.json(result)

    with st.expander("Raw result JSON"):
        st.code(json.dumps(result, indent=2), language="json")


if __name__ == "__main__":
    main()
