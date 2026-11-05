"""Use the inherited clinical plan/SQL/repair path with either GPU server."""

from clinical_trial_mvp.agent.analyst import (
    ClinicalTrialAnalystAgent, PLAN_CHAT_SYSTEM_MESSAGE, PLAN_JSON_SCHEMA,
    SQL_CHAT_SYSTEM_MESSAGE, SQL_JSON_SCHEMA, SUMMARY_CHAT_SYSTEM_MESSAGE,
)
from clinical_trial_mvp.serving.client import OpenAIClient


class ServingAnalystAgent(ClinicalTrialAnalystAgent):
    def __init__(self, settings, client: OpenAIClient, contexts: dict):
        super().__init__(settings)
        self.client = client
        self.contexts = contexts

    def schema_context(self, question):
        return self.contexts[question]

    def _invoke_llm(self, prompt, *, sql_task=True, plan_task=False):
        self.last_invocation_metrics = {}
        system = PLAN_CHAT_SYSTEM_MESSAGE if plan_task else (
            SQL_CHAT_SYSTEM_MESSAGE if sql_task else SUMMARY_CHAT_SYSTEM_MESSAGE)
        schema = PLAN_JSON_SCHEMA if plan_task else (SQL_JSON_SCHEMA if sql_task else None)
        result = self.client.complete([
            {"role": "system", "content": system}, {"role": "user", "content": prompt}], schema)
        self.last_invocation_metrics = result.to_dict()
        if not result.success:
            raise ValueError(result.error)
        return result.content
