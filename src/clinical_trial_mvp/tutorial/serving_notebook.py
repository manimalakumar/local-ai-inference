"""Chapter 00: local serving, context and structured-answer quality.

Teaching Markdown lives in chapter_prose.py. Runtime implementations are shared.
"""
FILENAME = '00_local_serving.ipynb'

CELLS = [
    ('markdown', '7b5e354b23e7', ''),
    ('code', 'c73045dd566e', "from pathlib import Path\nimport os\nimport sys\nROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'src/clinical_trial_mvp').is_dir())\nsys.path.insert(0, str(ROOT / 'src'))\nfrom IPython.display import JSON, Markdown, display\nfrom clinical_trial_mvp.tutorial.foundation import inventory, contexts, mac_history\nfrom clinical_trial_mvp.tutorial.foundation_study import FoundationStudy\nstudy = FoundationStudy(ROOT)\nprint('Chapter 00 mode:', study.mode)\nprint('Live: run embedding inference and call your local model server.' if study.mode == 'live' else\n      'Replay: display the supplied results.' if study.mode == 'replay' else\n      'Plan: inspect the configuration.')\n"),
    ('markdown', 'af1ac4f75873', ''),
    ('markdown', 'b937d0a634e6', ''),
    ('code', '786a85ecedf7', 'database = inventory(ROOT)\ndisplay(JSON(database))\n'),
    ('markdown', 'ac6979b6039b', ''),
    ('code', '5472cf0d7885', "context = contexts(ROOT)\ndisplay(JSON({'note': context['note'], 'cases': [\n    {key: value for key, value in row.items() if key not in ('full_prompt', 'retrieved_prompt')}\n    for row in context['rows']]}))\ndisplay(JSON({'full_prompt': context['rows'][0]['full_prompt'],\n              'retrieved_prompt': context['rows'][0]['retrieved_prompt']}))\n"),
    ('markdown', 'embedding-invocation-explanation', ''),
    ('code', 'embedding-model-call', "# Fresh local embedding inference; no chat-completions API is used here.\nif study.mode == 'live':\n    from sentence_transformers import SentenceTransformer\n    from clinical_trial_mvp.data.embedding_retrieval import SchemaRetriever\n\n    embedding_path = Path(study.config.get(\n        'embedding_model_path', 'models/Nemotron-3-Embed-1B-BF16')).expanduser()\n    if not embedding_path.is_absolute():\n        embedding_path = ROOT / embedding_path\n    if not embedding_path.is_dir():\n        raise FileNotFoundError('Download the embedding model first: see SETUP.md#notebook-00-embedding-model')\n    embedding_model = SentenceTransformer(\n        str(embedding_path), device='cpu', local_files_only=True, trust_remote_code=False)\n\n    class NotebookEmbedder:\n        def encode(self, texts):\n            # Inputs already carry passage: or query: prefixes from SchemaRetriever.\n            return embedding_model.encode(\n                texts, batch_size=8, normalize_embeddings=True,\n                show_progress_bar=False).tolist()\n\n    # Construction embeds schema facts; retrieve() embeds the question.\n    retriever = SchemaRetriever(NotebookEmbedder(), top_k=12, max_seed_tables=4)\n    embedding_question = context['rows'][0]['question']\n    selected_context = retriever.retrieve(embedding_question)\n    display(Markdown('**Question:** ' + embedding_question))\n    display(JSON({\n        'embedding_model': 'nvidia/Nemotron-3-Embed-1B-BF16',\n        'selected_tables': selected_context.tables,\n        'ranked_facts': [\n            {'fact': fact.text, 'similarity': selected_context.scores[fact.fact_id]}\n            for fact in sorted(\n                (fact for fact in retriever.facts if fact.fact_id in selected_context.scores),\n                key=lambda fact: selected_context.scores[fact.fact_id], reverse=True)],\n    }))\n    print('Retrieved schema text:')\n    print(selected_context.prompt)\nelse:\n    print('Select Live mode to invoke the embedding model; Replay uses the supplied selection.')\n"),
    ('markdown', 'first-api-explanation', ''),
    ('code', 'first-model-api-call', "# First model inference: an OpenAI-compatible chat-completions request.\nimport json\nimport urllib.request\nfrom clinical_trial_mvp.agent.analyst import (\n    PLAN_CHAT_SYSTEM_MESSAGE, PLAN_JSON_SCHEMA, build_plan_prompt,\n)\nfrom clinical_trial_mvp.serving.benchmark import Workload, prepare\nfrom clinical_trial_mvp.serving.client import Generation, OpenAIClient\n\nif study.mode == 'live':\n    endpoint = study.config['servers'][study.config['context_server']]\n    if endpoint['runtime'] not in {'llama_cpp', 'vllm', 'sglang'}:\n        raise ValueError('Select an OpenAI-compatible endpoint for this example; see SETUP.md.')\n    cases, schema_contexts, _, _ = prepare(\n        ROOT, Workload(profile='full'), study.config['case_ids'])\n    first_question = cases[0]['question']\n    schema_text, _, definitions = schema_contexts[first_question]\n    messages = [\n        {'role': 'system', 'content': PLAN_CHAT_SYSTEM_MESSAGE},\n        {'role': 'user', 'content': build_plan_prompt(first_question, schema_text, definitions)},\n    ]\n    # The same request settings and local-only transport as the benchmark.\n    api = OpenAIClient(endpoint['base_url'], endpoint['model'],\n        Generation(max_tokens=study.config['max_tokens'], seed=42),\n        request_profile=endpoint['runtime'])\n    api_url = api.base_url + '/chat/completions'\n    payload = api.request_payload(messages, PLAN_JSON_SCHEMA)\n    payload['stream'] = False\n    payload.pop('stream_options', None)\n    display(Markdown('**Question:** ' + first_question))\n    print('POST', api_url)\n    display(JSON(payload))\n\n    # This line contacts the running model server. It does not load a saved answer.\n    request = urllib.request.Request(\n        api_url, data=json.dumps(payload).encode('utf-8'),\n        headers={'Content-Type': 'application/json'}, method='POST')\n    with api.opener(request, timeout=180) as response:\n        api_response = json.load(response)\n    choice = api_response['choices'][0]\n    if choice.get('finish_reason') != 'stop':\n        raise RuntimeError('Model response did not finish normally: ' + str(choice.get('finish_reason')))\n    model_reply = choice['message']['content']\n    if not isinstance(model_reply, str) or not model_reply.strip():\n        raise RuntimeError('The model returned an empty answer plan.')\n    print('Model response:')\n    print(model_reply)\n    if api_response.get('usage'):\n        display(JSON(api_response['usage']))\nelse:\n    print('Select Live mode and start the server to execute this API example (see SETUP.md).')\n"),
    ('markdown', '0a756f9a063f', ''),
    ('code', 'e73102b7d644', "if study.mode == 'replay':\n    from clinical_trial_mvp.tutorial.foundation import context_results_table\n    history = mac_history(ROOT)\n    print('Loading results.')\n    display(Markdown(context_results_table(history['context'])))\nelse:\n    context_runs = study.context_rows()\n    display(Markdown(study.table(context_runs)))\n"),
    ('markdown', 'dd5f0ec80102', ''),
    ('markdown', '90c6cb614df5', ''),
    ('code', '310ff65dd9d8', "if study.mode == 'replay':\n    from clinical_trial_mvp.tutorial.foundation import local_serving_results_table\n    print('Loading results.')\n    display(Markdown(local_serving_results_table(history['serving'])))\nelse:\n    local_runs = study.local_rows()\n    display(Markdown(study.table(local_runs)))\n"),
    ('markdown', '552eb017d529', ''),
    ('markdown', '18c915d61247', ''),
    ('code', '5476bdc88385', "from clinical_trial_mvp.tutorial.foundation import local_answer_example, structured_answer_markdown\nif study.mode == 'replay':\n    answer = local_answer_example(ROOT)\nelse:\n    answer = study.answer()\ndisplay(Markdown(structured_answer_markdown(answer)))\n"),
    ('markdown', 'db1301631036', ''),
    ('code', 'baseline-summary', "from clinical_trial_mvp.tutorial.foundation import baseline_results_summary\nbaseline_history = history if study.mode == 'replay' else None\ndisplay(Markdown(baseline_results_summary(study, baseline_history, answer)))\n"),
 ]


def notebook_document():
    cells=[]
    for kind, slot, source in CELLS:
        cell=dict(cell_type=kind,id=slot,metadata={},source=source.splitlines(keepends=True))
        if kind=='code':
            cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    document=dict(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.11'}},nbformat=4,nbformat_minor=5)
    from clinical_trial_mvp.tutorial.navigation import apply_chapter_prose
    return apply_chapter_prose(document,FILENAME)
