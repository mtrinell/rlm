# Detective Agent

An RLM-powered behavioral derailment detector. Given a dataset of agent traces (file, directory, or archive) and an optional specification, it hunts for cases where an agent or component acted outside its mandate — accessing out-of-scope resources, acting on behalf of the wrong entity, or leaking data across request boundaries — and produces a structured **BEHAVIORAL DERAILMENT REPORT**.

## Getting started

### 1. Install dependencies

From the project root:

```bash
uv pip install -e .
```

### 2. Configure environment

Copy the example env file and fill in your values:

```bash
cp examples/detective_agent/.env.example examples/detective_agent/.env
```

Minimum required variables:

| Variable | Description |
|---|---|
| `LLM_API_KEY` | API key for your LLM provider |
| `LLM_MODEL` | Model identifier (e.g. `openai/gpt-5.4`) |
| `DATASET_PATH` | Path to the dataset to analyse |
| `SPEC_PATH` | Path to source code or docs describing the system's expected behaviour |

Optional:

| Variable | Description |
|---|---|
| `LLM_BASE_URL` | Custom base URL (e.g. a proxy) |
| `USER_PROMPT` | The original input that was given to the agent under test (e.g. `"Create a new GitHub project 'Agent-auditor'…"`). The detective uses this to verify that the agent did exactly what was asked — no more, no less — and files `DERAILMENT_USER` findings for anything it did that wasn't covered by this request. |
| `MAX_ITERATIONS` | Max RLM iterations (default: `20`) |
| `CONTEXT_TOKEN_LIMIT` | Token budget (default: `200000`) |
| `LOG_DIR` | Output directory (default: `results/`) |

### 3. Run

```bash
uv run python -m examples.detective_agent.main
```

Results are written to `results/` (or the directory set by `LOG_DIR`):
- `investigation_<dataset>_<timestamp>.txt` — full derailment report
- `SUMMARY_<dataset>_<timestamp>.txt` — summary with token usage and cost estimate
- `run_<dataset>_<timestamp>.log` — execution log

## How it works

The agent runs a 3-phase RLM loop:

1. **Phase 0 — Spec ingestion**: reads key files from the spec and synthesises a mandate reference for each component.
2. **Phase 1 — Setup**: explores the dataset, detects formats, extracts archives, and builds a `request→action` map (what was asked vs. what was actually done).
3. **Phase 2 — Derailment scan**: batch-analyses traces with `llm_query_batched`, flagging every action that accessed a resource or entity not justified by the request.

Findings are classified as **DERAILMENT**, **DERAILMENT_USER**, or **OK**, with an overall verdict of **DERAILED**, **DERAILED_USER**, or **COMPLIANT**.

## Supported dataset formats

Files, directories, or archives (`.zip`, `.tar.gz`) containing agent traces in:
JSON, JSONL, CSV, YAML, plain text / log files, and more.
The agent auto-detects the format and picks the appropriate reader.
