# RLM Engine

Root Cause Analysis engine using Recursive Language Models (RLMs).

## Overview

Reimplementation of the a3po RCA engine using [Recursive Language Models](https://github.com/alexzhang13/rlm). A single LM self-programs its investigation strategy through code execution in a REPL environment, replacing the multi-agent LangGraph architecture.

| Aspect | Original (LangGraph) | RLM Engine |
|--------|---------------------|------------|
| Architecture | 15 specialized agents in state machine | Single LM with REPL environment |
| Control Flow | Fixed graph with explicit routing | LM decides strategy dynamically |
| Context Handling | Passed in prompts (token limited) | Loaded in REPL (unlimited) |

## Running the RLM Engine

From the **a3po-engine** project root:

```bash
# Using uv (recommended)
uv run python -m src.rlm.main

# Or with an activated virtualenv
source .venv/bin/activate
python -m src.rlm.main
```

### Environment Variable Overrides

All settings in `.env` can be overridden inline:

```bash
MAX_ITERATIONS=30 TEST_DATASET=/absolute/path/to/dataset.tar uv run python -m src.rlm.main
```

## Configuration

The RLM engine reads its configuration from the a3po-engine root `.env` file via Pydantic Settings (`config.py`).

### Required Environment Variables

| Variable | Description |
|----------|-------------|
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |

### Optional Environment Variables (with defaults)

| Variable | Default | Description |
|----------|---------|-------------|
| `AZURE_OPENAI_API_VERSION` | `2024-02-15-preview` | API version |
| `AZURE_OPENAI_MODEL` | `gpt-4o` | Model deployment name |
| `MAX_ITERATIONS` | `20` | Top-level LLM iteration limit (each = 1 LLM call + code execution) |
| `MAX_DEPTH` | `1` | Recursive sub-call depth limit (`1` = single model, cheapest) |
| `CONTEXT_TOKEN_LIMIT` | `100000` | Token budget ceiling; truncation tiers activate at 50/70/85% |
| `TEST_DATASET` | *(see config.py)* | Path to test dataset tarball |
| `PROBLEM_STATEMENT` | *(see config.py)* | Problem description for the RCA investigation |
| `LOG_DIR` | `results` | Directory for JSONL iteration logs and result files |
| `VERBOSE` | `true` | Enable rich console output |

## Project Structure

```
src/rlm/
├── main.py              # Entry point
├── config.py            # Pydantic Settings configuration
├── preprocessor.py      # Tarball extraction and file manifest
├── logging.py           # Log setup
├── clients/             # LLM clients (Azure OpenAI, Anthropic, LiteLLM, OpenAI)
│   ├── base_lm.py       # Base client interface
│   ├── azure_openai.py  # Azure OpenAI client
│   ├── anthropic_client.py
│   ├── litellm_client.py
│   └── openai_client.py
├── core/                # RLM loop, LM handler, parsing, types
│   ├── rlm_loop.py      # Main recursive loop
│   ├── lm_handler.py    # LLM interaction handler
│   ├── parsing.py       # Response parser
│   ├── rlm_utils.py     # Utilities
│   ├── comms_utils.py   # Communication helpers
│   ├── base_prompts.py  # Prompt templates
│   └── types.py         # Type definitions
├── context/             # Context management
│   ├── budget.py        # ContextBudget — 4-tier token truncation
│   └── history_manager.py  # Compresses old REPL outputs under pressure
├── environment/         # REPL environments
│   ├── base_env.py      # Base environment
│   ├── local_repl.py    # Local REPL
│   ├── rca_repl.py      # RCA-specific REPL with injected helpers
│   └── helpers.py       # Filesystem helpers injected into REPL
├── logger/              # RLM iteration logger
│   ├── rlm_logger.py    # JSONL iteration logging
│   └── verbose.py       # Rich console output
└── prompts/             # RCA system prompt
    └── system_prompt.py # Domain-expert prompt with methodology
```

## How It Works

1. **Preprocessing** — `Preprocessor` extracts the tarball into a temp directory and builds a file manifest (path, size, line count) so the LM doesn't waste an iteration on extraction.
2. **RLM Initialization** — An `RcaREPL` environment is created with filesystem helpers and the file manifest injected into the REPL namespace. `ContextBudget` tracks token usage with 4-tier truncation (50/70/85/100%).
3. **Investigation Loop** — The LM reads files, analyzes logs and configs, runs code in the REPL, and builds its understanding incrementally. Phase-injection inserts budget-pressure reminders at key milestones.
4. **Output** — The final result and a summary are written to `results/`.

## Troubleshooting

| Error | Fix |
|-------|-----|
| `Tarball not found` | Check `TEST_DATASET` in `.env` — use an absolute path if relative resolution fails |
| `AZURE_OPENAI_API_KEY not set` | Ensure `AZURE_OPENAI_API_KEY` is set in the root `.env` |
| Investigation terminates early | Increase `MAX_ITERATIONS` in `.env` |
| `ModuleNotFoundError` | Run `uv sync` to install RLM dependencies (`openai`, `anthropic`, `litellm`, `pydantic-settings`, `rich`) |

## References

- [RLM Paper](https://arxiv.org/abs/2512.24601)
- [RLM Framework](https://github.com/alexzhang13/rlm)
