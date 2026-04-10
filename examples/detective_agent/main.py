"""
Detective Agent — behavioral derailment detection using Recursive Language Models.

Architecture:
  Input validation → DetectorREPL (dataset_path + optional spec injected)
  → ContextBudget + HistoryManager → RLM loop (budget-aware, phase injection) → results/

Use case:
  Given (1) a dataset — any file, directory, or archive in any format (logs, traces,
        events, CSVs, JSON, YAML, …), and
        (2) optionally a specification / documentation describing expected behaviour,
  the agent hunts for behavioral derailments: cases where an agent or component acted
  outside its mandate — accessing resources it was not asked to access, acting on behalf
  of the wrong entity, or leaking data across request boundaries.

  The LLM autonomously explores the dataset, builds a request→action map, and batch-
  analyses traces to produce a structured BEHAVIORAL DERAILMENT REPORT with per-finding
  classifications (DERAILMENT / DERAILMENT_USER / OK) and an overall verdict
  (DERAILED / DERAILED_USER / COMPLIANT).

  File-access helpers available in the REPL: list_files, read_file, read_json,
  read_jsonl, read_csv, detect_format, extract_archive, search_lines, …
"""

import logging
import os
import sys
import traceback
import uuid
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from examples.detective_agent.config import settings
from examples.detective_agent.context import ContextBudget, HistoryManager
from examples.detective_agent.core import RLM
from examples.detective_agent.log_utils import setup_logging
from examples.detective_agent.prompts import build_detector_task_prompt, get_detector_system_prompt

try:
    from examples.detective_agent.logger.rlm_logger import RLMLogger
except ImportError:
    RLMLogger = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


def _save_results(
    result_dir: Path,
    dataset_name: str,
    run_stamp: str,
    response: str,
    completed: bool,
    exec_time: float,
    iterations_used: int,
    max_iterations: int,
    usage_summary: object,
) -> tuple[Path, Path]:
    """Write investigation result and summary files. Returns (output_path, summary_path)."""
    content = response
    if not completed:
        content = f"INCOMPLETE ANALYSIS — hit MAX_ITERATIONS\n{'=' * 80}\n\n{content}"

    output_file = result_dir / f"investigation_{dataset_name}_{run_stamp}.txt"
    output_file.write_text(content)

    summary_file = result_dir / f"SUMMARY_{dataset_name}_{run_stamp}.txt"
    with open(summary_file, "w") as f:
        f.write(f"{'=' * 80}\nANALYSIS REPORT SUMMARY: {dataset_name}\n{'=' * 80}\n\n")
        f.write(f"Status: {'COMPLETED' if completed else 'INCOMPLETE (hit MAX_ITERATIONS)'}\n")
        f.write(f"Execution Time: {exec_time:.2f}s\n")
        f.write(f"Iterations Used: {iterations_used}/{max_iterations}\n")

        usage = usage_summary.to_dict() if hasattr(usage_summary, "to_dict") else {}
        for model_name, stats in usage.get("model_usage_summaries", {}).items():
            in_tok = stats.get("total_input_tokens", 0)
            out_tok = stats.get("total_output_tokens", 0)
            cost = in_tok * 2.5e-6 + out_tok * 10e-6
            f.write(f"Cost: ~${cost:.4f} ({in_tok:,} in + {out_tok:,} out tokens) [{model_name}]\n")

        f.write(f"\n{'-' * 80}\nRESULTS:\n{'-' * 80}\n\n{content}")

    return output_file, summary_file


# ─── Main entry point ─────────────────────────────────────────────────────────


def main() -> None:
    """Run a single dataset analysis investigation."""
    dataset_path = settings.resolved_dataset_path
    spec_path = settings.resolved_spec_path
    result_dir = settings.resolved_log_dir
    result_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = dataset_path.stem
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + str(uuid.uuid4())[:8]
    log_file = result_dir / f"run_{dataset_name}_{run_stamp}.log"
    run_logger = setup_logging(log_file)

    run_logger.info("=" * 80)
    run_logger.info("Dataset Analysis Agent")
    run_logger.info("=" * 80)
    run_logger.info(
        f"  Dataset:        {dataset_path}\n"
        f"  Spec:           {spec_path}\n"
        f"  Max Iterations: {settings.max_iterations}\n"
        f"  Context Limit:  {settings.context_token_limit:,} tokens\n"
    )
    if settings.investigation_focus:
        run_logger.info(f"  Focus:          {settings.investigation_focus}")
    if settings.user_prompt:
        run_logger.info(f"  User Prompt:    {settings.user_prompt[:120]}{'...' if len(settings.user_prompt) > 120 else ''}")

    # ── Validate inputs ────────────────────────────────────────────────────────
    if not dataset_path.exists():
        run_logger.error(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    run_logger.info("\n[1/4] Loading inputs...")

    if not spec_path.exists():
        run_logger.error(f"Spec path not found: {spec_path}")
        sys.exit(1)
    if spec_path.is_dir():
        file_count = sum(1 for _ in spec_path.rglob("*") if _.is_file())
        run_logger.info(f"  Spec:    directory with {file_count} files at {spec_path}")
    else:
        run_logger.info(f"  Spec:    {spec_path.stat().st_size / 1024:.1f} KB from {spec_path.name}")

    if dataset_path.is_dir():
        file_count = sum(1 for _ in dataset_path.rglob("*") if _.is_file())
        run_logger.info(f"  Dataset: directory with {file_count} files from {dataset_path}")
    else:
        run_logger.info(f"  Dataset: {dataset_path.stat().st_size / 1024:.1f} KB from {dataset_path}")

    try:
        run_logger.info("\n[2/4] Initialising RLM...")

        budget = ContextBudget(total_token_limit=settings.context_token_limit)
        history_manager = HistoryManager()

        rlm_logger = None
        if RLMLogger is not None:
            rlm_logger = RLMLogger(log_dir=str(result_dir), run_stamp=run_stamp)

        try:
            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": os.getenv("LLM_MODEL")},
                environment="detector",
                environment_kwargs={
                    "dataset_path": str(dataset_path),
                    "spec_path": str(spec_path),
                    "user_prompt": settings.user_prompt,
                    "budget": budget,
                    "inject_helpers": settings.helpers_injection_enabled,
                },
                max_iterations=settings.max_iterations,
                max_depth=settings.max_depth,
                custom_system_prompt=get_detector_system_prompt(),
                logger=rlm_logger,
                verbose=settings.verbose,
                budget=budget,
                history_manager=history_manager,
                phase_injection_enabled=settings.phase_injection_enabled,
            )
        except Exception:
            run_logger.exception("Failed to initialise RLM")
            run_logger.exception(traceback.format_exc())
            sys.exit(1)

        run_logger.info(f"\n[3/4] Running analysis (max {settings.max_iterations} iterations)...")
        run_logger.info("-" * 80)

        task_prompt = build_detector_task_prompt(
            dataset_path=str(dataset_path),
            dataset_name=dataset_name,
            max_iterations=settings.max_iterations,
            spec_path=str(spec_path),
            user_prompt=settings.user_prompt,
        )

        prompt_context: dict = {
            "dataset_path": str(dataset_path),
            "dataset_name": dataset_name,
            "spec_path": str(spec_path),
        }
        if settings.investigation_focus:
            prompt_context["investigation_focus"] = settings.investigation_focus
        if settings.user_prompt:
            prompt_context["user_prompt"] = settings.user_prompt

        try:
            result = rlm.completion(prompt=prompt_context, root_prompt=task_prompt)
        except Exception:
            run_logger.exception("Analysis failed")
            run_logger.exception(traceback.format_exc())
            sys.exit(1)
        finally:
            rlm.close()

        run_logger.info("\n[4/4] Saving results...")

        iterations_used = min(
            rlm_logger.iteration_count if rlm_logger else settings.max_iterations,
            settings.max_iterations,
        )
        completed = iterations_used < settings.max_iterations

        output_file, summary_file = _save_results(
            result_dir=result_dir,
            dataset_name=dataset_name,
            run_stamp=run_stamp,
            response=result.response,
            completed=completed,
            exec_time=result.execution_time,
            iterations_used=iterations_used,
            max_iterations=settings.max_iterations,
            usage_summary=result.usage_summary,
        )

        run_logger.info("=" * 80)
        run_logger.info("=" * 80)
        run_logger.info(f"Results:    {output_file}")
        run_logger.info(f"Summary:    {summary_file}")
        run_logger.info(f"Time:       {result.execution_time:.2f}s")
        run_logger.info(f"Iterations: {iterations_used}/{settings.max_iterations}")

        if budget:
            snap = budget.snapshot()
            run_logger.info(
                f"Context:    {snap['percent_used']:.1f}% ({snap['used_tokens']:,} / {snap['total_tokens']:,} tokens)",
            )

        if not completed:
            run_logger.warning("Analysis hit MAX_ITERATIONS — results may be incomplete.")

    finally:
        pass  # No temp dirs to clean up (extraction happens inside the REPL if needed)


if __name__ == "__main__":
    try:
        main()
    except ValidationError as e:
        for error in e.errors():
            field = " -> ".join(str(loc) for loc in error["loc"])
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)

