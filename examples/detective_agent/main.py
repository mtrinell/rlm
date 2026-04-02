"""
A3PO RLM Engine — Root Cause Analysis using Recursive Language Models.

Architecture:
  Preprocessor → RcaREPL (with injected helpers) → ContextBudget + HistoryManager
  → RLM loop (budget-aware, phase injection) → results/

Key improvements over the original prototype:
  1. Tarball is pre-extracted by Preprocessor before the loop starts (no wasted iteration 1).
  2. RcaREPL injects filesystem helpers + file manifest into the REPL namespace.
  3. ContextBudget tracks token usage with 4-tier truncation (50/70/85/100%).
  4. HistoryManager compresses old REPL outputs when pressure builds.
  5. Phase-injection: budget-pressure reminders and plan reminders at key milestones.
  6. Domain-expert system prompt with Frequency≠Importance and state-change methodology.
"""

import logging
import sys
import traceback
from pathlib import Path

from pydantic import ValidationError

from examples.detective_agent.config import settings
from examples.detective_agent.context import ContextBudget, HistoryManager
from examples.detective_agent.core import RLM
from examples.detective_agent.log_utils import setup_logging
from examples.detective_agent.preprocessor import Preprocessor
from examples.detective_agent.prompts import build_rca_task_prompt, get_rca_system_prompt

try:
    from examples.detective_agent.logger.rlm_logger import RLMLogger
except ImportError:
    RLMLogger = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


def _save_results(
    result_dir: Path,
    dataset_name: str,
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
        content = f"INCOMPLETE INVESTIGATION — hit MAX_ITERATIONS\n{'=' * 80}\n\n{content}"

    output_file = result_dir / f"investigation_{dataset_name}.txt"
    output_file.write_text(content)

    summary_file = result_dir / f"SUMMARY_{dataset_name}.txt"
    with open(summary_file, "w") as f:
        f.write(f"{'=' * 80}\nRCA INVESTIGATION SUMMARY: {dataset_name}\n{'=' * 80}\n\n")
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
    """Run a single RCA investigation."""
    tarball_path = settings.resolved_tarball
    result_dir = settings.resolved_log_dir
    result_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = tarball_path.stem
    log_file = result_dir / f"run_{dataset_name}.log"
    run_logger = setup_logging(log_file)

    run_logger.info("=" * 80)
    run_logger.info("A3PO RLM Engine — Root Cause Analysis")
    run_logger.info("=" * 80)
    run_logger.info(
        f"  Tarball:        {tarball_path}\n"
        f"  Problem:        {settings.problem_statement}\n"
        f"  Max Iterations: {settings.max_iterations}\n"
        f"  Context Limit:  {settings.context_token_limit:,} tokens\n"
        f"  Model:          {settings.azure_openai_model}",
    )

    if not tarball_path.exists():
        run_logger.error(f"Tarball not found: {tarball_path}")
        sys.exit(1)

    preprocessor = Preprocessor(tarball_path)
    if not preprocessor.validate():
        run_logger.error(f"Invalid or corrupt tarball: {tarball_path}")
        sys.exit(1)

    run_logger.info("\n[1/4] Extracting tarball...")
    try:
        prep_result = preprocessor.run()
    except Exception:
        run_logger.exception("Extraction failed")
        sys.exit(1)

    run_logger.info(f"  {prep_result.summary}")

    try:
        run_logger.info("\n[2/4] Initialising RLM...")

        budget = ContextBudget(total_token_limit=settings.context_token_limit)
        history_manager = HistoryManager()

        rlm_logger = None
        if RLMLogger is not None:
            rlm_logger = RLMLogger(log_dir=str(result_dir))

        try:
            rlm = RLM(
                backend="azure_openai",
                backend_kwargs=settings.backend_kwargs,
                environment="rca",
                environment_kwargs={
                    "files_path": str(prep_result.extract_dir),
                    "file_manifest": prep_result.file_manifest,
                },
                max_iterations=settings.max_iterations,
                max_depth=settings.max_depth,
                custom_system_prompt=get_rca_system_prompt(),
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

        run_logger.info(f"\n[3/4] Running investigation (max {settings.max_iterations} iterations)...")
        run_logger.info("-" * 80)

        task_prompt = build_rca_task_prompt(settings.problem_statement, settings.max_iterations)

        prompt_context = {
            "problem_statement": settings.problem_statement,
            "files_path": str(prep_result.extract_dir),
            "dataset_name": dataset_name,
            "file_count": prep_result.total_files,
            "total_size_mb": round(prep_result.total_size_bytes / 1024 / 1024, 2),
        }

        try:
            result = rlm.completion(prompt=prompt_context, root_prompt=task_prompt)
        except Exception:
            run_logger.exception("Investigation failed")
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
            response=result.response,
            completed=completed,
            exec_time=result.execution_time,
            iterations_used=iterations_used,
            max_iterations=settings.max_iterations,
            usage_summary=result.usage_summary,
        )

        run_logger.info("=" * 80)
        run_logger.info("=" * 80)
        run_logger.info(f"Results:   {output_file}")
        run_logger.info(f"Summary:   {summary_file}")
        run_logger.info(f"Time:      {result.execution_time:.2f}s")
        run_logger.info(f"Iterations: {iterations_used}/{settings.max_iterations}")

        if budget:
            snap = budget.snapshot()
            run_logger.info(
                f"Context:   {snap['percent_used']:.1f}% ({snap['used_tokens']:,} / {snap['total_tokens']:,} tokens)",
            )

        if not completed:
            run_logger.warning("Investigation hit MAX_ITERATIONS — results may be incomplete.")

    finally:
        prep_result.cleanup()


if __name__ == "__main__":
    try:
        main()
    except ValidationError as e:
        for error in e.errors():
            field = " -> ".join(str(loc) for loc in error["loc"])
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
