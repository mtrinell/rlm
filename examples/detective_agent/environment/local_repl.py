"""
Local REPL environment with persistent Python namespace.

Forked from rlms library -- executes code in a sandboxed namespace.
"""

import ast
import copy
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import types
import uuid
from contextlib import contextmanager, suppress
from typing import Any

from examples.detective_agent.core.comms_utils import LMRequest, send_lm_request, send_lm_request_batched
from examples.detective_agent.core.types import REPLResult, RLMChatCompletion

from examples.detective_agent.environment.base_env import NonIsolatedEnv

# =============================================================================
# Safe Builtins — blocks dangerous operations
# =============================================================================

_SAFE_BUILTINS = {
    "print": print,
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "bool": bool,
    "type": type,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    "range": range,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "any": any,
    "all": all,
    "pow": pow,
    "divmod": divmod,
    "chr": chr,
    "ord": ord,
    "hex": hex,
    "bin": bin,
    "oct": oct,
    "repr": repr,
    "ascii": ascii,
    "format": format,
    "hash": hash,
    "id": id,
    "iter": iter,
    "next": next,
    "slice": slice,
    "callable": callable,
    "hasattr": hasattr,
    "getattr": getattr,
    "setattr": setattr,
    "delattr": delattr,
    "dir": dir,
    "vars": vars,
    "bytes": bytes,
    "bytearray": bytearray,
    "memoryview": memoryview,
    "complex": complex,
    "object": object,
    "super": super,
    "property": property,
    "staticmethod": staticmethod,
    "classmethod": classmethod,
    "__import__": __import__,
    "open": open,
    # Exceptions
    "Exception": Exception,
    "BaseException": BaseException,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "FileNotFoundError": FileNotFoundError,
    "OSError": OSError,
    "IOError": IOError,
    "RuntimeError": RuntimeError,
    "NameError": NameError,
    "ImportError": ImportError,
    "StopIteration": StopIteration,
    "AssertionError": AssertionError,
    "NotImplementedError": NotImplementedError,
    "ArithmeticError": ArithmeticError,
    "LookupError": LookupError,
    "Warning": Warning,
    "UnicodeDecodeError": UnicodeDecodeError,
    "PermissionError": PermissionError,
    "TimeoutError": TimeoutError,
    "ConnectionError": ConnectionError,
    # Blocked -- use descriptive errors so the LLM knows what went wrong
    "input": lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("input() is blocked in this REPL.")),
    "eval": lambda *_a, **_kw: (_ for _ in ()).throw(
        RuntimeError("eval() is blocked in this REPL. Write code directly in ```repl blocks instead."),
    ),
    "exec": lambda *_a, **_kw: (_ for _ in ()).throw(
        RuntimeError("exec() is blocked in this REPL. Write code directly in ```repl blocks instead."),
    ),
    "compile": lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("compile() is blocked in this REPL.")),
    "globals": lambda: (_ for _ in ()).throw(
        RuntimeError(
            "globals() is blocked in this REPL. Variables persist across "
            "iterations automatically -- just reference them by name.",
        ),
    ),
    "locals": lambda: (_ for _ in ()).throw(
        RuntimeError(
            "locals() is blocked in this REPL. Variables persist across "
            "iterations automatically -- just reference them by name.",
        ),
    ),
}


class LocalREPL(NonIsolatedEnv):
    """
    Local REPL environment with persistent Python namespace.

    Executes code in a sandboxed namespace with access to context data.
    """

    def __init__(
        self,
        lm_handler_address: tuple[str, int] | None = None,
        context_payload: dict | list | str | None = None,
        setup_code: str | None = None,
        persistent: bool = False,
        depth: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(persistent=persistent, depth=depth, **kwargs)

        self.lm_handler_address = lm_handler_address
        self.original_cwd = os.getcwd()
        self.temp_dir = tempfile.mkdtemp(prefix=f"repl_env_{uuid.uuid4()}_")
        self._lock = threading.Lock()
        self._context_count: int = 0
        self._history_count: int = 0

        self.setup()

        if context_payload is not None:
            self.load_context(context_payload)

        if setup_code:
            self.execute_code(setup_code)

    def setup(self) -> None:
        """Setup the environment — create sandboxed globals and register helpers."""
        self.globals: dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS.copy(),
            "__name__": "__main__",
        }
        self.locals: dict[str, Any] = {}
        self._pending_llm_calls: list[RLMChatCompletion] = []

        self.globals["FINAL_VAR"] = self._final_var
        self.globals["SHOW_VARS"] = self._show_vars
        self.globals["llm_query"] = self._llm_query
        self.globals["llm_query_batched"] = self._llm_query_batched

    def _final_var(self, variable_name: str) -> str:
        variable_name = variable_name.strip().strip("\"'")
        if variable_name in self.locals:
            return str(self.locals[variable_name])
        available = [k for k in self.locals if not k.startswith("_")]
        if available:
            return (
                f"Error: Variable '{variable_name}' not found. "
                f"Available variables: {available}. "
                "You must create and assign the variable BEFORE calling FINAL_VAR."
            )
        return (
            f"Error: Variable '{variable_name}' not found. No variables created yet. "
            "Create and assign the variable in a repl block first."
        )

    def _show_vars(self) -> str:
        available = {k: type(v).__name__ for k, v in self.locals.items() if not k.startswith("_")}
        if not available:
            return "No variables created yet."
        return f"Available variables: {available}"

    def _llm_query(self, prompt: str, model: str | None = None) -> str:
        if not self.lm_handler_address:
            return "Error: No LM handler configured"
        try:
            request = LMRequest(prompt=prompt, model=model, depth=self.depth)
            response = send_lm_request(self.lm_handler_address, request)
            if not response.success:
                return f"Error: {response.error}"
            self._pending_llm_calls.append(response.chat_completion)
            return response.chat_completion.response
        except Exception as e:
            return f"Error: LM query failed - {e}"

    def _llm_query_batched(self, prompts: list[str], model: str | None = None) -> list[str]:
        if not self.lm_handler_address:
            return ["Error: No LM handler configured"] * len(prompts)
        try:
            responses = send_lm_request_batched(
                self.lm_handler_address,
                prompts,
                model=model,
                depth=self.depth,
            )
            results = []
            for response in responses:
                if not response.success:
                    results.append(f"Error: {response.error}")
                else:
                    self._pending_llm_calls.append(response.chat_completion)
                    results.append(response.chat_completion.response)
            return results
        except Exception as e:
            return [f"Error: LM query failed - {e}"] * len(prompts)

    def load_context(self, context_payload: dict | list | str) -> None:
        """Load context into the environment as context_0 (and 'context' alias)."""
        self.add_context(context_payload, 0)

    def add_context(
        self,
        context_payload: dict | list | str,
        context_index: int | None = None,
    ) -> int:
        if context_index is None:
            context_index = self._context_count

        var_name = f"context_{context_index}"

        if isinstance(context_payload, str):
            context_path = os.path.join(self.temp_dir, f"context_{context_index}.txt")
            with open(context_path, "w") as f:
                f.write(context_payload)
            self.execute_code(
                f"with open(r'{context_path}', 'r') as f:\n    {var_name} = f.read()",
            )
        else:
            context_path = os.path.join(self.temp_dir, f"context_{context_index}.json")
            with open(context_path, "w") as f:
                json.dump(context_payload, f)
            self.execute_code(
                f"import json\nwith open(r'{context_path}', 'r') as f:\n    {var_name} = json.load(f)",
            )

        if context_index == 0:
            self.execute_code(f"context = {var_name}")

        self._context_count = max(self._context_count, context_index + 1)
        return context_index

    def update_handler_address(self, address: tuple[str, int]) -> None:
        self.lm_handler_address = address

    def get_context_count(self) -> int:
        return self._context_count

    def add_history(
        self,
        message_history: list[dict[str, Any]],
        history_index: int | None = None,
    ) -> int:
        if history_index is None:
            history_index = self._history_count

        var_name = f"history_{history_index}"
        self.locals[var_name] = copy.deepcopy(message_history)
        if history_index == 0:
            self.locals["history"] = self.locals[var_name]

        self._history_count = max(self._history_count, history_index + 1)
        return history_index

    def get_history_count(self) -> int:
        return self._history_count

    @contextmanager
    def _capture_output(self) -> Any:
        """Thread-safe context manager to capture stdout/stderr."""
        with self._lock:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
            try:
                sys.stdout, sys.stderr = stdout_buf, stderr_buf
                yield stdout_buf, stderr_buf
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

    @contextmanager
    def _temp_cwd(self) -> Any:
        old_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            yield
        finally:
            os.chdir(old_cwd)

    @staticmethod
    def _split_last_expr(code: str) -> tuple[str | None, str | None]:
        """
        Split code into (body, last_expr) if the last statement is a bare expression.

        Returns (body_code, last_expr_code) where body_code is everything except
        the final expression, and last_expr_code is the final expression source.
        If the last statement is NOT a bare expression, returns (None, None) —
        meaning the caller should execute the code as-is.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None, None

        if not tree.body:
            return None, None

        last_node = tree.body[-1]
        if not isinstance(last_node, ast.Expr):
            return None, None

        # Don't auto-display calls to print/FINAL_VAR — they already produce output
        if isinstance(last_node.value, ast.Call):
            func = last_node.value.func
            if isinstance(func, ast.Name) and func.id in ("print", "FINAL_VAR"):
                return None, None

        # Extract source for the last expression using line numbers
        last_expr_code = ast.get_source_segment(code, last_node)
        if last_expr_code is None:
            # Fallback: grab from start line to end
            lines = code.splitlines(keepends=True)
            last_expr_code = "".join(lines[last_node.lineno - 1 :])

        # Body = everything before the last statement
        if len(tree.body) <= 1:
            body_code = None
        else:
            lines = code.splitlines(keepends=True)
            body_code = "".join(lines[: last_node.lineno - 1])

        return body_code, last_expr_code

    def execute_code(self, code: str) -> REPLResult:
        """Execute code in the persistent namespace and return result."""
        start_time = time.perf_counter()
        self._pending_llm_calls = []

        # Builtin names must never be persisted to self.locals — if they are,
        # they shadow the real builtins on every subsequent iteration.
        builtins_dict = self.globals.get("__builtins__", {})
        _is_builtin = (
            builtins_dict.__contains__ if isinstance(builtins_dict, dict) else lambda _k: hasattr(builtins_dict, _k)
        )

        with self._capture_output() as (stdout_buf, stderr_buf), self._temp_cwd():
            try:
                # Rebuild combined namespace, dropping any locals that shadow builtins
                # (cleans up corruption from prior iterations)
                combined = {**self.globals}
                combined.update(
                    {k: v for k, v in self.locals.items() if not _is_builtin(k)},
                )

                # Jupyter-style auto-display: if the last statement is a bare
                # expression, exec the body then eval+print the last expression.
                body_code, last_expr_code = self._split_last_expr(code)
                if last_expr_code is not None:
                    if body_code:
                        exec(body_code, combined, combined)  # noqa: S102
                    _last_val = eval(last_expr_code, combined, combined)  # noqa: S307
                    if _last_val is not None:
                        pass
                else:
                    exec(code, combined, combined)  # noqa: S102

                for key, value in combined.items():
                    if key not in self.globals and not key.startswith("_") and not _is_builtin(key):
                        self.locals[key] = value

                # Also purge any previously-persisted builtin shadows
                for k in list(self.locals):
                    if _is_builtin(k):
                        del self.locals[k]

                # Special case: persist FINAL_VAR if model assigned it as a string.
                # The model naturally writes `FINAL_VAR = "..."` (assignment form), which
                # overwrites the global callable in `combined` but is blocked from persisting
                # to locals by the `key not in self.globals` guard above. Explicitly persist.
                fv = combined.get("FINAL_VAR")
                if fv is not None and isinstance(fv, str):
                    self.locals["FINAL_VAR"] = fv

                stdout = stdout_buf.getvalue()
                stderr = stderr_buf.getvalue()
            except Exception as e:
                stdout = stdout_buf.getvalue()
                stderr = stderr_buf.getvalue() + f"\n{type(e).__name__}: {e}"

        return REPLResult(
            stdout=stdout,
            stderr=stderr,
            locals=self.locals.copy(),
            execution_time=time.perf_counter() - start_time,
            rlm_calls=self._pending_llm_calls.copy(),
        )

    def cleanup(self) -> None:
        with suppress(Exception):
            shutil.rmtree(self.temp_dir)
        self.globals.clear()
        self.locals.clear()

    def __enter__(self) -> "LocalREPL":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> bool:
        self.cleanup()
        return False

    def __del__(self) -> None:
        self.cleanup()
