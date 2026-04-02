"""
REPL helper functions for behavioral deviation detection.

These helpers provide structured access to OpenTelemetry JSONL traces,
enabling the LLM to parse, query, and analyse application behavior
without generating error-prone trace-parsing code from scratch.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# OTEL severity number → text mapping
_SEVERITY_MAP: dict[int, str] = {
    1: "TRACE",
    5: "DEBUG",
    9: "INFO",
    13: "WARN",
    17: "ERROR",
    21: "FATAL",
}

_NS_PER_SEC = 1_000_000_000


def _ns_to_dt(ns: int) -> datetime:
    """Convert nanosecond UNIX timestamp to UTC datetime."""
    return datetime.fromtimestamp(ns / _NS_PER_SEC, tz=UTC)


def _extract_attr_value(attr: dict[str, Any]) -> Any:
    """Extract a value from an OTEL attribute dict (handles all value types)."""
    val = attr.get("value", {})
    for key in ("stringValue", "intValue", "doubleValue", "boolValue"):
        if key in val:
            return val[key]
    if "arrayValue" in val:
        values = val["arrayValue"].get("values", [])
        return [_extract_attr_value({"value": v}) for v in values]
    return None


def _attrs_to_dict(attributes: list[dict[str, Any]]) -> dict[str, Any]:
    """Flatten an OTEL attributes list into a plain dict."""
    return {a["key"]: _extract_attr_value(a) for a in attributes if "key" in a}


def build_detector_helpers(traces_path: str, budget: Any = None) -> dict[str, Any]:
    """
    Build the dict of helper functions for OTEL trace analysis.

    Args:
        traces_path: Path to the OTEL JSONL traces file.
        budget: Optional ContextBudget instance for context_budget() calls.

    Returns:
        Dict mapping function names to callables, ready for REPL injection.

    """

    def load_traces(max_lines: int | None = None) -> list[dict[str, Any]]:
        """
        Parse the OTEL JSONL file into a list of resource batch dicts.

        Each element corresponds to one JSON line in the file, which may
        contain `resourceLogs` and/or `resourceSpans` keys.

        Args:
            max_lines: If set, read only the first N lines (useful for sampling).

        Returns:
            List of parsed JSON objects.

        """
        path = Path(traces_path)
        results = []
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh):
                    if max_lines is not None and i >= max_lines:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        except FileNotFoundError:
            return [{"error": f"File not found: {traces_path}"}]
        except OSError as e:
            return [{"error": str(e)}]
        return results

    def extract_log_records(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Flatten all log records from parsed OTEL resource batches.

        Args:
            traces: Output of load_traces().

        Returns:
            List of flat log record dicts with keys:
              timestamp_ns, timestamp_dt, severity, severity_text, body,
              scope, attributes, trace_id, span_id, service_name.

        """
        records: list[dict[str, Any]] = []
        for batch in traces:
            # Extract service name from resource attributes
            service_name = "unknown"
            for rl in batch.get("resourceLogs", []):
                resource_attrs = _attrs_to_dict(rl.get("resource", {}).get("attributes", []))
                service_name = resource_attrs.get("service.name", "unknown")
                for scope_log in rl.get("scopeLogs", []):
                    scope_name = scope_log.get("scope", {}).get("name", "")
                    for rec in scope_log.get("logRecords", []):
                        ts_ns = int(rec.get("timeUnixNano", 0) or 0)
                        sev_num = int(rec.get("severityNumber", 9))
                        body = rec.get("body", {}).get("stringValue", "")
                        attrs = _attrs_to_dict(rec.get("attributes", []))
                        records.append(
                            {
                                "timestamp_ns": ts_ns,
                                "timestamp_dt": _ns_to_dt(ts_ns) if ts_ns else None,
                                "severity": sev_num,
                                "severity_text": rec.get("severityText") or _SEVERITY_MAP.get(sev_num, str(sev_num)),
                                "body": body,
                                "scope": scope_name,
                                "attributes": attrs,
                                "trace_id": rec.get("traceId", ""),
                                "span_id": rec.get("spanId", ""),
                                "service_name": service_name,
                            }
                        )
        # Sort by timestamp
        records.sort(key=lambda r: r["timestamp_ns"])
        return records

    def extract_spans(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Flatten all span records from parsed OTEL resource batches.

        Args:
            traces: Output of load_traces().

        Returns:
            List of flat span dicts with keys:
              trace_id, span_id, parent_span_id, name, kind,
              start_ns, end_ns, duration_ms, service_name, scope, attributes, status.

        """
        spans: list[dict[str, Any]] = []
        for batch in traces:
            for rs in batch.get("resourceSpans", []):
                resource_attrs = _attrs_to_dict(rs.get("resource", {}).get("attributes", []))
                service_name = resource_attrs.get("service.name", "unknown")
                for scope_span in rs.get("scopeSpans", []):
                    scope_name = scope_span.get("scope", {}).get("name", "")
                    for span in scope_span.get("spans", []):
                        start_ns = int(span.get("startTimeUnixNano", 0) or 0)
                        end_ns = int(span.get("endTimeUnixNano", 0) or 0)
                        duration_ms = (end_ns - start_ns) / 1_000_000 if end_ns > start_ns else 0
                        spans.append(
                            {
                                "trace_id": span.get("traceId", ""),
                                "span_id": span.get("spanId", ""),
                                "parent_span_id": span.get("parentSpanId", ""),
                                "name": span.get("name", ""),
                                "kind": span.get("kind", 0),
                                "start_ns": start_ns,
                                "end_ns": end_ns,
                                "duration_ms": duration_ms,
                                "service_name": service_name,
                                "scope": scope_name,
                                "attributes": _attrs_to_dict(span.get("attributes", [])),
                                "status": span.get("status", {}),
                            }
                        )
        spans.sort(key=lambda s: s["start_ns"])
        return spans

    def get_errors(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Filter log records to severity >= 17 (ERROR and FATAL).

        Args:
            records: Output of extract_log_records().

        Returns:
            Filtered list of error/fatal records.

        """
        return [r for r in records if r["severity"] >= 17]

    def get_component_timeline(
        records: list[dict[str, Any]],
        max_per_component: int = 200,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Group log records by component scope, sorted by timestamp.

        Args:
            records: Output of extract_log_records().
            max_per_component: Cap the number of records returned per component.

        Returns:
            Dict mapping scope name → list of records (capped at max_per_component).

        """
        timeline: dict[str, list[dict[str, Any]]] = {}
        for rec in records:
            scope = rec["scope"] or "unknown"
            timeline.setdefault(scope, []).append(rec)
        for scope in timeline:
            timeline[scope] = timeline[scope][:max_per_component]
        return timeline

    def search_traces(
        records: list[dict[str, Any]],
        pattern: str,
        flags: str = "i",
    ) -> list[dict[str, Any]]:
        """
        Search log record bodies with a regex pattern.

        Args:
            records: Output of extract_log_records().
            pattern: Regex pattern to search for.
            flags: 'i' = case-insensitive (default), 'n' = case-sensitive.

        Returns:
            Matching records.

        """
        re_flags = re.IGNORECASE if "i" in flags else 0
        try:
            compiled = re.compile(pattern, re_flags)
        except re.error:
            return [{"error": f"Invalid regex: {pattern}"}]
        return [r for r in records if compiled.search(r.get("body", ""))]

    def summarize_component(records: list[dict[str, Any]], component: str) -> str:
        """
        Return a human-readable text summary of all records for a given scope.

        Args:
            records: Output of extract_log_records().
            component: Scope name to filter on (exact match or substring).

        Returns:
            Multi-line string suitable for passing to llm_query().

        """
        filtered = [r for r in records if component in r.get("scope", "")]
        if not filtered:
            return f"No records found for component '{component}'."
        lines: list[str] = [f"=== {component} ({len(filtered)} records) ==="]
        for r in filtered:
            dt_str = r["timestamp_dt"].isoformat() if r.get("timestamp_dt") else "?"
            sev = r.get("severity_text", "INFO")
            body = r.get("body", "")
            attrs = r.get("attributes", {})
            exc = attrs.get("exception.message", "")
            line = f"[{dt_str}] [{sev}] {body}"
            if exc:
                line += f" | exception: {exc}"
            lines.append(line)
        return "\n".join(lines)

    def context_budget() -> dict[str, Any]:
        """Return current context token budget status."""
        if budget is None:
            return {
                "used_tokens": 0,
                "total_tokens": 0,
                "percent_used": 0.0,
                "level": "none",
                "tokens_remaining": 0,
            }
        snap = budget.snapshot()
        return snap

    return {
        "load_traces": load_traces,
        "extract_log_records": extract_log_records,
        "extract_spans": extract_spans,
        "get_errors": get_errors,
        "get_component_timeline": get_component_timeline,
        "search_traces": search_traces,
        "summarize_component": summarize_component,
        "context_budget": context_budget,
    }
