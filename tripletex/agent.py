"""
Three-phase Tripletex agent.

Phase 0: Extract (Flash, no tools, only if files present) — pull structured
         data (names, amounts, dates) from PDFs/images.

Phase 1: Discover + Plan (Flash, agentic, PLAN_TOOL) — identify endpoints,
         check existing entities, read endpoint notes from SQLite, output a
         structured plan with endpoints_used.

Code step: Load schemas from openapi.json + notes from SQLite for each
           endpoint in the plan — no LLM calls.

Phase 2: Execute (Pro, GEMINI_TOOL) — receives task + extracted data + plan
         + pre-loaded context block. In DEV_MODE, writes new rules to SQLite
         when a 422 is encountered.

A 5-minute timer runs from entry and is visible to all phases.
"""

import base64
import json
import logging
import re
import threading
import time

from google.api_core.exceptions import TooManyRequests, ServiceUnavailable

import vertexai
from vertexai.generative_models import Content, GenerativeModel, Part

import knowledge
from client import TripletexClient, TripletexError
from config import settings
from tools import GEMINI_TOOL, PLAN_TOOL, TOOL_MAP
from tools.schema_tools import get_endpoint_schema

logger = logging.getLogger(__name__)

_NUMERIC_SEGMENT = re.compile(r"/\d+")


def _normalize_endpoint(method: str, path: str) -> str:
    """Replace numeric path segments with {id} so KB keys are generic."""
    return f"{method.upper()} {_NUMERIC_SEGMENT.sub('/{id}', path)}"

TOTAL_BUDGET = 300  # seconds
PLAN_MAX_TURNS = 15

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM_PROMPT = """\
Extract all structured data from the attached files.
Output ONLY valid JSON (no markdown fences):
{
  "line_items": [{"description": "...", "quantity": 1, "unit_price": 0.0}],
  "names": ["..."],
  "amounts": [{"label": "...", "value": 0.0, "currency": "NOK"}],
  "dates": [{"label": "...", "value": "YYYY-MM-DD"}],
  "reference_numbers": ["..."],
  "other": {}
}
Include every piece of data you can find. Use null for unknown fields."""

PLAN_SYSTEM_PROMPT = """\
You are a Tripletex planning agent. Build an optimized execution plan in two phases.

PHASE 1 — DISCOVER: For each operation in the task, call list_endpoints to find the exact API path.

PHASE 2 — REFINE: For every write endpoint (POST/PUT/DELETE) found:
  - Call get_endpoint_notes to check for known dependencies or requirements
  - If a note says a prerequisite is needed (e.g. look up a vatType ID first), add that step to the plan
  - Check if a batch endpoint exists for repeated similar operations and prefer it
  - Remove any redundant steps

Do not fetch live data. Do not call the same tool with the same arguments twice.

Output ONLY valid JSON (no markdown fences):
{
  "endpoints_used": [{"path": "/customer", "method": "POST"}, ...],
  "steps": ["step 1", "step 2", ...],
  "notes": "anything unusual"
}"""

EXECUTION_SYSTEM_PROMPT = """\
You are a Tripletex execution agent for a Norwegian ERP system.
You have a plan. Execute it step by step using call_api.

call_api:
- method: GET, POST, PUT, DELETE
- endpoint: exact path, e.g. "/customer", "/invoice/123/:payment"
- body: JSON object (POST/PUT)
- params: query parameters dict
- Use camelCase field names

Currency: Tripletex operates in NOK. All amounts passed to the API must be in NOK. If the task involves foreign currencies, convert to NOK using the provided exchange rate before calling the API.

Rules:
- Follow the plan. Endpoint notes and schemas are already in the context below — read them before each write
- Use GET via call_api to fetch any IDs you need at runtime that are not in the reference data below
- If you need to use an endpoint NOT in the plan (e.g. after an error): call list_endpoints to find it
- Many resources have sub-resource endpoints (e.g. /order/orderline). If you already included those sub-resources in the parent body during creation, do NOT also call the sub-resource endpoint without first checking whether they already exist — it will create duplicates
- If a write fails, read the error and fix it — one retry only
- When a write succeeds, move on immediately — do NOT verify with a GET
- When all steps are done, stop

Time:
- Remaining < 60s: skip get_endpoint_notes calls, writes only
- Remaining < 20s: stop immediately"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _send_with_retry(chat, parts, tools, max_attempts=4):
    """Send a message with exponential backoff on 429/503."""
    delay = 10
    for attempt in range(max_attempts):
        try:
            return chat.send_message(parts, tools=tools)
        except (TooManyRequests, ServiceUnavailable) as exc:
            if attempt == max_attempts - 1:
                raise
            logger.warning("Rate limited (%s), retrying in %ds...", exc.__class__.__name__, delay)
            time.sleep(delay)
            delay *= 2


def _dispatch_tool(client: TripletexClient, fn_name: str, fn_args: dict):
    fn = TOOL_MAP.get(fn_name)
    if not fn:
        return {"error": f"Unknown tool: {fn_name}"}
    return fn(client, **fn_args)


def _run_tool_loop(
    chat,
    client: TripletexClient,
    initial_parts: list[Part],
    tool: object,
    max_turns: int = 30,
    on_422=None,
    planned_keys: set[str] = None,
) -> tuple[str | None, int]:
    """
    Send initial_parts, then dispatch tool calls until the model stops.
    Returns (final_text, turns_used).

    planned_keys: endpoints whose notes are already in the context block —
                  skip the KB intercept for these to avoid redundant turns.
    on_422: optional callable(endpoint_key, error_body) called after a 4xx response.
    """
    response = _send_with_retry(chat, initial_parts, [tool])
    turns = 0
    _notes_shown: set[str] = set()  # tracks unplanned endpoints where notes have been shown

    while turns < max_turns:
        function_calls = [
            p for p in response.candidates[0].content.parts
            if p.function_call and p.function_call.name
        ]
        if not function_calls:
            break

        turns += 1
        function_responses: list[Part] = []
        for part in function_calls:
            fc = part.function_call
            fn_name = fc.name
            fn_args = dict(fc.args)
            logger.info("Tool call: %s(%s)", fn_name, fn_args)
            # For call_api writes on UNPLANNED endpoints: show KB notes before
            # executing so the model can reconsider. Planned endpoints already
            # have their notes in the context block — intercepting again wastes a turn.
            _ep = fn_args.get("endpoint") or fn_args.get("path", "")
            _key = _normalize_endpoint(fn_args.get("method", ""), _ep)
            _is_unplanned = planned_keys is None or _key not in planned_keys
            if fn_name == "call_api" and fn_args.get("method", "").upper() in ("POST", "PUT", "DELETE") and _is_unplanned and _key not in _notes_shown:
                _notes = knowledge.get_notes(_key)
                if _notes:
                    _notes_shown.add(_key)
                    logger.info("Intercepted %s — returning notes before call", _key)
                    function_responses.append(
                        Part.from_function_response(name=fn_name, response={"result": {
                            "endpoint_notes": _notes,
                            "status": "call not executed — review notes above and call again to proceed",
                        }})
                    )
                    continue
            try:
                result = _dispatch_tool(client, fn_name, fn_args)
                if result is None:
                    result = {"ok": True}
            except TripletexError as exc:
                logger.warning("Tool %s failed: %s", fn_name, exc)
                result = {"error": str(exc)}
                if on_422 and exc.status_code in (400, 422):
                    method = fn_args.get("method", "")
                    path = fn_args.get("endpoint") or fn_args.get("path") or fn_name
                    endpoint_hint = _normalize_endpoint(method, path) if method else path
                    on_422(endpoint_hint, {"error": str(exc), "tool": fn_name, "args": fn_args})
            except Exception as exc:
                logger.warning("Tool %s failed: %s", fn_name, exc)
                result = {"error": str(exc)}
            function_responses.append(
                Part.from_function_response(name=fn_name, response={"result": result})
            )

        response = _send_with_retry(chat, Content(role="user", parts=function_responses), [tool])

    try:
        return response.text.strip(), turns
    except Exception:
        # Last response had no text (model was mid-tool-call when we stopped).
        # Send one more message to force a JSON summary.
        try:
            force = chat.send_message(
                Content(role="user", parts=[Part.from_text(
                    "You have reached the tool call limit. Output your plan as JSON now with what you have gathered so far."
                )]),
                tools=[tool],
            )
            return force.text.strip(), turns
        except Exception:
            return None, turns


def _parse_json(text: str) -> dict:
    if not text:
        return {}
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        t = "\n".join(lines[1:])
        if t.rstrip().endswith("```"):
            t = t[: t.rfind("```")]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON: %.200s", text)
        return {"raw": text[:1000]}


def _time_context(start: float) -> str:
    elapsed = time.time() - start
    remaining = TOTAL_BUDGET - elapsed
    return f"[Elapsed: {elapsed:.0f}s | Remaining: {remaining:.0f}s]"


def _build_user_parts(files: list[dict], text: str) -> list[Part]:
    parts: list[Part] = []
    for f in (files or []):
        try:
            data = base64.b64decode(f["content_base64"])
            parts.append(Part.from_data(data=data, mime_type=f["mime_type"]))
            logger.info("Attached file: %s (%s, %d bytes)", f["filename"], f["mime_type"], len(data))
        except Exception as exc:
            logger.warning("Failed to decode file %s: %s", f.get("filename"), exc)
    parts.append(Part.from_text(text))
    return parts


_VAT_ENDPOINTS = {"/product", "/order", "/invoice", "/ledger/voucher", "/supplierInvoice"}
_TRAVEL_ENDPOINTS = {"/travelExpense/cost"}
_PROJECT_ENDPOINTS = {"/project"}
_TIMESHEET_ENDPOINTS = {"/timesheet/entry", "/timesheet/entry/list"}


def _load_reference_data(client: TripletexClient, plan: dict) -> str:
    """Pre-fetch reference data needed by the plan, injected into executor context."""
    paths = {ep.get("path", "") for ep in plan.get("endpoints_used", [])}
    lines = []

    if paths & _VAT_ENDPOINTS:
        try:
            result = client.get("/ledger/vatType", params={"fields": "id,name,percentage", "count": 100})
            vat_types = result.get("values") or []
            if vat_types:
                formatted = ", ".join(
                    f"{v.get('name','')} {v.get('percentage','')}% → id={v['id']}" for v in vat_types
                )
                lines.append(f"VAT types: {formatted}")
        except Exception as exc:
            logger.warning("Failed to pre-load vatTypes: %s", exc)

    if paths & _TRAVEL_ENDPOINTS:
        try:
            result = client.get("/travelExpense/paymentType", params={"fields": "id,name", "count": 100})
            payment_types = result.get("values") or []
            if payment_types:
                formatted = ", ".join(f"{p.get('name','')} → id={p['id']}" for p in payment_types)
                lines.append(f"Travel expense payment types: {formatted}")
        except Exception as exc:
            logger.warning("Failed to pre-load travel paymentTypes: %s", exc)

    if paths & _PROJECT_ENDPOINTS:
        try:
            result = client.get("/project", params={"fields": "projectManager", "count": 1})
            values = result.get("values") or []
            if values and values[0].get("projectManager"):
                pm = values[0]["projectManager"]
                lines.append(f"Valid project manager id: {pm['id']} — use this, do not guess")
        except Exception as exc:
            logger.warning("Failed to pre-load project manager: %s", exc)

    if paths & _TIMESHEET_ENDPOINTS:
        try:
            result = client.get("/activity", params={"fields": "id,name", "count": 20})
            activities = result.get("values") or []
            if activities:
                formatted = ", ".join(f"{a.get('name','')} → id={a['id']}" for a in activities)
                lines.append(f"Activities: {formatted}")
        except Exception as exc:
            logger.warning("Failed to pre-load activities: %s", exc)

    if not lines:
        return ""
    return "=== Reference data (account-specific ids — use directly, do not re-fetch) ===\n" + "\n".join(lines)


def _build_context_block(plan: dict) -> str:
    """
    Load schemas and notes for each endpoint in the plan.
    Returns a formatted string injected into the executor's context.
    """
    endpoints_used = plan.get("endpoints_used", [])
    if not endpoints_used:
        return ""

    schema_lines = []
    note_lines = []

    for ep in endpoints_used:
        path = ep.get("path", "")
        method = ep.get("method", "GET").upper()
        key = f"{method} {path}"

        # Load schema from openapi.json (no LLM call)
        schema_result = get_endpoint_schema(path=path, method=method)
        fields = schema_result.get("fields", [])
        if fields:
            field_names = ", ".join(f["name"] for f in fields)
            schema_lines.append(f"{key}: {field_names}")

        # Load notes from SQLite
        notes = knowledge.get_notes(key)
        if notes:
            note_lines.append(f"{key}:\n  {notes}")

    parts = []
    if schema_lines:
        parts.append("=== Endpoint schemas ===\n" + "\n".join(schema_lines))
    if note_lines:
        parts.append("=== Endpoint notes (known dependencies & gotchas) ===\n" + "\n".join(note_lines))

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Background rule writer
# ---------------------------------------------------------------------------

_RULE_WRITER_PROMPT = (
    "You update a knowledge base for a Tripletex API agent. "
    "Given an endpoint and a 422 error response, produce a concise rule "
    "explaining what is required or what dependency must be satisfied first. "
    "Output ONLY the rule text (1-3 sentences, no JSON, no markdown)."
)


def _write_rules_async(model_name: str, pending: list[tuple[str, dict]]) -> None:
    """Write 422 rules to the knowledge base. Runs in a daemon thread."""
    try:
        rule_writer = GenerativeModel(model_name=model_name, system_instruction=_RULE_WRITER_PROMPT)
        for endpoint_key, error_body in pending:
            existing = knowledge.get_notes(endpoint_key) or ""
            rule_prompt = (
                f"Endpoint: {endpoint_key}\n"
                f"Existing notes: {existing or '(none)'}\n"
                f"422 error: {json.dumps(error_body, ensure_ascii=False)[:600]}\n\n"
                "Write the updated notes for this endpoint (include existing notes + new rule):"
            )
            try:
                resp = rule_writer.generate_content(rule_prompt)
                knowledge.upsert_notes(endpoint_key, resp.text.strip())
                logger.info("Knowledge: updated notes for %s", endpoint_key)
            except Exception as exc:
                logger.warning("Knowledge: failed to write notes for %s: %s", endpoint_key, exc)
    except Exception as exc:
        logger.warning("Knowledge: rule writer setup failed: %s", exc)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_agent(prompt: str, client: TripletexClient, files: list[dict] = None) -> None:
    vertexai.init(project=settings.gcp_project, location=settings.vertex_location)
    start = time.time()

    logger.info("Agent starting | model=%s | prompt=%.120s", settings.gemini_model, prompt)

    # -------------------------------------------------------------------------
    # Phase 0: Extract (only if files are present)
    # -------------------------------------------------------------------------
    extracted_data: dict = {}
    if files:
        extractor = GenerativeModel(
            model_name=settings.gemini_fast_model,
            system_instruction=EXTRACT_SYSTEM_PROMPT,
        )
        file_parts = _build_user_parts(files, prompt)
        try:
            extract_response = extractor.generate_content(file_parts)
            extracted_data = _parse_json(extract_response.text)
            logger.info("Extract phase: %s", json.dumps(extracted_data, ensure_ascii=False)[:300])
        except Exception as exc:
            logger.warning("Extract phase failed: %s", exc)

    # -------------------------------------------------------------------------
    # Phase 1: Discover + Plan (Flash, agentic, PLAN_TOOL)
    # -------------------------------------------------------------------------
    planner = GenerativeModel(
        model_name=settings.gemini_fast_model,
        system_instruction=PLAN_SYSTEM_PROMPT,
    )
    plan_chat = planner.start_chat(response_validation=False)

    plan_text_input = f"{_time_context(start)}\n\n{prompt}"
    if extracted_data:
        plan_text_input += f"\n\nExtracted from attached files:\n{json.dumps(extracted_data, ensure_ascii=False, indent=2)}"

    plan_parts = _build_user_parts(files if not extracted_data else [], plan_text_input)

    plan_text, plan_turns = _run_tool_loop(
        plan_chat, client, plan_parts, PLAN_TOOL, max_turns=PLAN_MAX_TURNS
    )

    if plan_turns == 0:
        logger.info("Plan phase: no tool calls (fast path)")
        plan = {}
    else:
        logger.info("Plan phase: %d turns", plan_turns)
        plan = _parse_json(plan_text)
        logger.info("Plan: %s", json.dumps(plan, ensure_ascii=False)[:500])

    # -------------------------------------------------------------------------
    # Code step: pre-load schemas + notes for planned endpoints
    # -------------------------------------------------------------------------
    context_block = _build_context_block(plan)
    if context_block:
        logger.info("Context block built (%d chars)", len(context_block))

    # -------------------------------------------------------------------------
    # Phase 2: Execute (Pro, GEMINI_TOOL)
    # -------------------------------------------------------------------------
    executor = GenerativeModel(
        model_name=settings.gemini_model,
        system_instruction=EXECUTION_SYSTEM_PROMPT,
    )
    exec_chat = executor.start_chat(response_validation=False)

    ref_data = _load_reference_data(client, plan)

    exec_prompt = f"{_time_context(start)}\n\nTask: {prompt}"
    if extracted_data:
        exec_prompt += f"\n\nExtracted data:\n{json.dumps(extracted_data, ensure_ascii=False, indent=2)}"
    exec_prompt += f"\n\nPlan:\n{json.dumps(plan, ensure_ascii=False, indent=2)}"
    if context_block:
        exec_prompt += f"\n\n{context_block}"
    if ref_data:
        exec_prompt += f"\n\n{ref_data}"

    exec_parts = _build_user_parts([], exec_prompt)

    # Always capture 422s — write rules to knowledge base async after returning
    _pending_422: list[tuple[str, dict]] = []

    def _on_422(endpoint_key: str, error_body: dict) -> None:
        _pending_422.append((endpoint_key, error_body))

    planned_keys = {
        _normalize_endpoint(ep.get("method", "GET"), ep.get("path", ""))
        for ep in plan.get("endpoints_used", [])
    }
    _, exec_turns = _run_tool_loop(exec_chat, client, exec_parts, GEMINI_TOOL, on_422=_on_422, planned_keys=planned_keys)
    logger.info("Execute phase: %d turns", exec_turns)

    logger.info(
        "Agent done | total=%.1fs | plan=%d turns | exec=%d turns",
        time.time() - start, plan_turns, exec_turns,
    )

    # -------------------------------------------------------------------------
    # Persist rules for any 422s — runs in background, doesn't block response
    # -------------------------------------------------------------------------
    if _pending_422:
        logger.info("Scheduling async rule write for %d 422 error(s)", len(_pending_422))
        threading.Thread(
            target=_write_rules_async,
            args=(settings.gemini_fast_model, list(_pending_422)),
            daemon=True,
        ).start()
