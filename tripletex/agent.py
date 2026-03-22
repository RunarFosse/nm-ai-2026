"""
Two-phase Tripletex agent.

Phase 1: Research (read-only tools, fast model) — discover what exists, find endpoints/schemas, check deps.
Phase 2: Execute (all tools, pro model) — use research findings to complete the task.

A 5-minute timer runs from entry and is visible to both phases.
"""

import base64
import json
import logging
import time

import vertexai
from vertexai.generative_models import Content, GenerativeModel, Part

from client import TripletexClient
from config import settings
from tools import GEMINI_TOOL, RESEARCH_TOOL, TOOL_MAP

logger = logging.getLogger(__name__)

TOTAL_BUDGET = 300  # seconds
RESEARCH_MAX_TURNS = 15

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

RESEARCH_SYSTEM_PROMPT = """You are a Tripletex research agent. Your ONLY job is to gather information — do NOT create, update, or delete anything.

Dedicated tools already exist for these entities — do NOT use list_endpoints or get_endpoint_schema for them:
employee, customer, product, order, invoice, travelExpense, project, department, ledger/voucher, ledger/account

For the given task:
1. Identify what entities need to be created or modified
2. Check what already exists by listing relevant entities — ONLY if the task references existing data (e.g. "add phone to Kari", "delete travel expense", "register payment for invoice")
3. For operations NOT covered by a dedicated tool, use list_endpoints + get_endpoint_schema to find the correct path and fields
4. Check critical dependencies: department exists for employee? bank account registered for invoicing?
5. When you have enough information, stop calling tools and output your findings

Skip research entirely if the task is purely creation of new entities with no existing data needed.

Output ONLY valid JSON (no markdown fences):
{
  "entities_needed": ["list of entity types to create or modify"],
  "entities_found": {"customers": [...], "employees": [...], "departments": [...], ...},
  "endpoints": {"operation_name": {"path": "...", "method": "...", "fields": [...]}},
  "dependencies_met": {"department": true, "bank_account": true, ...},
  "missing": ["things that must be created as prerequisites"],
  "notes": "anything unusual or important"
}"""

EXECUTION_SYSTEM_PROMPT = """You are a Tripletex execution agent for a Norwegian ERP system.
You have research findings from a prior discovery phase. Use them to complete the task efficiently.

Critical rules:
- Use IDs and field names from research findings — don't re-fetch what research already found
- Use dedicated tools (create_employee, create_order, create_invoice, etc.) for known entities — they handle field mapping correctly
- create_order order_lines must include unit_price_ex_vat — never omit the price
- For unfamiliar endpoints use list_endpoints/get_endpoint_schema then call_api
- If a write fails, read the error and fix it in ONE retry — no blind retries
- Use list_invoices (not call_api GET /invoice) to list invoices
- Before register_payment, confirm the invoice has amountOutstanding > 0
- If invoice POST fails with bank account error: GET /ledger/account?number=1920, PUT /ledger/account/{id} with bankAccountNumber="12345678903", retry
- When done, stop calling tools

Time awareness:
- If remaining time < 60s: skip verification GETs, execute writes only
- If remaining time < 20s: stop immediately"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
) -> tuple[str | None, int]:
    """
    Send initial_parts, then dispatch tool calls until the model stops.
    Returns (final_text, turns_used).
    """
    response = chat.send_message(initial_parts, tools=[tool])
    turns = 0

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
            try:
                result = _dispatch_tool(client, fn_name, fn_args)
                if result is None:
                    result = {"ok": True}
            except Exception as exc:
                logger.warning("Tool %s failed: %s", fn_name, exc)
                result = {"error": str(exc)}
            function_responses.append(
                Part.from_function_response(name=fn_name, response={"result": result})
            )

        response = chat.send_message(Content(role="user", parts=function_responses))

    try:
        return response.text.strip(), turns
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_agent(prompt: str, client: TripletexClient, files: list[dict] = None) -> None:
    vertexai.init(project=settings.gcp_project, location=settings.vertex_location)
    start = time.time()

    logger.info("Agent starting | model=%s | prompt=%.120s", settings.gemini_model, prompt)

    # -------------------------------------------------------------------------
    # Phase 1: Research (fast model — speed over reasoning depth)
    # -------------------------------------------------------------------------
    researcher = GenerativeModel(
        model_name=settings.gemini_fast_model,
        system_instruction=RESEARCH_SYSTEM_PROMPT,
    )
    research_chat = researcher.start_chat(response_validation=False)
    research_prompt = f"{_time_context(start)}\n\n{prompt}"
    research_parts = _build_user_parts(files, research_prompt)

    research_text, research_turns = _run_tool_loop(
        research_chat, client, research_parts, RESEARCH_TOOL, max_turns=RESEARCH_MAX_TURNS
    )

    if research_turns == 0:
        # Fast path: no tool calls — simple task, skip to planning with empty findings
        logger.info("Research phase: no tool calls (fast path)")
        research_findings = {}
    else:
        logger.info("Research phase: %d turns", research_turns)
        research_findings = _parse_json(research_text)
        logger.info("Research findings: %s", json.dumps(research_findings, ensure_ascii=False)[:500])

    # -------------------------------------------------------------------------
    # Phase 2: Execute
    # -------------------------------------------------------------------------
    executor = GenerativeModel(
        model_name=settings.gemini_model,
        system_instruction=EXECUTION_SYSTEM_PROMPT,
    )
    exec_chat = executor.start_chat(response_validation=False)
    exec_prompt = (
        f"{_time_context(start)}\n\n"
        f"Task: {prompt}\n\n"
        f"Research findings:\n{json.dumps(research_findings, ensure_ascii=False, indent=2)}"
    )
    exec_parts = _build_user_parts(files, exec_prompt)

    _, exec_turns = _run_tool_loop(exec_chat, client, exec_parts, GEMINI_TOOL)
    logger.info("Execute phase: %d turns", exec_turns)
    logger.info(
        "Agent done | total=%.1fs | research=%d turns | exec=%d turns",
        time.time() - start, research_turns, exec_turns,
    )
