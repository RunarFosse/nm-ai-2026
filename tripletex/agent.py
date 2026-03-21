"""
Gemini orchestrator agent — two-phase architecture.

Phase 1: Planning (no tools) — LLM extracts task type, prerequisites, fields,
         and order of operations from the prompt.
Phase 2: Execution (tools enabled) — LLM executes the plan step by step,
         dispatching tool calls to the Tripletex API.
"""

import base64
import json
import logging

import vertexai
from vertexai.generative_models import Content, GenerativeModel, Part

from client import TripletexClient
from config import settings
from tools import GEMINI_TOOL, TOOL_MAP

logger = logging.getLogger(__name__)

PLANNING_SYSTEM_PROMPT = """You are an expert Tripletex accounting agent. Your job is to analyse an accounting task and produce a structured execution plan.

Output ONLY valid JSON with this structure:
{
  "task_type": "<e.g. create_employee, create_invoice, register_payment, delete_travel_expense>",
  "summary": "<one-line description of what needs to happen>",
  "prerequisites": ["<list of entities that must exist or be created first, e.g. 'department', 'customer', 'order'>"],
  "steps": [
    {"action": "<create|update|delete|list>", "entity": "<employee|customer|invoice|...>", "fields": {}}
  ],
  "notes": "<any special considerations, e.g. 'department required before employee', 'need payment type ID'>"
}

The task prompt may be in Norwegian, English, Spanish, Portuguese, German, French, or Nynorsk — understand it regardless.
Always identify prerequisites (department before employee; customer + order before invoice; find payment type before registering payment)."""

EXECUTION_SYSTEM_PROMPT = """You are an expert accounting agent for Tripletex, a Norwegian ERP system.
You have already analysed the task and produced an execution plan. Execute it precisely.

Critical rules:
- Execute steps in the order from the plan — prerequisites first
- GET calls are free and do not count against you — use them to confirm state and look up IDs
- Validate all inputs before any POST/PUT/DELETE — every 4xx error hurts your score
- If a write call fails, read the error message carefully and fix it in ONE retry — no blind retries
- After creating an entity, you can trust the ID from the response — no need to GET it again unless verifying fields
- When done, stop calling tools

Known Tripletex API field names (memorise these):
- Employee: firstName, lastName, email, phoneNumberMobile, userType="STANDARD" (required), department={"id": N} (required)
- Customer: name, email, phoneNumber, organizationNumber, isCustomer=true (required)
- Product: name, number, costExcludingVatCurrency (NOT priceExVat, NOT price)
- Order: customer={"id": N}, orderDate (YYYY-MM-DD), deliveryDate (required), orderLines=[{"product":{"id":N},"count":N,"unitPriceExcludingVatCurrency":N}]
- Invoice: requires an order — POST /invoice with orders=[{"id": order_id}], invoiceDate, invoiceDueDate
- Payment: use list_payment_types first to find a valid payment_type_id, then register_payment
- Voucher postings must sum to zero (debits = credits)
- TravelExpense tool params: employee_id, title (required), departure_date (required), return_date (required), purpose (optional)
- Project: startDate is required; projectManager must be an employee with project manager role in the account

Prerequisites (always check/create these first):
- Employee needs a department → list_departments first; if empty, create_department then use its ID
- Invoice needs customer + order → create_customer, create_order, then create_invoice
- Payment needs invoice → create_invoice first, then register_payment
- Invoice 422 "bankkontonummer" → GET /ledger/account?number=1920, then PUT /ledger/account/{id} with bankAccountNumber="12345678903" (any valid Norwegian account number), then retry the invoice

Using endpoints beyond the dedicated tools:
- Call list_endpoints(tag="...") or list_endpoints(query="...") to discover available paths
- Call get_endpoint_schema(path, method) BEFORE any call_api POST/PUT to get the exact field names
- call_api will silently strip fields not in the spec schema — always get the schema first to avoid data loss
"""


def run_agent(prompt: str, client: TripletexClient, files: list[dict] = None) -> None:
    """
    Run the two-phase Gemini orchestrator on the given accounting task.

    Args:
        prompt: Natural language task description (any supported language)
        client: Authenticated TripletexClient for the current submission
        files: Optional list of file dicts with keys: filename, content_base64, mime_type
    """
    vertexai.init(project=settings.gcp_project, location=settings.vertex_location)

    logger.info("Agent starting | model=%s | prompt=%.120s", settings.gemini_model, prompt)

    # Build user parts — decode and attach any files before the text prompt
    user_parts: list[Part] = []
    for f in (files or []):
        try:
            data = base64.b64decode(f["content_base64"])
            user_parts.append(Part.from_data(data=data, mime_type=f["mime_type"]))
            logger.info("Attached file: %s (%s, %d bytes)", f["filename"], f["mime_type"], len(data))
        except Exception as exc:
            logger.warning("Failed to decode file %s: %s", f.get("filename"), exc)
    user_parts.append(Part.from_text(prompt))

    # -------------------------------------------------------------------------
    # Phase 1: Parse & Plan (no tools — pure reasoning)
    # -------------------------------------------------------------------------
    planner = GenerativeModel(
        model_name=settings.gemini_model,
        system_instruction=PLANNING_SYSTEM_PROMPT,
    )
    plan_response = planner.generate_content(user_parts)
    plan_text = plan_response.text.strip()

    # Strip markdown code fences if the model wraps the JSON
    if plan_text.startswith("```"):
        lines = plan_text.split("\n")
        plan_text = "\n".join(lines[1:])
        if plan_text.rstrip().endswith("```"):
            plan_text = plan_text[: plan_text.rfind("```")]

    try:
        plan = json.loads(plan_text)
        logger.info("Plan: %s", json.dumps(plan, ensure_ascii=False))
    except json.JSONDecodeError:
        logger.warning("Planner returned non-JSON — continuing without structured plan")
        plan = {"summary": plan_text[:500]}

    # -------------------------------------------------------------------------
    # Phase 2: Execute (tool-calling loop)
    # -------------------------------------------------------------------------
    executor = GenerativeModel(
        model_name=settings.gemini_model,
        system_instruction=EXECUTION_SYSTEM_PROMPT,
    )
    chat = executor.start_chat(response_validation=False)

    execution_prompt = (
        f"{prompt}\n\n"
        f"---\nExecution plan:\n{json.dumps(plan, ensure_ascii=False, indent=2)}"
    )

    # If there were file attachments, send them again so the executor can read them
    if files:
        exec_parts = user_parts[:-1] + [Part.from_text(execution_prompt)]
    else:
        exec_parts = [Part.from_text(execution_prompt)]

    response = chat.send_message(exec_parts, tools=[GEMINI_TOOL])

    while True:
        function_calls = [
            part
            for part in response.candidates[0].content.parts
            if part.function_call and part.function_call.name
        ]

        if not function_calls:
            logger.info("Agent finished — no more tool calls")
            break

        function_responses: list[Part] = []
        for part in function_calls:
            fc = part.function_call
            fn_name = fc.name
            fn_args = dict(fc.args)

            logger.info("Tool call: %s(%s)", fn_name, fn_args)

            try:
                fn = TOOL_MAP[fn_name]
                result = fn(client, **fn_args)
                if result is None:
                    result = {"ok": True}
            except Exception as exc:
                logger.warning("Tool %s failed: %s", fn_name, exc)
                result = {"error": str(exc)}

            function_responses.append(
                Part.from_function_response(name=fn_name, response={"result": result})
            )

        response = chat.send_message(
            Content(role="user", parts=function_responses)
        )
