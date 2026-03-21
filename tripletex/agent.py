"""
Gemini orchestrator agent.

Runs a function-calling loop until the model stops invoking tools.
Each tool call is dispatched to the corresponding implementation in tools/,
with the TripletexClient injected automatically.
"""

import logging

import vertexai
from vertexai.generative_models import Content, GenerativeModel, Part

from client import TripletexClient
from config import settings
from tools import GEMINI_TOOL, TOOL_MAP

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert accounting agent for Tripletex, a Norwegian accounting system.
Your job is to complete accounting tasks accurately and efficiently.

Rules:
- Read the task carefully before making any write calls (POST/PUT/DELETE)
- Use GET calls freely to understand the current state
- Avoid trial-and-error — validate your understanding before writing
- If you need a prerequisite resource (e.g. a customer before an invoice), create it first
- When the task is complete, stop calling tools
- Respond only in the context of accounting tasks — do not perform unrelated actions
"""


def run_agent(prompt: str, client: TripletexClient) -> None:
    """
    Run the Gemini orchestrator on the given accounting task prompt.

    Args:
        prompt: Natural language task description (any supported language)
        client: Authenticated TripletexClient for the current submission
    """
    vertexai.init(project=settings.gcp_project, location=settings.vertex_location)
    model = GenerativeModel(
        model_name=settings.gemini_model,
        system_instruction=SYSTEM_PROMPT,
    )
    chat = model.start_chat()

    logger.info("Agent starting | model=%s | prompt=%s", settings.gemini_model, prompt[:120])

    response = chat.send_message(prompt, tools=[GEMINI_TOOL])

    while True:
        function_calls = [
            part
            for part in response.candidates[0].content.parts
            if part.function_call.name  # empty name means no function call
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
