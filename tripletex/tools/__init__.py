"""
Collects all Tripletex tool definitions and implementations.

- GEMINI_TOOL: full tool set for the execution phase
- PLAN_TOOL: discovery + notes tools for the plan phase
- TOOL_MAP: maps function name → callable(client, **kwargs)
"""

from vertexai.generative_models import Tool

from tools.call_api import CALL_API, call_api
from tools.schema_tools import (
    GET_ENDPOINT_SCHEMA,
    LIST_ENDPOINTS,
    get_endpoint_schema,
    list_endpoints,
)
from tools.knowledge_tools import (
    GET_ENDPOINT_NOTES,
    UPDATE_ENDPOINT_NOTES,
    get_endpoint_notes,
    update_endpoint_notes,
)

# Execution phase: call_api + discovery + schema only
# (get_endpoint_notes removed — notes are pre-loaded in context block)
ALL_DECLARATIONS = [
    CALL_API,
    LIST_ENDPOINTS,
    GET_ENDPOINT_SCHEMA,
]
GEMINI_TOOL = Tool(function_declarations=ALL_DECLARATIONS)

# Plan phase: discovery + notes only (no API calls)
PLAN_DECLARATIONS = [
    LIST_ENDPOINTS,
    GET_ENDPOINT_NOTES,
]
PLAN_TOOL = Tool(function_declarations=PLAN_DECLARATIONS)

TOOL_MAP: dict = {
    "call_api": call_api,
    "list_endpoints": list_endpoints,
    "get_endpoint_schema": get_endpoint_schema,
    "get_endpoint_notes": get_endpoint_notes,
    "update_endpoint_notes": update_endpoint_notes,
}
