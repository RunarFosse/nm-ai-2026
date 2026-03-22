"""
Tools for reading and writing endpoint-specific notes in the SQLite knowledge base.
"""

import re

from vertexai.generative_models import FunctionDeclaration

import knowledge

_NUMERIC_SEGMENT = re.compile(r"/\d+")


def _normalize(endpoint: str) -> str:
    """Replace numeric path segments with {id} so keys are generic."""
    return _NUMERIC_SEGMENT.sub("/{id}", endpoint)


GET_ENDPOINT_NOTES = FunctionDeclaration(
    name="get_endpoint_notes",
    description=(
        "Get accumulated knowledge and rules for a Tripletex endpoint from the "
        "knowledge base. Call this for any endpoint you are about to use to learn "
        "about required fields, dependencies, and gotchas discovered from past runs. "
        "Returns the notes text, or null if no notes exist yet."
    ),
    parameters={
        "type": "object",
        "properties": {
            "endpoint": {
                "type": "string",
                "description": "Endpoint key, e.g. 'POST /invoice' or 'PUT /ledger/account/{id}'",
            }
        },
        "required": ["endpoint"],
    },
)

UPDATE_ENDPOINT_NOTES = FunctionDeclaration(
    name="update_endpoint_notes",
    description=(
        "Update the knowledge base with a rule or dependency discovered for this endpoint. "
        "Always include the FULL updated notes text (existing notes + new finding). "
        "Use this when you encounter a 422 error or discover an undocumented dependency."
    ),
    parameters={
        "type": "object",
        "properties": {
            "endpoint": {
                "type": "string",
                "description": "Endpoint key, e.g. 'POST /invoice'",
            },
            "notes": {
                "type": "string",
                "description": "Full updated notes text for this endpoint (existing + new finding)",
            },
        },
        "required": ["endpoint", "notes"],
    },
)


def get_endpoint_notes(client=None, endpoint: str = None, path: str = None, method: str = None, **_) -> dict:
    if not endpoint and path and method:
        endpoint = f"{method.upper()} {path}"
    endpoint = _normalize(endpoint)
    notes = knowledge.get_notes(endpoint)
    if notes is None:
        return {"endpoint": endpoint, "notes": None, "message": "No notes found"}
    return {"endpoint": endpoint, "notes": notes}


def update_endpoint_notes(client=None, endpoint: str = None, notes: str = None, **_) -> dict:
    endpoint = _normalize(endpoint)
    knowledge.upsert_notes(endpoint, notes)
    return {"ok": True, "endpoint": endpoint}
