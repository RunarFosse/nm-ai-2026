import json
import logging

from vertexai.generative_models import FunctionDeclaration

from client import TripletexClient
from tools.schema_tools import _get_spec, _match_spec_path, _schema_for_operation, _collect_properties

logger = logging.getLogger(__name__)

CALL_API = FunctionDeclaration(
    name="call_api",
    description=(
        "Generic Tripletex API call. Use this for any endpoint not covered by a dedicated tool, "
        "or when you need precise control over the request body. "
        "Endpoint must start with / (e.g. /travelExpense, /ledger/voucher/123). "
        "method: GET, POST, PUT, or DELETE. "
        "Call get_endpoint_schema first for POST/PUT to get the correct field names."
    ),
    parameters={
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "description": "HTTP method: GET, POST, PUT, or DELETE",
            },
            "endpoint": {
                "type": "string",
                "description": "API path starting with / (e.g. /employee/123, /travelExpense)",
            },
            "body": {
                "type": "object",
                "description": "JSON request body for POST/PUT requests",
            },
            "params": {
                "type": "object",
                "description": "Query parameters as key-value pairs (e.g. fields, count, from)",
            },
        },
        "required": ["method", "endpoint"],
    },
)


def _validate_body(endpoint: str, method: str, body: dict) -> dict:
    """Strip unknown top-level fields from body using openapi.json schema. Returns cleaned body."""
    spec = _get_spec()
    spec_path = _match_spec_path(spec, endpoint)
    if not spec_path:
        return body

    _, schema = _schema_for_operation(spec, spec_path, method)
    if schema is None:
        return body

    valid_fields = set(_collect_properties(spec, schema).keys())
    if not valid_fields:
        return body

    stripped = {k for k in body if k not in valid_fields}
    if stripped:
        logger.warning(
            "call_api: stripping unknown fields %s from %s %s (not in spec schema)",
            stripped, method.upper(), endpoint,
        )
    return {k: v for k, v in body.items() if k in valid_fields}


def call_api(
    client: TripletexClient,
    method: str,
    endpoint: str = None,
    body=None,
    params: dict = None,
    path: str = None,
    query_params: dict = None,
    **_,
):
    # Accept 'path' as alias for 'endpoint'
    if endpoint is None:
        endpoint = path
    # Accept 'query_params' as alias for 'params'
    if params is None and query_params:
        params = query_params
    if not endpoint:
        raise ValueError("endpoint (or path) is required")
    method = method.upper()
    # Normalise body: accept JSON strings (model sometimes serialises before calling)
    if isinstance(body, str):
        try:
            body = json.loads(body) if body.strip() else None
        except Exception:
            body = None
    if body and method in ("POST", "PUT"):
        body = _validate_body(endpoint, method, body)

    if method == "GET":
        return client.get(endpoint, params=params)
    elif method == "POST":
        return client.post(endpoint, json=body, params=params)
    elif method == "PUT":
        return client.put(endpoint, json=body, params=params)
    elif method == "DELETE":
        client.delete(endpoint, params=params)
        return {"deleted": True}
    else:
        raise ValueError(f"Unsupported method: {method}")
