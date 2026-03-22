"""
Tools for Tripletex API discovery backed by the local openapi.json spec.

- list_endpoints: search paths by tag or keyword
- get_endpoint_schema: return request body field names for a path+method
"""

import json
import logging
import pathlib
import re

from vertexai.generative_models import FunctionDeclaration

logger = logging.getLogger(__name__)

_SPEC: dict | None = None
_SPEC_PATH = pathlib.Path(__file__).parent.parent / "openapi.json"


def _get_spec() -> dict:
    global _SPEC
    if _SPEC is None:
        _SPEC = json.loads(_SPEC_PATH.read_text())
    return _SPEC


def _resolve_ref(spec: dict, ref: str) -> dict:
    """Follow a $ref string like '#/components/schemas/Foo' to its schema dict."""
    node = spec
    for part in ref.lstrip("#/").split("/"):
        node = node[part]
    return node


def _collect_properties(spec: dict, schema: dict) -> dict:
    """Return merged top-level properties dict, handling allOf."""
    props: dict = {}
    if "properties" in schema:
        props.update(schema["properties"])
    for item in schema.get("allOf", []):
        if "$ref" in item:
            props.update(_collect_properties(spec, _resolve_ref(spec, item["$ref"])))
        elif "properties" in item:
            props.update(item["properties"])
    return props


def _schema_for_operation(spec: dict, path: str, method: str) -> tuple[str | None, dict | None]:
    """Return (schema_name, schema_dict) for the requestBody of a path+method, or (None, None)."""
    ops = spec["paths"].get(path, {})
    op = ops.get(method.lower())
    if not op:
        return None, None
    content = op.get("requestBody", {}).get("content", {})
    for ct_val in content.values():
        s = ct_val.get("schema", {})
        if "$ref" in s:
            name = s["$ref"].split("/")[-1]
            return name, _resolve_ref(spec, s["$ref"])
        if "properties" in s:
            return None, s
    return None, None


def _match_spec_path(spec: dict, endpoint: str) -> str | None:
    """Find the spec path key that matches a concrete endpoint like /employee/123."""
    paths = spec["paths"]
    if endpoint in paths:
        return endpoint

    # Strip query string
    endpoint = endpoint.split("?")[0]
    if endpoint in paths:
        return endpoint

    # Try template matching: replace numeric segments with {id}-style placeholders
    segments = endpoint.split("/")
    for spec_path in paths:
        spec_segs = spec_path.split("/")
        if len(spec_segs) != len(segments):
            continue
        match = True
        for s, e in zip(spec_segs, segments):
            if s.startswith("{") and s.endswith("}"):
                continue  # wildcard
            if s != e:
                match = False
                break
        if match:
            return spec_path
    return None


# ---------------------------------------------------------------------------
# Function declarations
# ---------------------------------------------------------------------------

_TOP_LEVEL_TAGS = (
    "activity, asset, attestation, bank, balanceSheet, company, contact, country, "
    "currency, customer, deliveryAddress, department, division, document, "
    "documentArchive, employee, event, incomingInvoice, inventory, invoice, "
    "invoiceRemark, ledger, municipality, order, pension, penneo, pickupPoint, "
    "product, project, purchaseOrder, reminder, resultbudget, saft, subscription, "
    "supplier, supplierCustomer, supplierInvoice, transportType, travelExpense, "
    "userLicense, vatTermSizeSettings, voucherApprovalListElement, voucherInbox, "
    "voucherMessage, voucherStatus, yearEnd"
)

LIST_ENDPOINTS = FunctionDeclaration(
    name="list_endpoints",
    description=(
        "List available Tripletex API endpoints from the OpenAPI spec. "
        "Returns path, HTTP method, and summary for each matching operation. "
        "You MUST provide at least one of tag or query — calling with no filters is not allowed. "
        "Use tag= for top-level entity groups. "
        f"Valid top-level tags: {_TOP_LEVEL_TAGS}. "
        "Sub-resources (e.g. ledger/voucher, employee/employment, travelExpense/mileageAllowance, "
        "project/task, salary/payslip, timesheet/entry) are not listed as tags — use query= to find them. "
        "Examples: list_endpoints(tag='invoice'), list_endpoints(query='mileage'), "
        "list_endpoints(tag='travelExpense', query='approve')."
    ),
    parameters={
        "type": "object",
        "properties": {
            "tag": {
                "type": "string",
                "description": "Filter by top-level API tag (see description for valid values)",
            },
            "query": {
                "type": "string",
                "description": "Keyword to search in path or summary (case-insensitive). Use for sub-resources.",
            },
        },
    },
)

GET_ENDPOINT_SCHEMA = FunctionDeclaration(
    name="get_endpoint_schema",
    description=(
        "Get the exact request body field names and types for a Tripletex endpoint. "
        "Call this before using call_api with POST or PUT to know the correct field names — "
        "avoids 422 errors from hallucinated or wrong field names."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "API path starting with / (e.g. /travelExpense, /employee)",
            },
            "method": {
                "type": "string",
                "description": "HTTP method: GET, POST, PUT, or DELETE",
            },
        },
        "required": ["path", "method"],
    },
)


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


def list_endpoints(client=None, tag: str = None, query: str = None, **_) -> dict:
    if not tag and not query:
        return {
            "error": "You must provide at least one filter (tag or query). "
            f"Valid top-level tags: {_TOP_LEVEL_TAGS}"
        }

    spec = _get_spec()
    results = []
    q = query.lower() if query else None

    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            if not isinstance(op, dict):
                continue
            summary = op.get("summary", "")
            tags = op.get("tags", [])

            if tag and not any(t.lower() == tag.lower() for t in tags):
                continue
            if q and q not in path.lower() and q not in summary.lower():
                continue

            results.append({
                "path": path,
                "method": method.upper(),
                "summary": summary,
                "tags": tags,
            })

            if len(results) >= 100:
                break
        if len(results) >= 100:
            break

    return {"count": len(results), "results": results}


def get_endpoint_schema(client=None, path: str = None, method: str = None, endpoint: str = None, **_) -> dict:
    path = path or endpoint  # accept either parameter name
    spec = _get_spec()

    spec_path = _match_spec_path(spec, path)
    if not spec_path:
        return {"error": f"Path '{path}' not found in spec"}

    schema_name, schema = _schema_for_operation(spec, spec_path, method)
    if schema is None:
        return {
            "path": spec_path,
            "method": method.upper(),
            "note": "No request body for this operation",
            "fields": [],
        }

    props = _collect_properties(spec, schema)
    fields = []
    for name, prop in props.items():
        field: dict = {"name": name}
        if "$ref" in prop:
            ref_name = prop["$ref"].split("/")[-1]
            field["type"] = f"object ({ref_name})"
        elif "type" in prop:
            field["type"] = prop["type"]
        if "description" in prop:
            field["description"] = prop["description"]
        if "enum" in prop:
            field["enum"] = prop["enum"]
        fields.append(field)

    return {
        "path": spec_path,
        "method": method.upper(),
        "schema_name": schema_name,
        "fields": fields,
    }
