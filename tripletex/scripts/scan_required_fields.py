"""
Empirically discover required fields for Tripletex POST/PUT endpoints
by sending iterative requests and collecting 422 validation messages.

Usage:
    cd tripletex
    python scripts/scan_required_fields.py

Outputs required_fields.py in the tripletex/ directory.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("TRIPLETEX_SANDBOX_URL", "").rstrip("/")
TOKEN = os.getenv("TRIPLETEX_SANDBOX_TOKEN", "")

if not BASE_URL or not TOKEN:
    sys.exit("Set TRIPLETEX_SANDBOX_URL and TRIPLETEX_SANDBOX_TOKEN in .env")

AUTH = ("0", TOKEN)

# Load spec once for filler generation
_SPEC = json.loads((Path(__file__).parent.parent / "openapi.json").read_text())


def _resolve_ref(ref: str) -> dict:
    node = _SPEC
    for part in ref.lstrip("#/").split("/"):
        node = node[part]
    return node


def _filler_from_schema(prop: dict) -> object:
    """Generate a minimal valid filler value from an OpenAPI property schema."""
    # Resolve $ref
    if "$ref" in prop:
        prop = _resolve_ref(prop["$ref"])

    # allOf — merge and use first
    if "allOf" in prop:
        for item in prop["allOf"]:
            resolved = _resolve_ref(item["$ref"]) if "$ref" in item else item
            val = _filler_from_schema(resolved)
            if val is not None:
                return val

    typ = prop.get("type")
    fmt = prop.get("format", "")
    enum = prop.get("enum")

    if enum:
        return enum[0]
    if typ == "string":
        if "date" in fmt or "date" in prop.get("description", "").lower():
            return "2026-01-01"
        return "Test"
    if typ == "integer":
        return 1
    if typ == "number":
        return 1.0
    if typ == "boolean":
        return True
    if typ == "array":
        return []
    if typ == "object":
        return {}
    # Object ref without explicit type (e.g. Department, Country)
    if "properties" in prop:
        # Return {id: 1} for ref objects — most Tripletex refs just need an id
        if "id" in prop.get("properties", {}):
            return {"id": 1}
        return {}
    return None


def _get_schema_for_endpoint(path: str, method: str) -> dict | None:
    """Return the properties dict for a POST/PUT endpoint's request body."""
    ops = _SPEC["paths"].get(path, {})
    op = ops.get(method.lower(), {})
    content = op.get("requestBody", {}).get("content", {})
    for ct_val in content.values():
        s = ct_val.get("schema", {})
        if "$ref" in s:
            resolved = _resolve_ref(s["$ref"])
            props = resolved.get("properties", {})
            for item in resolved.get("allOf", []):
                r = _resolve_ref(item["$ref"]) if "$ref" in item else item
                props.update(r.get("properties", {}))
            return props
        if "properties" in s:
            return s["properties"]
    return None


# Endpoints to scan — path, method, and a minimal seed body to get past the
# first validation layer (some fields only become required after others exist)
ENDPOINTS = [
    # (path, method, seed_body)
    ("/employee",           "POST", {}),
    ("/customer",           "POST", {}),
    ("/product",            "POST", {}),
    ("/order",              "POST", {}),
    ("/invoice",            "POST", {}),
    ("/travelExpense",      "POST", {}),
    ("/project",            "POST", {}),
    ("/department",         "POST", {}),
    ("/ledger/voucher",     "POST", {}),
    ("/supplier",           "POST", {}),
    ("/supplierInvoice",    "POST", {}),
    ("/purchaseOrder",      "POST", {}),
    ("/timesheet/entry",    "POST", {}),
    ("/salary/type",        "POST", {}),
]

# Fields to try injecting to get past the first layer and reveal deeper requirements
# Maps endpoint path → dict of plausible filler values
FILLERS = {
    "/employee": {
        "firstName": "Test",
        "lastName": "Scan",
        "userType": "STANDARD",
        "department": {"id": 1},
    },
    "/order": {
        "customer": {"id": 1},
        "orderDate": "2026-01-01",
        "deliveryDate": "2026-01-01",
    },
    "/invoice": {
        "orders": [{"id": 1}],
        "invoiceDate": "2026-01-01",
        "invoiceDueDate": "2026-01-31",
    },
    "/travelExpense": {
        "employee": {"id": 1},
        "title": "Test",
        "travelDetails": {"departureDate": "2026-01-01", "returnDate": "2026-01-02"},
    },
    "/project": {
        "name": "Test",
        "startDate": "2026-01-01",
    },
    "/ledger/voucher": {
        "date": "2026-01-01",
        "postings": [],
    },
}


def post(path: str, body: dict) -> dict:
    url = f"{BASE_URL}{path}"
    r = requests.post(url, json=body, auth=AUTH, timeout=10)
    return r.status_code, r.json() if r.content else {}


def put(path: str, body: dict) -> dict:
    url = f"{BASE_URL}{path}"
    r = requests.put(url, json=body, auth=AUTH, timeout=10)
    return r.status_code, r.json() if r.content else {}


def extract_required(response_body: dict) -> list[str]:
    """Pull field names from 422 validation messages."""
    msgs = response_body.get("validationMessages") or []
    fields = []
    for m in msgs:
        field = m.get("field")
        message = m.get("message", "")
        if field and any(kw in message.lower() for kw in
                         ["null", "fylles", "required", "must", "obligat", "mangler"]):
            # field may be compound like "orderGroups.orderLines.product.id,orderLines.product.id"
            # take the shortest path (most top-level)
            parts = field.split(",")
            shortest = min(parts, key=lambda x: x.count("."))
            top = shortest.split(".")[0]
            if top not in fields:
                fields.append(top)
    return fields


def scan_endpoint(path: str, method: str, seed: dict) -> list[str]:
    print(f"\n{'─'*60}")
    print(f"Scanning {method} {path}")

    schema_props = _get_schema_for_endpoint(path, method) or {}
    required: list[str] = []
    body = dict(seed)
    fn = post if method == "POST" else put

    for iteration in range(6):
        status, resp = fn(path, body)
        print(f"  [{iteration}] {status} body={json.dumps(body)[:120]}")

        if status in (200, 201):
            print(f"  → Success — no more required fields")
            break
        if status == 422:
            new_fields = extract_required(resp)
            if not new_fields:
                msgs = resp.get("validationMessages") or []
                print(f"  → 422 with no field errors: {[m.get('message') for m in msgs]}")
                break
            added = [f for f in new_fields if f not in required]
            if not added:
                print(f"  → No new required fields, stopping")
                break
            print(f"  → Required: {added}")
            required.extend(added)
            # Fill from FILLERS first, then spec, then last resort
            fillers = FILLERS.get(path, {})
            for f in added:
                if f in body:
                    continue
                if f in fillers:
                    body[f] = fillers[f]
                elif f in schema_props:
                    val = _filler_from_schema(schema_props[f])
                    body[f] = val if val is not None else "Test"
                else:
                    body[f] = "Test"
        elif status == 400:
            print(f"  → 400: {resp.get('message', '')}")
            break
        elif status == 404:
            print(f"  → 404 — endpoint may not exist in sandbox")
            break
        else:
            print(f"  → {status}: {str(resp)[:200]}")
            break

        time.sleep(0.3)

    return required


def main():
    results: dict[str, list[str]] = {}

    for path, method, seed in ENDPOINTS:
        key = f"{method} {path}"
        required = scan_endpoint(path, method, seed)
        results[key] = required
        print(f"  RESULT {key}: {required}")

    # Write required_fields.py
    out_path = Path(__file__).parent.parent / "required_fields.py"
    lines = [
        '"""',
        "Empirically discovered required fields for Tripletex API endpoints.",
        "Generated by scripts/scan_required_fields.py — re-run to refresh.",
        '"""',
        "",
        "REQUIRED_FIELDS: dict[str, list[str]] = {",
    ]
    for key, fields in results.items():
        lines.append(f'    "{key}": {json.dumps(fields)},')
    lines += ["}", ""]

    out_path.write_text("\n".join(lines))
    print(f"\n✓ Written to {out_path}")

    # Also print a summary
    print("\nSummary:")
    for key, fields in results.items():
        print(f"  {key}: {fields}")


if __name__ == "__main__":
    main()
