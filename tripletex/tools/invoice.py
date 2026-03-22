from vertexai.generative_models import FunctionDeclaration

from client import TripletexClient

# --- Function declarations ---

LIST_INVOICES = FunctionDeclaration(
    name="list_invoices",
    description="List invoices in Tripletex. Returns id, invoiceNumber, customer, invoiceDate, amountCurrency.",
    parameters={
        "type": "object",
        "properties": {
            "customer_id": {"type": "integer", "description": "Filter by customer ID"},
            "invoice_date_from": {"type": "string", "description": "From date (YYYY-MM-DD)"},
            "invoice_date_to": {"type": "string", "description": "To date (YYYY-MM-DD)"},
        },
    },
)

CREATE_INVOICE = FunctionDeclaration(
    name="create_invoice",
    description=(
        "Create and send an invoice in Tripletex. Requires an existing order ID. "
        "Returns the new invoice ID."
    ),
    parameters={
        "type": "object",
        "properties": {
            "order_id": {"type": "integer", "description": "Tripletex order ID to invoice"},
            "invoice_date": {"type": "string", "description": "Invoice date (YYYY-MM-DD)"},
            "invoice_due_date": {"type": "string", "description": "Payment due date (YYYY-MM-DD) — REQUIRED, typically 14-30 days after invoice_date"},
            "send_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": "How to send: ['EMAIL'], ['EHF'], or omit to not send",
            },
        },
        "required": ["order_id", "invoice_date", "invoice_due_date"],
    },
)

REGISTER_PAYMENT = FunctionDeclaration(
    name="register_payment",
    description="Register a payment against an invoice. Uses PUT /invoice/{id}/:payment.",
    parameters={
        "type": "object",
        "properties": {
            "invoice_id": {"type": "integer", "description": "Tripletex invoice ID"},
            "amount": {"type": "number", "description": "Payment amount"},
            "payment_date": {"type": "string", "description": "Payment date (YYYY-MM-DD)"},
            "payment_type_id": {
                "type": "integer",
                "description": "Payment type ID (use list_payment_types to find valid IDs)",
            },
        },
        "required": ["invoice_id", "amount", "payment_date", "payment_type_id"],
    },
)

LIST_PAYMENT_TYPES = FunctionDeclaration(
    name="list_payment_types",
    description="List available payment types in Tripletex. Use to find the correct payment_type_id.",
    parameters={"type": "object", "properties": {}},
)

# --- Implementations ---


def list_invoices(
    client: TripletexClient,
    customer_id: int = None,
    invoice_date_from: str = None,
    invoice_date_to: str = None,
    **_,
) -> dict:
    params = {
        "fields": "id,invoiceNumber,customer,invoiceDate,amountCurrency,amountOutstanding",
        "count": 100,
        "invoiceDateFrom": invoice_date_from or "1900-01-01",
        "invoiceDateTo": invoice_date_to or "2099-12-31",
    }
    if customer_id:
        params["customerId"] = customer_id
    return client.get("/invoice", params=params)


def create_invoice(
    client: TripletexClient,
    order_id: int,
    invoice_date: str,
    invoice_due_date: str,
    send_types: list = None,
    **_,
) -> dict:
    if not order_id:
        raise ValueError("order_id is required")
    if not invoice_date:
        raise ValueError("invoice_date is required (YYYY-MM-DD)")
    if not invoice_due_date:
        raise ValueError("invoice_due_date is required (YYYY-MM-DD)")

    body = {
        "invoiceDate": invoice_date,
        "invoiceDueDate": invoice_due_date,
        "orders": [{"id": order_id}],
    }
    result = client.post("/invoice", json=body)

    # Send the invoice via the separate :send endpoint if send_types requested
    if send_types:
        invoice_id = (result.get("value") or {}).get("id") or result.get("id")
        if invoice_id:
            for send_type in send_types:
                client.put(f"/invoice/{invoice_id}/:send", params={"sendType": send_type})

    return result


def register_payment(
    client: TripletexClient,
    invoice_id: int,
    amount: float,
    payment_date: str,
    payment_type_id: int,
    **_,
) -> dict:
    if not invoice_id:
        raise ValueError("invoice_id is required")
    if amount is None or amount <= 0:
        raise ValueError("amount must be positive")
    if not payment_date:
        raise ValueError("payment_date is required (YYYY-MM-DD)")
    if not payment_type_id:
        raise ValueError("payment_type_id is required — use list_payment_types to find valid IDs")

    return client.put(
        f"/invoice/{invoice_id}/:payment",
        params={
            "paymentDate": payment_date,
            "paidAmount": amount,
            "paymentTypeId": payment_type_id,
        },
    )


def list_payment_types(client: TripletexClient, **_) -> dict:
    return client.get("/invoice/paymentType", params={"fields": "id,description", "count": 100})
