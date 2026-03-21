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
            "invoice_due_date": {"type": "string", "description": "Payment due date (YYYY-MM-DD)"},
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
    description="Register a payment against an invoice.",
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
    }
    if customer_id:
        params["customerId"] = customer_id
    if invoice_date_from:
        params["invoiceDateFrom"] = invoice_date_from
    if invoice_date_to:
        params["invoiceDateTo"] = invoice_date_to
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
    if send_types:
        body["sendTypes"] = send_types

    return client.post("/invoice", json=body)


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

    return client.post(
        f"/invoice/{invoice_id}/:createReminder",
        json={
            "amount": amount,
            "paymentDate": payment_date,
            "paymentType": {"id": payment_type_id},
        },
    )


def list_payment_types(client: TripletexClient, **_) -> dict:
    return client.get("/invoice/paymentType", params={"fields": "id,name", "count": 100})
