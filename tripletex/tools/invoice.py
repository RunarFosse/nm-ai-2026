from vertexai.generative_models import FunctionDeclaration

from client import TripletexClient

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

LIST_PAYMENT_TYPES = FunctionDeclaration(
    name="list_payment_types",
    description="List available payment types in Tripletex. Use to find the correct payment_type_id.",
    parameters={"type": "object", "properties": {}},
)


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


def list_payment_types(client: TripletexClient, **_) -> dict:
    return client.get("/invoice/paymentType", params={"fields": "id,description", "count": 100})
