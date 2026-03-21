from vertexai.generative_models import FunctionDeclaration

from client import TripletexClient

# --- Function declarations ---

CREATE_ORDER = FunctionDeclaration(
    name="create_order",
    description=(
        "Create a new order in Tripletex. Orders are a prerequisite for invoices. "
        "Returns the new order ID."
    ),
    parameters={
        "type": "object",
        "properties": {
            "customer_id": {"type": "integer", "description": "Tripletex customer ID"},
            "order_date": {"type": "string", "description": "Order date (YYYY-MM-DD)"},
            "delivery_date": {"type": "string", "description": "Delivery date (YYYY-MM-DD)"},
            "order_lines": {
                "type": "array",
                "description": "List of order lines",
                "items": {
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "integer"},
                        "description": {"type": "string"},
                        "count": {"type": "number"},
                        "unit_price_ex_vat": {"type": "number"},
                    },
                },
            },
        },
        "required": ["customer_id", "order_date"],
    },
)

# --- Implementations ---


def create_order(
    client: TripletexClient,
    customer_id: int,
    order_date: str,
    delivery_date: str = None,
    order_lines: list = None,
    **_,
) -> dict:
    if not customer_id:
        raise ValueError("customer_id is required")
    if not order_date:
        raise ValueError("order_date is required (YYYY-MM-DD)")

    body = {
        "customer": {"id": customer_id},
        "orderDate": order_date,
    }
    if delivery_date:
        body["deliveryDate"] = delivery_date
    if order_lines:
        body["orderLines"] = [
            {
                **({"product": {"id": line["product_id"]}} if "product_id" in line else {}),
                **({"description": line["description"]} if "description" in line else {}),
                **({"count": line["count"]} if "count" in line else {}),
                **(
                    {"unitPriceExVat": line["unit_price_ex_vat"]}
                    if "unit_price_ex_vat" in line
                    else {}
                ),
            }
            for line in order_lines
        ]

    return client.post("/order", json=body)
