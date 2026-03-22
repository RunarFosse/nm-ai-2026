from vertexai.generative_models import FunctionDeclaration

from client import TripletexClient

LIST_PRODUCTS = FunctionDeclaration(
    name="list_products",
    description="List products in Tripletex. Returns id, name, number, costExcludingVatCurrency, vatType.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Filter by name (partial match)"},
        },
    },
)

LIST_VAT_TYPES = FunctionDeclaration(
    name="list_vat_types",
    description=(
        "List available VAT types in Tripletex. Returns id, number, name, percentage. "
        "Call this before creating a product when a specific VAT rate is needed — VAT type IDs "
        "are account-specific and must be looked up at runtime. "
        "Typical Norwegian rates: 25% (høy sats), 15% (middels), 12% (lav), 0% (fritatt)."
    ),
    parameters={"type": "object", "properties": {}},
)


def list_products(client: TripletexClient, name: str = None, **_) -> dict:
    params = {"fields": "id,name,number,costExcludingVatCurrency,vatType", "count": 100}
    if name:
        params["name"] = name
    return client.get("/product", params=params)


def list_vat_types(client: TripletexClient, **_) -> dict:
    return client.get(
        "/ledger/vatType",
        params={"fields": "id,number,name,percentage", "count": 100},
    )
