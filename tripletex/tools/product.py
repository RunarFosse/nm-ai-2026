from vertexai.generative_models import FunctionDeclaration

from client import TripletexClient

# --- Function declarations ---

LIST_PRODUCTS = FunctionDeclaration(
    name="list_products",
    description="List products in Tripletex. Returns id, name, number, priceExVat.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Filter by name (partial match)"},
        },
    },
)

CREATE_PRODUCT = FunctionDeclaration(
    name="create_product",
    description="Create a new product or service in Tripletex.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Product name"},
            "number": {"type": "string", "description": "Product number / SKU"},
            "price_ex_vat": {"type": "number", "description": "Unit price excl. VAT (alias: cost_excluding_vat_currency)"},
            "cost_excluding_vat_currency": {"type": "number", "description": "Unit cost excl. VAT (same as price_ex_vat)"},
        },
        "required": ["name"],
    },
)

# --- Implementations ---


def list_products(client: TripletexClient, name: str = None, **_) -> dict:
    params = {"fields": "id,name,number,priceExVat", "count": 100}
    if name:
        params["name"] = name
    return client.get("/product", params=params)


def create_product(
    client: TripletexClient,
    name: str,
    number: str = None,
    price_ex_vat: float = None,
    cost_excluding_vat_currency: float = None,
    **_,
) -> dict:
    if not name or not name.strip():
        raise ValueError("name is required")

    body = {"name": name.strip()}
    if number:
        body["number"] = number.strip()
    price = price_ex_vat if price_ex_vat is not None else cost_excluding_vat_currency
    if price is not None:
        body["costExcludingVatCurrency"] = price

    return client.post("/product", json=body)
