from vertexai.generative_models import FunctionDeclaration

from client import TripletexClient

LIST_CUSTOMERS = FunctionDeclaration(
    name="list_customers",
    description="List customers in Tripletex. Returns id, name, email.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Filter by name (partial match)"},
        },
    },
)


def list_customers(client: TripletexClient, name: str = None, **_) -> dict:
    params = {"fields": "id,name,email,phoneNumber", "count": 100, "isCustomer": True}
    if name:
        params["name"] = name
    return client.get("/customer", params=params)
