from vertexai.generative_models import FunctionDeclaration

from client import TripletexClient

LIST_EMPLOYEES = FunctionDeclaration(
    name="list_employees",
    description="List employees in Tripletex. Returns id, firstName, lastName, email, phoneNumberMobile.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Filter by name (partial match)"},
        },
    },
)


def list_employees(client: TripletexClient, name: str = None, **_) -> dict:
    params = {"fields": "id,firstName,lastName,email,phoneNumberMobile,department", "count": 100}
    if name:
        params["name"] = name
    return client.get("/employee", params=params)
