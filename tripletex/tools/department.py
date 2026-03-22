from vertexai.generative_models import FunctionDeclaration

from client import TripletexClient

LIST_DEPARTMENTS = FunctionDeclaration(
    name="list_departments",
    description="List departments in Tripletex. Returns id, name, departmentNumber.",
    parameters={"type": "object", "properties": {}},
)


def list_departments(client: TripletexClient, **_) -> dict:
    return client.get(
        "/department",
        params={"fields": "id,name,departmentNumber", "count": 100},
    )
