from vertexai.generative_models import FunctionDeclaration

from client import TripletexClient

LIST_PROJECTS = FunctionDeclaration(
    name="list_projects",
    description="List projects in Tripletex. Returns id, name, customer, projectManager.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Filter by name (partial match)"},
        },
    },
)


def list_projects(client: TripletexClient, name: str = None, **_) -> dict:
    params = {"fields": "id,name,customer,projectManager,startDate,endDate", "count": 100}
    if name:
        params["name"] = name
    return client.get("/project", params=params)
