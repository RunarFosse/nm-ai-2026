from vertexai.generative_models import FunctionDeclaration

from client import TripletexClient

# --- Function declarations ---

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

CREATE_PROJECT = FunctionDeclaration(
    name="create_project",
    description="Create a new project in Tripletex.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Project name"},
            "customer_id": {"type": "integer", "description": "Customer ID to link the project to"},
            "project_manager_id": {"type": "integer", "description": "Employee ID of project manager"},
            "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
            "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"},
        },
        "required": ["name", "customer_id", "project_manager_id", "start_date"],
    },
)

# --- Implementations ---


def list_projects(client: TripletexClient, name: str = None, **_) -> dict:
    params = {"fields": "id,name,customer,projectManager,startDate,endDate", "count": 100}
    if name:
        params["name"] = name
    return client.get("/project", params=params)


def create_project(
    client: TripletexClient,
    name: str,
    customer_id: int,
    project_manager_id: int,
    start_date: str,
    end_date: str = None,
    **_,
) -> dict:
    if not name or not name.strip():
        raise ValueError("name is required")
    if not customer_id:
        raise ValueError("customer_id is required")
    if not project_manager_id:
        raise ValueError("project_manager_id is required")
    if not start_date:
        raise ValueError("start_date is required (YYYY-MM-DD)")

    body = {
        "name": name.strip(),
        "customer": {"id": customer_id},
        "projectManager": {"id": project_manager_id},
        "startDate": start_date,
    }
    if end_date:
        body["endDate"] = end_date

    return client.post("/project", json=body)
