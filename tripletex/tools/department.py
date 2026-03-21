from vertexai.generative_models import FunctionDeclaration

from client import TripletexClient

# --- Function declarations ---

LIST_DEPARTMENTS = FunctionDeclaration(
    name="list_departments",
    description="List departments in Tripletex. Returns id, name, departmentNumber.",
    parameters={"type": "object", "properties": {}},
)

CREATE_DEPARTMENT = FunctionDeclaration(
    name="create_department",
    description="Create a new department in Tripletex.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Department name"},
            "department_number": {"type": "string", "description": "Department number/code"},
        },
        "required": ["name"],
    },
)

# --- Implementations ---


def list_departments(client: TripletexClient, **_) -> dict:
    return client.get(
        "/department",
        params={"fields": "id,name,departmentNumber", "count": 100},
    )


def create_department(
    client: TripletexClient,
    name: str,
    department_number: str = None,
    **_,
) -> dict:
    if not name or not name.strip():
        raise ValueError("name is required")

    body = {"name": name.strip()}
    if department_number:
        body["departmentNumber"] = department_number.strip()

    return client.post("/department", json=body)
