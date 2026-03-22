from vertexai.generative_models import FunctionDeclaration

from client import TripletexClient

# --- Function declarations (for Gemini) ---

LIST_EMPLOYEES = FunctionDeclaration(
    name="list_employees",
    description="List all employees in Tripletex. Returns id, firstName, lastName, email for each.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Filter by name (partial match)"},
        },
    },
)

CREATE_EMPLOYEE = FunctionDeclaration(
    name="create_employee",
    description=(
        "Create a new employee in Tripletex. "
        "The Tripletex API requires a department — if department_id is not provided, "
        "the tool will find or create one automatically."
    ),
    parameters={
        "type": "object",
        "properties": {
            "first_name": {"type": "string", "description": "Employee first name"},
            "last_name": {"type": "string", "description": "Employee last name"},
            "email": {"type": "string", "description": "Employee email address"},
            "phone_number_mobile": {"type": "string", "description": "Mobile phone number"},
            "department_id": {
                "type": "integer",
                "description": "Tripletex department ID. If omitted, the tool uses the first existing department or creates a default one.",
            },
        },
        "required": ["first_name", "last_name"],
    },
)

UPDATE_EMPLOYEE = FunctionDeclaration(
    name="update_employee",
    description="Update an existing employee by their Tripletex ID.",
    parameters={
        "type": "object",
        "properties": {
            "employee_id": {"type": "integer", "description": "Tripletex employee ID"},
            "first_name": {"type": "string"},
            "last_name": {"type": "string"},
            "email": {"type": "string"},
            "phone_number_mobile": {"type": "string"},
        },
        "required": ["employee_id"],
    },
)

# --- Implementations ---


def _ensure_department(client: TripletexClient) -> int:
    """Return an existing department ID, or create a default one."""
    resp = client.get("/department", params={"fields": "id,name", "count": 1})
    values = resp.get("values", [])
    if values:
        return values[0]["id"]
    # No departments — create a default
    created = client.post("/department", json={"name": "General"})
    return created["value"]["id"]


def list_employees(client: TripletexClient, name: str = None, **_) -> dict:
    params = {"fields": "id,firstName,lastName,email,phoneNumberMobile", "count": 100}
    if name:
        params["name"] = name
    return client.get("/employee", params=params)


def create_employee(
    client: TripletexClient,
    first_name: str,
    last_name: str,
    email: str = None,
    phone_number_mobile: str = None,
    department_id: int = None,
    **_,
) -> dict:
    if not first_name or not first_name.strip():
        raise ValueError("first_name is required")
    if not last_name or not last_name.strip():
        raise ValueError("last_name is required")

    dept_id = department_id or _ensure_department(client)

    body = {
        "firstName": first_name.strip(),
        "lastName": last_name.strip(),
        "userType": "STANDARD",
        "department": {"id": dept_id},
    }
    if email:
        body["email"] = email.strip()
    if phone_number_mobile:
        body["phoneNumberMobile"] = phone_number_mobile.strip()

    return client.post("/employee", json=body)


def update_employee(
    client: TripletexClient,
    employee_id: int,
    first_name: str = None,
    last_name: str = None,
    email: str = None,
    phone_number_mobile: str = None,
    **_,
) -> dict:
    if not employee_id:
        raise ValueError("employee_id is required")

    existing = client.get(f"/employee/{employee_id}", params={"fields": "*"})
    body = existing["value"]

    if first_name is not None:
        body["firstName"] = first_name.strip()
    if last_name is not None:
        body["lastName"] = last_name.strip()
    if email is not None:
        body["email"] = email.strip()
    if phone_number_mobile is not None:
        body["phoneNumberMobile"] = phone_number_mobile.strip()

    return client.put(f"/employee/{employee_id}", json=body)
