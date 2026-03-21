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
    description="Create a new employee in Tripletex.",
    parameters={
        "type": "object",
        "properties": {
            "first_name": {"type": "string", "description": "Employee first name"},
            "last_name": {"type": "string", "description": "Employee last name"},
            "email": {"type": "string", "description": "Employee email address"},
            "phone_number_mobile": {"type": "string", "description": "Mobile phone number"},
            "is_account_manager": {"type": "boolean", "description": "Grant account manager role"},
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
            "is_account_manager": {"type": "boolean"},
        },
        "required": ["employee_id"],
    },
)

# --- Implementations ---


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
    is_account_manager: bool = False,
    **_,
) -> dict:
    if not first_name or not first_name.strip():
        raise ValueError("first_name is required")
    if not last_name or not last_name.strip():
        raise ValueError("last_name is required")

    body = {"firstName": first_name.strip(), "lastName": last_name.strip(), "userType": "STANDARD"}
    if email:
        body["email"] = email.strip()
    if phone_number_mobile:
        body["phoneNumberMobile"] = phone_number_mobile.strip()

    result = client.post("/employee", json=body)
    employee_id = result["value"]["id"]

    if is_account_manager:
        client.put(f"/employee/{employee_id}/employeeRole", json={"administrator": True})

    return result


def update_employee(
    client: TripletexClient,
    employee_id: int,
    first_name: str = None,
    last_name: str = None,
    email: str = None,
    phone_number_mobile: str = None,
    is_account_manager: bool = None,
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

    result = client.put(f"/employee/{employee_id}", json=body)

    if is_account_manager is not None:
        client.put(
            f"/employee/{employee_id}/employeeRole",
            json={"administrator": is_account_manager},
        )

    return result
