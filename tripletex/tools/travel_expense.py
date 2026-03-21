from vertexai.generative_models import FunctionDeclaration

from client import TripletexClient

# --- Function declarations ---

LIST_TRAVEL_EXPENSES = FunctionDeclaration(
    name="list_travel_expenses",
    description="List travel expense reports in Tripletex. Returns id, title, employee, state.",
    parameters={
        "type": "object",
        "properties": {
            "employee_id": {"type": "integer", "description": "Filter by employee ID"},
        },
    },
)

CREATE_TRAVEL_EXPENSE = FunctionDeclaration(
    name="create_travel_expense",
    description=(
        "Create a new travel expense report in Tripletex. "
        "Dates go inside travelDetails (departureDate, returnDate)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "employee_id": {"type": "integer", "description": "Employee ID"},
            "title": {"type": "string", "description": "Title/name of the travel expense"},
            "departure_date": {"type": "string", "description": "Departure date (YYYY-MM-DD)"},
            "return_date": {"type": "string", "description": "Return date (YYYY-MM-DD)"},
            "purpose": {"type": "string", "description": "Purpose of the trip (goes in travelDetails.purpose)"},
        },
        "required": ["employee_id", "title", "departure_date", "return_date"],
    },
)

DELETE_TRAVEL_EXPENSE = FunctionDeclaration(
    name="delete_travel_expense",
    description="Delete a travel expense report by its Tripletex ID.",
    parameters={
        "type": "object",
        "properties": {
            "travel_expense_id": {"type": "integer", "description": "Tripletex travel expense ID"},
        },
        "required": ["travel_expense_id"],
    },
)

# --- Implementations ---


def list_travel_expenses(client: TripletexClient, employee_id: int = None, **_) -> dict:
    params = {
        "fields": "id,title,employee,state",
        "count": 100,
    }
    if employee_id:
        params["employeeId"] = employee_id
    return client.get("/travelExpense", params=params)


def create_travel_expense(
    client: TripletexClient,
    employee_id: int,
    title: str,
    departure_date: str,
    return_date: str,
    purpose: str = None,
    **_,
) -> dict:
    if not employee_id:
        raise ValueError("employee_id is required")
    if not title or not title.strip():
        raise ValueError("title is required")
    if not departure_date:
        raise ValueError("departure_date is required (YYYY-MM-DD)")
    if not return_date:
        raise ValueError("return_date is required (YYYY-MM-DD)")

    body = {
        "employee": {"id": employee_id},
        "title": title.strip(),
        "travelDetails": {
            "departureDate": departure_date,
            "returnDate": return_date,
        },
    }
    if purpose:
        body["travelDetails"]["purpose"] = purpose.strip()

    return client.post("/travelExpense", json=body)


def delete_travel_expense(
    client: TripletexClient,
    travel_expense_id: int,
    **_,
) -> None:
    if not travel_expense_id:
        raise ValueError("travel_expense_id is required")
    client.delete(f"/travelExpense/{travel_expense_id}")
    return {"deleted": True, "id": travel_expense_id}
