from vertexai.generative_models import FunctionDeclaration

from client import TripletexClient

# --- Function declarations ---

LIST_TRAVEL_EXPENSES = FunctionDeclaration(
    name="list_travel_expenses",
    description="List travel expense reports in Tripletex. Returns id, description, employee, from/to dates.",
    parameters={
        "type": "object",
        "properties": {
            "employee_id": {"type": "integer", "description": "Filter by employee ID"},
        },
    },
)

CREATE_TRAVEL_EXPENSE = FunctionDeclaration(
    name="create_travel_expense",
    description="Create a new travel expense report in Tripletex.",
    parameters={
        "type": "object",
        "properties": {
            "employee_id": {"type": "integer", "description": "Employee ID"},
            "description": {"type": "string", "description": "Purpose of travel"},
            "date_from": {"type": "string", "description": "Travel start date (YYYY-MM-DD)"},
            "date_to": {"type": "string", "description": "Travel end date (YYYY-MM-DD)"},
        },
        "required": ["employee_id", "description", "date_from", "date_to"],
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
        "fields": "id,description,employee,dateFrom,dateTo,status",
        "count": 100,
    }
    if employee_id:
        params["employeeId"] = employee_id
    return client.get("/travelExpense", params=params)


def create_travel_expense(
    client: TripletexClient,
    employee_id: int,
    description: str,
    date_from: str,
    date_to: str,
    **_,
) -> dict:
    if not employee_id:
        raise ValueError("employee_id is required")
    if not description or not description.strip():
        raise ValueError("description is required")
    if not date_from:
        raise ValueError("date_from is required (YYYY-MM-DD)")
    if not date_to:
        raise ValueError("date_to is required (YYYY-MM-DD)")

    return client.post(
        "/travelExpense",
        json={
            "employee": {"id": employee_id},
            "description": description.strip(),
            "dateFrom": date_from,
            "dateTo": date_to,
        },
    )


def delete_travel_expense(
    client: TripletexClient,
    travel_expense_id: int,
    **_,
) -> None:
    if not travel_expense_id:
        raise ValueError("travel_expense_id is required")
    client.delete(f"/travelExpense/{travel_expense_id}")
    return {"deleted": True, "id": travel_expense_id}
