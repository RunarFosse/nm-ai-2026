from vertexai.generative_models import FunctionDeclaration

from client import TripletexClient

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


def list_travel_expenses(client: TripletexClient, employee_id: int = None, **_) -> dict:
    params = {
        "fields": "id,title,employee,state",
        "count": 100,
    }
    if employee_id:
        params["employeeId"] = employee_id
    return client.get("/travelExpense", params=params)
