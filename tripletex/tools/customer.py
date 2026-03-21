from vertexai.generative_models import FunctionDeclaration

from client import TripletexClient

# --- Function declarations ---

LIST_CUSTOMERS = FunctionDeclaration(
    name="list_customers",
    description="List customers in Tripletex. Returns id, name, email.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Filter by name (partial match)"},
        },
    },
)

CREATE_CUSTOMER = FunctionDeclaration(
    name="create_customer",
    description="Create a new customer in Tripletex.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Company or person name"},
            "email": {"type": "string", "description": "Customer email"},
            "phone_number": {"type": "string", "description": "Phone number"},
            "organization_number": {"type": "string", "description": "Norwegian org number"},
        },
        "required": ["name"],
    },
)

UPDATE_CUSTOMER = FunctionDeclaration(
    name="update_customer",
    description="Update an existing customer by their Tripletex ID.",
    parameters={
        "type": "object",
        "properties": {
            "customer_id": {"type": "integer", "description": "Tripletex customer ID"},
            "name": {"type": "string"},
            "email": {"type": "string"},
            "phone_number": {"type": "string"},
        },
        "required": ["customer_id"],
    },
)

# --- Implementations ---


def list_customers(client: TripletexClient, name: str = None, **_) -> dict:
    params = {"fields": "id,name,email,phoneNumber", "count": 100, "isCustomer": True}
    if name:
        params["name"] = name
    return client.get("/customer", params=params)


def create_customer(
    client: TripletexClient,
    name: str,
    email: str = None,
    phone_number: str = None,
    organization_number: str = None,
    **_,
) -> dict:
    if not name or not name.strip():
        raise ValueError("name is required")

    body = {"name": name.strip(), "isCustomer": True}
    if email:
        body["email"] = email.strip()
    if phone_number:
        body["phoneNumber"] = phone_number.strip()
    if organization_number:
        body["organizationNumber"] = organization_number.strip()

    return client.post("/customer", json=body)


def update_customer(
    client: TripletexClient,
    customer_id: int,
    name: str = None,
    email: str = None,
    phone_number: str = None,
    **_,
) -> dict:
    if not customer_id:
        raise ValueError("customer_id is required")

    existing = client.get(f"/customer/{customer_id}", params={"fields": "*"})
    body = existing["value"]

    if name is not None:
        body["name"] = name.strip()
    if email is not None:
        body["email"] = email.strip()
    if phone_number is not None:
        body["phoneNumber"] = phone_number.strip()

    return client.put(f"/customer/{customer_id}", json=body)
