"""
Seed the Tripletex sandbox account with a known set of test entities.

Run from the tripletex/ directory:
    python scripts/seed_sandbox.py

Requires TRIPLETEX_SANDBOX_URL and TRIPLETEX_SANDBOX_TOKEN to be set
in tripletex/.env (or in the environment).
"""

import sys
from pathlib import Path

# Allow imports from parent (tripletex/) directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from client import TripletexClient, TripletexError
from config import settings


def seed(client: TripletexClient) -> None:
    print("Seeding Tripletex sandbox...\n")

    # --- Employees ---
    print("Creating employees...")
    try:
        emp = client.post(
            "/employee",
            json={"firstName": "Test", "lastName": "Manager", "email": "manager@test.no", "userType": "STANDARD"},
        )
        emp_id = emp["value"]["id"]
        print(f"  Employee created: id={emp_id} (Test Manager)")
    except TripletexError as e:
        print(f"  Employee creation failed: {e}")
        emp_id = None

    # --- Customers ---
    print("\nCreating customers...")
    customers = [
        {"name": "Acme AS", "email": "post@acme.no"},
        {"name": "Test Kunde AS", "email": "kontakt@testkunde.no"},
    ]
    customer_ids = []
    for c in customers:
        try:
            resp = client.post("/customer", json={**c, "isCustomer": True})
            cid = resp["value"]["id"]
            customer_ids.append(cid)
            print(f"  Customer created: id={cid} ({c['name']})")
        except TripletexError as e:
            print(f"  Customer creation failed ({c['name']}): {e}")

    # --- Products ---
    print("\nCreating products...")
    products = [
        {"name": "Konsulenttime", "number": "KONS-01", "costExcludingVatCurrency": 1500.0},
        {"name": "Programvarelisens", "number": "SW-01", "costExcludingVatCurrency": 500.0},
    ]
    for p in products:
        try:
            resp = client.post("/product", json=p)
            pid = resp["value"]["id"]
            print(f"  Product created: id={pid} ({p['name']})")
        except TripletexError as e:
            print(f"  Product creation failed ({p['name']}): {e}")

    print("\nSeed complete.")
    print(f"  Employee ID  : {emp_id}")
    print(f"  Customer IDs : {customer_ids}")


def main() -> None:
    if not settings.sandbox_url or not settings.sandbox_token:
        print(
            "ERROR: TRIPLETEX_SANDBOX_URL and TRIPLETEX_SANDBOX_TOKEN must be set.\n"
            "Copy tripletex/.env.example to tripletex/.env and fill in your sandbox credentials."
        )
        sys.exit(1)

    client = TripletexClient(
        base_url=settings.sandbox_url,
        session_token=settings.sandbox_token,
    )
    seed(client)


if __name__ == "__main__":
    main()
