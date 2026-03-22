from vertexai.generative_models import FunctionDeclaration

from client import TripletexClient

# --- Function declarations ---

LIST_VOUCHERS = FunctionDeclaration(
    name="list_vouchers",
    description="List vouchers (bilag) in Tripletex. Returns id, date, description, voucherType.",
    parameters={
        "type": "object",
        "properties": {
            "date_from": {"type": "string", "description": "From date (YYYY-MM-DD)"},
            "date_to": {"type": "string", "description": "To date (YYYY-MM-DD)"},
        },
    },
)

CREATE_VOUCHER = FunctionDeclaration(
    name="create_voucher",
    description="Create a new accounting voucher (bilag) with postings in Tripletex.",
    parameters={
        "type": "object",
        "properties": {
            "date": {"type": "string", "description": "Voucher date (YYYY-MM-DD)"},
            "description": {"type": "string", "description": "Voucher description"},
            "postings": {
                "type": "array",
                "description": "List of debit/credit postings — must balance to zero",
                "items": {
                    "type": "object",
                    "properties": {
                        "account_id": {"type": "integer", "description": "Ledger account ID"},
                        "amount": {"type": "number", "description": "Amount (positive = debit, negative = credit)"},
                        "description": {"type": "string"},
                    },
                },
            },
        },
        "required": ["date", "postings"],
    },
)

LIST_ACCOUNTS = FunctionDeclaration(
    name="list_accounts",
    description="List ledger accounts (chart of accounts) in Tripletex.",
    parameters={
        "type": "object",
        "properties": {
            "number": {"type": "integer", "description": "Filter by exact account number (e.g. 1920)"},
            "is_bank_account": {"type": "boolean", "description": "Filter to bank accounts only"},
        },
    },
)

# --- Implementations ---


def list_vouchers(
    client: TripletexClient,
    date_from: str = None,
    date_to: str = None,
    **_,
) -> dict:
    params = {"fields": "id,date,description,voucherType", "count": 100}
    if date_from:
        params["dateFrom"] = date_from
    if date_to:
        params["dateTo"] = date_to
    return client.get("/ledger/voucher", params=params)


def create_voucher(
    client: TripletexClient,
    date: str,
    postings: list,
    description: str = None,
    **_,
) -> dict:
    if not date:
        raise ValueError("date is required (YYYY-MM-DD)")
    if not postings:
        raise ValueError("postings are required")
    total = sum(p.get("amount", 0) for p in postings)
    if abs(total) > 0.01:
        raise ValueError(f"Postings must balance to zero (current sum: {total})")

    body = {
        "date": date,
        "postings": [
            {
                "account": {"id": p["account_id"]},
                "amount": p["amount"],
                **({"description": p["description"]} if "description" in p else {}),
            }
            for p in postings
        ],
    }
    if description:
        body["description"] = description

    return client.post("/ledger/voucher", json=body)


def list_accounts(
    client: TripletexClient,
    number: int = None,
    is_bank_account: bool = None,
    **_,
) -> dict:
    params = {"fields": "id,number,name,bankAccountNumber", "count": 200}
    if number is not None:
        params["number"] = number
    if is_bank_account is not None:
        params["isBankAccount"] = is_bank_account
    return client.get("/ledger/account", params=params)
