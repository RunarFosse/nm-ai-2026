from vertexai.generative_models import FunctionDeclaration

from client import TripletexClient

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
