"""
Collects all Tripletex tool definitions and implementations.

- ALL_DECLARATIONS: list of FunctionDeclaration objects passed to Gemini
- TOOL_MAP: maps function name → callable(client, **kwargs)
"""

from vertexai.generative_models import Tool

from tools.customer import (
    CREATE_CUSTOMER,
    LIST_CUSTOMERS,
    UPDATE_CUSTOMER,
    create_customer,
    list_customers,
    update_customer,
)
from tools.department import (
    CREATE_DEPARTMENT,
    LIST_DEPARTMENTS,
    create_department,
    list_departments,
)
from tools.employee import (
    CREATE_EMPLOYEE,
    LIST_EMPLOYEES,
    UPDATE_EMPLOYEE,
    create_employee,
    list_employees,
    update_employee,
)
from tools.invoice import (
    CREATE_INVOICE,
    LIST_INVOICES,
    LIST_PAYMENT_TYPES,
    REGISTER_PAYMENT,
    create_invoice,
    list_invoices,
    list_payment_types,
    register_payment,
)
from tools.order import CREATE_ORDER, create_order
from tools.product import CREATE_PRODUCT, LIST_PRODUCTS, create_product, list_products
from tools.project import CREATE_PROJECT, LIST_PROJECTS, create_project, list_projects
from tools.travel_expense import (
    CREATE_TRAVEL_EXPENSE,
    DELETE_TRAVEL_EXPENSE,
    LIST_TRAVEL_EXPENSES,
    create_travel_expense,
    delete_travel_expense,
    list_travel_expenses,
)
from tools.call_api import CALL_API, call_api
from tools.schema_tools import (
    GET_ENDPOINT_SCHEMA,
    LIST_ENDPOINTS,
    get_endpoint_schema,
    list_endpoints,
)
from tools.knowledge_tools import (
    GET_ENDPOINT_NOTES,
    UPDATE_ENDPOINT_NOTES,
    get_endpoint_notes,
    update_endpoint_notes,
)
from tools.voucher import (
    CREATE_VOUCHER,
    LIST_ACCOUNTS,
    LIST_VOUCHERS,
    create_voucher,
    list_accounts,
    list_vouchers,
)

ALL_DECLARATIONS = [
    # Employees
    LIST_EMPLOYEES,
    CREATE_EMPLOYEE,
    UPDATE_EMPLOYEE,
    # Customers
    LIST_CUSTOMERS,
    CREATE_CUSTOMER,
    UPDATE_CUSTOMER,
    # Products
    LIST_PRODUCTS,
    CREATE_PRODUCT,
    # Orders
    CREATE_ORDER,
    # Invoices
    LIST_INVOICES,
    CREATE_INVOICE,
    REGISTER_PAYMENT,
    LIST_PAYMENT_TYPES,
    # Travel expenses
    LIST_TRAVEL_EXPENSES,
    CREATE_TRAVEL_EXPENSE,
    DELETE_TRAVEL_EXPENSE,
    # Projects
    LIST_PROJECTS,
    CREATE_PROJECT,
    # Departments
    LIST_DEPARTMENTS,
    CREATE_DEPARTMENT,
    # Vouchers / ledger
    LIST_VOUCHERS,
    CREATE_VOUCHER,
    LIST_ACCOUNTS,
    # Generic escape hatch
    CALL_API,
    # API discovery
    LIST_ENDPOINTS,
    GET_ENDPOINT_SCHEMA,
]

GEMINI_TOOL = Tool(function_declarations=ALL_DECLARATIONS)

# Read-only tool set for the research phase — no write operations
RESEARCH_DECLARATIONS = [
    LIST_EMPLOYEES,
    LIST_CUSTOMERS,
    LIST_PRODUCTS,
    LIST_INVOICES,
    LIST_PAYMENT_TYPES,
    LIST_TRAVEL_EXPENSES,
    LIST_PROJECTS,
    LIST_DEPARTMENTS,
    LIST_VOUCHERS,
    LIST_ACCOUNTS,
    LIST_ENDPOINTS,
    GET_ENDPOINT_SCHEMA,
]
RESEARCH_TOOL = Tool(function_declarations=RESEARCH_DECLARATIONS)

# Plan tool: list tools + endpoint discovery (no schema — loaded by code) + notes
PLAN_DECLARATIONS = [
    LIST_EMPLOYEES,
    LIST_CUSTOMERS,
    LIST_PRODUCTS,
    LIST_INVOICES,
    LIST_PAYMENT_TYPES,
    LIST_TRAVEL_EXPENSES,
    LIST_PROJECTS,
    LIST_DEPARTMENTS,
    LIST_VOUCHERS,
    LIST_ACCOUNTS,
    LIST_ENDPOINTS,
    GET_ENDPOINT_NOTES,
]
PLAN_TOOL = Tool(function_declarations=PLAN_DECLARATIONS)

TOOL_MAP: dict = {
    "list_employees": list_employees,
    "create_employee": create_employee,
    "update_employee": update_employee,
    "list_customers": list_customers,
    "create_customer": create_customer,
    "update_customer": update_customer,
    "list_products": list_products,
    "create_product": create_product,
    "create_order": create_order,
    "list_invoices": list_invoices,
    "create_invoice": create_invoice,
    "register_payment": register_payment,
    "list_payment_types": list_payment_types,
    "list_travel_expenses": list_travel_expenses,
    "create_travel_expense": create_travel_expense,
    "delete_travel_expense": delete_travel_expense,
    "list_projects": list_projects,
    "create_project": create_project,
    "list_departments": list_departments,
    "create_department": create_department,
    "list_vouchers": list_vouchers,
    "create_voucher": create_voucher,
    "list_accounts": list_accounts,
    "call_api": call_api,
    "list_endpoints": list_endpoints,
    "get_endpoint_schema": get_endpoint_schema,
    "get_endpoint_notes": get_endpoint_notes,
    "update_endpoint_notes": update_endpoint_notes,
}
