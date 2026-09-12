from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(
    title="OmniDoc Enterprise APIs",
    description="Mock enterprise APIs simulating webMethods integrations",
    version="1.0.0"
)


CUSTOMERS = {

    "C001": {
        "customer_id": "C001",
        "name": "John Peterson",
        "status": "ACTIVE"
    },

    "C002": {
        "customer_id": "C002",
        "name": "Sarah Williams",
        "status": "ACTIVE"
    }
}


ACCOUNTS = {

    "C001": {
        "customer_id": "C001",
        "account_number": "1234567890",
        "balance": 75000
    },

    "C002": {
        "customer_id": "C002",
        "account_number": "9876543210",
        "balance": 25000
    }
}


@app.get("/")
def health_check():

    return {
        "status": "success",
        "message": "Enterprise API is running"
    }


@app.get("/customers/{customer_id}")
def get_customer(
    customer_id: str
):

    customer = CUSTOMERS.get(
        customer_id
    )

    if not customer:

        return {
            "status": "NOT_FOUND",
            "message": "Customer not found"
        }

    return customer


class PANRequest(BaseModel):

    pan_number: str


@app.post("/pan/verify")
def verify_pan(
    request: PANRequest
):

    if request.pan_number == "ABCDE1234F":

        return {
            "status": "VERIFIED",
            "pan_number": request.pan_number
        }

    return {
        "status": "INVALID",
        "pan_number": request.pan_number
    }


@app.get("/accounts/{customer_id}")
def get_account(
    customer_id: str
):

    account = ACCOUNTS.get(
        customer_id
    )

    if not account:

        return {
            "status": "NOT_FOUND"
        }

    return account


@app.post("/loan/eligibility")
def check_loan_eligibility(
    customer_id: str
):

    account = ACCOUNTS.get(
        customer_id
    )

    if not account:

        return {
            "eligible": False,
            "reason": "Customer not found"
        }

    if account["balance"] >= 50000:

        return {
            "eligible": True,
            "reason": (
                "Customer meets balance criteria"
            )
        }

    return {
        "eligible": False,
        "reason": (
            "Insufficient account balance"
        )
    }