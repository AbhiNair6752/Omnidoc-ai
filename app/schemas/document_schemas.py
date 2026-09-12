from pydantic import BaseModel
from typing import Optional

class InvoiceData(BaseModel):
    invoice_number: Optional[str] = None

    customer_name: Optional[str] = None

    account_number: Optional[str] = None

    invoice_date: Optional[str] = None

    subtotal: Optional[str] = None

    tax: Optional[str] = None

    total: Optional[str] = None

class DocumentData(BaseModel):

    document_type: str

    extracted_information: dict