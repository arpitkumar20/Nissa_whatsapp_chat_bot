from pydantic import BaseModel, Field

class BookingRequest(BaseModel):
    service_type: str = Field(description="hospital|hotel")
    name: str
    date: str
    customer_phone: str
    notes: str | None = None

class BookingConfirmation(BaseModel):
    booking_id: str
    status: str
    provider: str
    details: dict
