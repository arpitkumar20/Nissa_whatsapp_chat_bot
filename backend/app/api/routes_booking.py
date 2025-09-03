# from flask import Blueprint, request, jsonify
# from ..models.booking import BookingRequest, BookingConfirmation

# bp = Blueprint("booking", __name__)

# @bp.post("/create")
# def create_booking():
#     data = request.get_json(force=True)
#     br = BookingRequest.model_validate(data)
#     # mock integration; replace with HMS/hotel API call
#     confirmation = BookingConfirmation(
#         booking_id="BK-" + br.customer_phone[-4:],
#         status="CONFIRMED",
#         provider="mock",
#         details=br.model_dump()
#     )
#     return jsonify(confirmation.model_dump())
