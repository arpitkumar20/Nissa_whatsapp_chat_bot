from flask import Blueprint, jsonify, request
from app.services.web_scraper import scracper_fun


scraper_bp = Blueprint("scrap", __name__)

@scraper_bp.route("/web-scraper", methods=["POST"])
def scraper():
    data = request.json or {}

    site_url = data.get("url")

    if not site_url:
        return jsonify({"error": "Missing required fields (site_url)"}), 400

    result = scracper_fun(site_url)
    return jsonify(result)