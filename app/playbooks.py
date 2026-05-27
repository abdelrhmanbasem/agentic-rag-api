# app/playbooks.py
# Universal assistant playbooks and style profiles.
#
# Future assistants inherit smart workflow behavior from here.

from typing import Dict, Any


DEFAULT_PROFILE = {
    "style": "sharp_operator",
    "dialect": "same_as_user",
    "verbosity": "short_but_useful",
    "cta_style": "single_next_step",
    "risk_policy": "do_not_overpromise",
}


PLAYBOOKS = {
    "car_sales": {
        "goal": "qualify buyer, recommend suitable inventory, handle objections, book viewing",
        "primary_cta": "viewing",
        "required_to_confirm": ["preferred_viewing_date", "preferred_viewing_time", "location", "phone_number"],
        "important_variables": [
            "car_brand",
            "car_condition",
            "budget_max",
            "transmission",
            "matched_car_model",
            "matched_car_price",
            "phone_number",
            "location",
        ],
        "objections": ["price", "financing", "mileage", "condition", "comparison"],
    },
    "service_booking": {
        "goal": "identify service, collect appointment details, confirm booking",
        "primary_cta": "appointment",
        "required_to_confirm": ["service_needed", "appointment_date", "appointment_time", "phone_number"],
        "important_variables": [
            "service_needed",
            "appointment_date",
            "appointment_time",
            "doctor_preference",
            "phone_number",
        ],
        "objections": ["price", "availability", "trust", "urgency"],
    },
    "real_estate": {
        "goal": "qualify property needs, recommend options, book viewing",
        "primary_cta": "property_viewing",
        "required_to_confirm": ["property_location", "budget_max", "viewing_date", "viewing_time", "phone_number"],
        "important_variables": [
            "property_type",
            "property_location",
            "bedrooms",
            "budget_max",
            "phone_number",
        ],
        "objections": ["price", "location", "space", "availability"],
    },
    "ecommerce": {
        "goal": "identify product, confirm variant, answer availability, progress order",
        "primary_cta": "order",
        "required_to_confirm": ["product_name", "phone_number", "shipping_address"],
        "important_variables": [
            "product_name",
            "product_category",
            "size",
            "color",
            "phone_number",
            "shipping_address",
        ],
        "objections": ["price", "delivery", "quality", "returns"],
    },
    "general": {
        "goal": "answer helpfully and collect the next useful detail",
        "primary_cta": "continue",
        "required_to_confirm": [],
        "important_variables": [],
        "objections": [],
    },
}


def get_playbook(workflow: str) -> Dict[str, Any]:
    return PLAYBOOKS.get(workflow or "general", PLAYBOOKS["general"])


def get_assistant_profile(assistant_id: str = "") -> Dict[str, Any]:
    assistant_id = (assistant_id or "").lower()
    profile = dict(DEFAULT_PROFILE)

    if any(x in assistant_id for x in ["car", "sales", "dealer"]):
        profile.update(
            {
                "style": "sharp_sales_operator",
                "verbosity": "short_persuasive",
                "cta_style": "soft_close",
            }
        )

    if any(x in assistant_id for x in ["clinic", "medical", "doctor", "dental"]):
        profile.update(
            {
                "style": "calm_booking_operator",
                "verbosity": "short_reassuring",
                "cta_style": "collect_one_detail",
            }
        )

    return profile
