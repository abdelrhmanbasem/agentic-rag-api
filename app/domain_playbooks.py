# app/domain_playbooks.py
# Domain playbooks for all future assistants.
#
# Purpose:
# - Keep assistant behavior consistent and smart across verticals.
# - Give the brain a domain-specific strategy without using GPT.

from typing import Dict, Any, List


DOMAIN_PLAYBOOKS: Dict[str, Dict[str, Any]] = {
    "car_sales": {
        "goal": "qualify buyer, recommend suitable inventory, handle objections, and book a viewing",
        "primary_cta": "viewing",
        "tone": "sharp_sales_operator",
        "preferred_style": "short_persuasive",
        "required_to_confirm": [
            "preferred_viewing_date",
            "preferred_viewing_time",
            "location",
            "phone_number",
        ],
        "important_variables": [
            "car_brand",
            "car_condition",
            "budget_max",
            "transmission",
            "matched_car_model",
            "matched_car_year",
            "matched_car_km",
            "matched_car_price",
            "selected_item",
            "location",
            "phone_number",
        ],
        "common_objections": [
            "price",
            "mileage",
            "condition",
            "financing",
            "comparison",
            "trust",
        ],
        "next_moves": {
            "new": "match_inventory",
            "matched_inventory": "answer_then_soft_close",
            "budget_confirmed": "book_viewing",
            "viewing_requested": "collect_viewing_details",
            "confirmed": "confirm_and_handoff",
        },
    },
    "service_booking": {
        "goal": "identify service, collect appointment details, and confirm booking",
        "primary_cta": "appointment",
        "tone": "calm_booking_operator",
        "preferred_style": "short_reassuring",
        "required_to_confirm": [
            "service_needed",
            "appointment_date",
            "appointment_time",
            "phone_number",
        ],
        "important_variables": [
            "service_needed",
            "appointment_date",
            "appointment_time",
            "doctor_preference",
            "patient_name",
            "phone_number",
        ],
        "common_objections": [
            "price",
            "availability",
            "urgency",
            "trust",
        ],
        "next_moves": {
            "new": "identify_service",
            "service_identified": "collect_date_time",
            "booking_requested": "collect_booking_details",
            "confirmed": "confirm_and_handoff",
        },
    },
    "real_estate": {
        "goal": "qualify needs, recommend properties, handle objections, book viewing",
        "primary_cta": "property_viewing",
        "tone": "sharp_property_operator",
        "preferred_style": "short_persuasive",
        "required_to_confirm": [
            "property_location",
            "budget_max",
            "viewing_date",
            "viewing_time",
            "phone_number",
        ],
        "important_variables": [
            "property_type",
            "property_location",
            "bedrooms",
            "budget_max",
            "phone_number",
        ],
        "common_objections": [
            "price",
            "location",
            "space",
            "availability",
        ],
        "next_moves": {
            "new": "qualify_location_budget",
            "matched_inventory": "answer_then_soft_close",
            "viewing_requested": "collect_viewing_details",
            "confirmed": "confirm_and_handoff",
        },
    },
    "ecommerce": {
        "goal": "identify product, answer availability, confirm variant, progress order",
        "primary_cta": "order",
        "tone": "helpful_sales_operator",
        "preferred_style": "short",
        "required_to_confirm": [
            "product_name",
            "phone_number",
            "shipping_address",
        ],
        "important_variables": [
            "product_name",
            "product_category",
            "size",
            "color",
            "phone_number",
            "shipping_address",
        ],
        "common_objections": [
            "price",
            "delivery",
            "quality",
            "returns",
        ],
        "next_moves": {
            "new": "identify_product",
            "matched_inventory": "answer_then_soft_close",
            "order_requested": "collect_order_details",
            "confirmed": "confirm_order",
        },
    },
    "general": {
        "goal": "answer clearly and collect one useful next detail",
        "primary_cta": "continue",
        "tone": "helpful_operator",
        "preferred_style": "short",
        "required_to_confirm": [],
        "important_variables": [],
        "common_objections": [],
        "next_moves": {
            "general": "continue",
        },
    },
}


ASSISTANT_STYLE_PROFILES: Dict[str, Dict[str, Any]] = {
    "sharp_sales_operator": {
        "language": "same_as_user",
        "verbosity": "short_but_persuasive",
        "cta_style": "single_next_step",
        "do": [
            "answer first",
            "connect answer to user goal",
            "move conversation forward",
            "sound confident but not pushy",
        ],
        "dont": [
            "overpromise",
            "ask multiple questions",
            "sound robotic",
            "repeat the same CTA",
        ],
    },
    "calm_booking_operator": {
        "language": "same_as_user",
        "verbosity": "short_reassuring",
        "cta_style": "collect_one_detail",
        "do": [
            "confirm what was understood",
            "ask for one missing detail",
            "be calm and clear",
        ],
        "dont": [
            "overwhelm user",
            "ask multiple fields at once",
            "invent availability",
        ],
    },
    "helpful_operator": {
        "language": "same_as_user",
        "verbosity": "short_clear",
        "cta_style": "continue",
        "do": [
            "be direct",
            "be useful",
            "ask one natural follow-up",
        ],
        "dont": [
            "over-explain",
            "invent facts",
        ],
    },
}


def get_domain_playbook(workflow: str) -> Dict[str, Any]:
    return DOMAIN_PLAYBOOKS.get(workflow or "general", DOMAIN_PLAYBOOKS["general"])


def get_style_profile(style_name: str) -> Dict[str, Any]:
    return ASSISTANT_STYLE_PROFILES.get(style_name or "helpful_operator", ASSISTANT_STYLE_PROFILES["helpful_operator"])


def infer_style_for_workflow(workflow: str) -> Dict[str, Any]:
    playbook = get_domain_playbook(workflow)
    return get_style_profile(playbook.get("tone", "helpful_operator"))


def required_fields_for_workflow(workflow: str) -> List[str]:
    return list(get_domain_playbook(workflow).get("required_to_confirm", []))


def missing_required_fields(workflow: str, variables: Dict[str, Any]) -> List[str]:
    variables = variables or {}
    missing = []

    for key in required_fields_for_workflow(workflow):
        value = variables.get(key)
        if value is None or value == "":
            missing.append(key)

    return missing
