# app/domain_playbooks.py
# Domain playbooks to inject business logic directly into the LangGraph LLM brain.

from typing import Dict, Any, List

DOMAIN_PLAYBOOKS: Dict[str, Dict[str, Any]] = {
    "car_sales": {
        "goal": "Qualify the buyer, recommend suitable inventory, gracefully handle price objections, and book a physical viewing.",
        "primary_cta": "Schedule a physical viewing",
        "required_to_confirm": [
            "preferred_viewing_date",
            "preferred_viewing_time",
            "location",
            "phone_number",
        ],
        "important_variables": [
            "car_brand", "car_condition", "budget_max", "transmission",
            "matched_car_model", "matched_car_price"
        ],
        "common_objections": ["price", "mileage", "condition", "financing"]
    },
    "service_booking": {
        "goal": "Identify the exact service needed, collect appointment details calmly, and confirm the booking.",
        "primary_cta": "Confirm appointment time and date",
        "required_to_confirm": [
            "service_needed",
            "appointment_date",
            "appointment_time",
            "phone_number",
        ],
        "important_variables": ["service_needed", "doctor_preference", "patient_name", "phone_number"],
        "common_objections": ["price", "availability", "urgency", "trust"]
    },
    "real_estate": {
        "goal": "Qualify housing needs, recommend properties, handle location/price objections, and book a viewing.",
        "primary_cta": "Book a property viewing",
        "required_to_confirm": [
            "property_location",
            "budget_max",
            "viewing_date",
            "viewing_time",
            "phone_number",
        ],
        "important_variables": ["property_type", "bedrooms", "budget_max"],
        "common_objections": ["price", "location", "space", "availability"]
    },
    "ecommerce": {
        "goal": "Identify the product, confirm availability/variants (size/color), and finalize the order details.",
        "primary_cta": "Confirm order and shipping",
        "required_to_confirm": [
            "product_name",
            "phone_number",
            "shipping_address",
        ],
        "important_variables": ["product_name", "size", "color"],
        "common_objections": ["price", "delivery time", "quality", "returns"]
    },
    "general": {
        "goal": "Answer clearly, be helpful, and organically collect any missing details to assist the user.",
        "primary_cta": "Continue the conversation naturally",
        "required_to_confirm": [],
        "important_variables": [],
        "common_objections": []
    },
}


ASSISTANT_STYLE_PROFILES: Dict[str, Dict[str, Any]] = {
    "sharp_sales_operator": {
        "do": [
            "Answer the user's immediate question first.",
            "Connect the answer to their ultimate goal naturally.",
            "Ask exactly ONE conversational follow-up question to move towards the CTA.",
            "Sound confident, native, and helpful, not pushy or robotic."
        ],
        "dont": [
            "Do not overpromise facts you don't have.",
            "Do not ask multiple questions at the same time.",
            "Do not sound like a scripted bot."
        ],
    },
    "calm_booking_operator": {
        "do": [
            "Confirm what you just understood calmly.",
            "Ask for ONE missing detail required for booking at a time.",
            "Be empathetic if they are asking about medical or sensitive services."
        ],
        "dont": [
            "Do not overwhelm the user with a massive form.",
            "Do not invent availability or doctor names."
        ],
    },
    "helpful_operator": {
        "do": [
            "Be direct, highly useful, and conversational.",
            "Ask one natural follow-up if context is missing."
        ],
        "dont": [
            "Do not over-explain.",
            "Do not invent facts outside of the provided knowledge."
        ],
    },
}


def get_domain_playbook(workflow: str) -> Dict[str, Any]:
    return DOMAIN_PLAYBOOKS.get(workflow or "general", DOMAIN_PLAYBOOKS["general"])

def get_style_profile(style_name: str) -> Dict[str, Any]:
    # Default to helpful_operator if not found
    return ASSISTANT_STYLE_PROFILES.get(style_name or "helpful_operator", ASSISTANT_STYLE_PROFILES["helpful_operator"])

def build_playbook_prompt(workflow: str, tone_profile: str = "helpful_operator") -> str:
    """
    Transforms the playbook dictionaries into a strict set of LLM instructions.
    This replaces thousands of lines of procedural code.
    """
    playbook = get_domain_playbook(workflow)
    style = get_style_profile(tone_profile)
    
    do_rules = "\n".join([f"- {rule}" for rule in style["do"]])
    dont_rules = "\n".join([f"- {rule}" for rule in style["dont"]])
    
    missing_fields_instruction = ""
    if playbook["required_to_confirm"]:
        reqs = ", ".join(playbook["required_to_confirm"])
        missing_fields_instruction = f"CRITICAL: To achieve your goal, you eventually need to collect these fields: [{reqs}]. If any are missing in the 'Known Details', naturally ask the user for ONE of them in your response."

    prompt = f"""
=== DOMAIN PLAYBOOK & STRATEGY ===
Your Goal: {playbook['goal']}
Primary CTA: {playbook['primary_cta']}

{missing_fields_instruction}

CONVERSATION STYLE RULES ({tone_profile}):
DO:
{do_rules}

DO NOT:
{dont_rules}
==================================
"""
    return prompt
