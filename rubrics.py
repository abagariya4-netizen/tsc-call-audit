"""
The 8 scoring rubrics for TSC call audits.
One rubric per lead source. Each human rubric is worth 100 points and uses identical parameters.
"""

RUBRIC_VERSION = "v8"

# Standard unified parameters used for all 7 human agent rubrics (fatal parameter completely removed)
HUMAN_PARAMETERS = [
    {
        "key": "greeting_introduction",
        "name": "Greeting & Introduction",
        "max_points": 5,
        "check": "Did agent greet the customer appropriately, introduce themselves and the company, and start within 3 seconds?",
        "failure_modes": [
            "Delay opening more than 3 seconds",
            "Did not greet the customer appropriately along with introduction of themselves and the company"
        ]
    },
    {
        "key": "understanding_customer_needs",
        "name": "Understanding Customer Needs",
        "max_points": 20,
        "check": "Did agent actively listen, ask relevant probing questions, acknowledge the customer query, and fully understand the customer specific needs?",
        "failure_modes": [
            "Agent did not actively listen to customer issue",
            "Relevant questions not asked to understand customer needs",
            "Did not probe when the customer was not interested",
            "Did not acknowledge customer query"
        ]
    },
    {
        "key": "sales_pitch",
        "name": "Sales Pitch",
        "max_points": 20,
        "check": "Did agent explain features and benefits of the product with confidence and product knowledge? Did agent pitch the right product for the customer needs?",
        "failure_modes": [
            "Did not explain features and benefits of the product",
            "Did not display confidence or knowledge about the product"
        ]
    },
    {
        "key": "advisor_behaviour",
        "name": "Advisor Behaviour",
        "max_points": 5,
        "check": "Did agent maintain professional polite non-rude non-abrupt behaviour throughout the entire call?",
        "failure_modes": [
            "Rude and abrupt behaviour observed"
        ]
    },
    {
        "key": "complaint",
        "name": "Complaint",
        "max_points": 10,
        "check": "If customer raised a complaint was it acknowledged and addressed properly? If no complaint was raised score NA.",
        "failure_modes": [
            "Complaint not acknowledged or not addressed"
        ]
    },
    {
        "key": "ownership_resolution",
        "name": "Ownership & Resolution",
        "max_points": 10,
        "check": "Did agent take full ownership, provide correct and complete information, and manage the call properly end to end?",
        "failure_modes": [
            "Did not note the conversation or missing information",
            "Information provided was incorrect or incomplete",
            "Did not tag the call as per the query",
            "Provided wrong pricing, wrong offer details, or wrong product specification"
        ]
    },
    {
        "key": "hold_mute",
        "name": "Hold & Mute",
        "max_points": 5,
        "check": "Did agent avoid dead air more than 10 seconds and long holds or mutes more than 120 seconds?",
        "failure_modes": [
            "Dead air more than 10 seconds",
            "Long hold or mute more than 120 seconds observed"
        ]
    },
    {
        "key": "communication",
        "name": "Communication",
        "max_points": 10,
        "check": "Was agent tone professional, grammar correct, pace appropriate, and did agent avoid speaking over the customer?",
        "failure_modes": [
            "Tone of voice and grammatical errors",
            "Spoke too fast or agent was unclear in communication",
            "Spoke over the caller",
            "Incorrect sentence formation"
        ]
    },
    {
        "key": "closing",
        "name": "Closing",
        "max_points": 15,
        "check": (
            "How the agent wraps up and ends the call. Read the FINAL portion of the transcript. Judge it as a fair human auditor would — consider what the customer did, not just what the agent said.\n"
            "A strong close usually contains some or all of:\n"
            "- Confirming clear next steps (callback time, delivery, store visit, order confirmation, follow-up)\n"
            "- A brief recap of what was discussed or agreed\n"
            "- Checking if the customer has any further questions\n"
            "- A polite, branded sign-off (thanking the customer, 'The Sleep Company')\n"
            "- Securing the commitment relevant to the call's purpose\n\n"
            "How to decide the score — THINK LIKE A HUMAN AUDITOR, IN THIS ORDER:\n"
            "STEP 1 — Did the agent even have the OPPORTUNITY to close?\n"
            "Look at how the call ended. If the customer hung up abruptly, the line dropped, the customer ended the conversation mid-flow, or the transcript simply cuts off before any natural wrap-up point — the agent was NOT given the chance to close. In that case score 'NA'.\n"
            "CRITICAL: NA is ONLY for 'no opportunity.' NEVER use NA just because the close was bad or missing. If the agent reached a natural end-of-call point and simply did a poor job, that is a SCORE, not NA.\n\n"
            "STEP 2 — If the agent HAD the opportunity, score on the 0–15 scale:\n"
            "13–15: Clean, complete close — next steps confirmed AND a clear recap or commitment AND a polite/branded sign-off.\n"
            "8–12: Decent close but missing elements — e.g. signed off politely but didn't confirm next steps, or confirmed next steps but no recap, or rushed wrap-up. Give credit for what they did do.\n"
            "4–7: Weak close — only a bare goodbye, no confirmation of anything, no recap, no commitment, but the agent did acknowledge the call was ending.\n"
            "1–3: Barely any attempt — agent let the call fizzle out with almost nothing.\n"
            "0: Agent clearly had the chance to close and did absolutely nothing — no sign-off, no next steps, no acknowledgement, just stopped."
        ),
        "failure_modes": [
            "Did not confirm next steps",
            "Did not ask for further help and provide assurance",
            "Missed polite/branded sign-off",
            "Barely any attempt or zero attempt when given the chance"
        ]
    }
]

RUBRICS = {
    "find_a_store": {
        "name": "Find a Store",
        "description": "High-intent customer search for nearest store. Confirm store and push for visit, do not over-pitch product.",
        "parameters": HUMAN_PARAMETERS
    },

    "arrange_callback": {
        "name": "Arrange Callback",
        "description": "Exploratory request from website. Qualify customer sleep needs first, recommend product, push next step.",
        "parameters": HUMAN_PARAMETERS
    },

    "inbound": {
        "name": "Inbound Call",
        "description": "Highest intent. Customer initiated call. Listen actively, answer questions directly, recommend next step.",
        "parameters": HUMAN_PARAMETERS
    },

    "shopflo_abandoned_cart": {
        "name": "Shopflo Abandoned Cart",
        "description": "High intent recovery. Customer abandoned cart. Probe to find the blocker (price, delivery, trust) and resolve it.",
        "parameters": HUMAN_PARAMETERS
    },

    "next_day_delivery": {
        "name": "Next Day Delivery",
        "description": "Urgency focus. Confirm pincode eligibility, explain the free pillow guarantee, close quickly.",
        "parameters": HUMAN_PARAMETERS
    },

    "no_cost_emi": {
        "name": "No-Cost EMI",
        "description": "Price sensitive. Explain zero-interest EMI math clearly, do monthly breakdowns, mention bank offers.",
        "parameters": HUMAN_PARAMETERS
    },

    "sales_team": {
        "name": "Sales Team Audit Baseline",
        "description": "Baseline rubric used for standard sales team evaluation. Focuses on deep customer qualification, SmartGRID technology pitch, and professional closing.",
        "parameters": HUMAN_PARAMETERS
    },

    "ai_voice_bot": {
        "name": "AI Voice Bot",
        "description": "Automated call analysis. Evaluate conversation quality, understanding, latency, and loops.",
        "parameters": [
            {
                "key": "greeting_introduction",
                "name": "Opening",
                "max_points": 5,
                "check": "Did the bot greet professionally, identify The Sleep Company (TSC), and transition smoothly into the conversation?",
                "failure_modes": ["Unprofessional or delayed opening"]
            },
            {
                "key": "advisor_behaviour",
                "name": "Bot Conduct / Robotic Delivery",
                "max_points": 15,
                "check": "Voice quality — natural pacing, clear pronunciation, no robotic monotone, appropriate inflection and warmth.",
                "failure_modes": ["Robotic, scripted, or monotone delivery throughout"]
            },
            {
                "key": "customer_understanding",
                "name": "Customer Understanding",
                "max_points": 20,
                "check": "Did the bot correctly understand what the customer said, including Hindi/English code-switching, accents, partial sentences, and follow-up clarifications?",
                "failure_modes": ["Misunderstood basic customer intent or responses"]
            },
            {
                "key": "active_listening",
                "name": "Active Listening",
                "max_points": 10,
                "check": "Did the bot wait for the customer to finish speaking, avoid interrupting mid-sentence, and verbally acknowledge what was said before responding?",
                "failure_modes": ["Interrupted customer or failed to wait for speaking to end"]
            },
            {
                "key": "need_identification",
                "name": "Need Identification",
                "max_points": 15,
                "check": "Did the bot ask relevant qualifying questions to understand the customer's specific need or stage?",
                "failure_modes": ["Failed to qualify customer needs or stage"]
            },
            {
                "key": "information_accuracy",
                "name": "Information Accuracy",
                "max_points": 10,
                "check": "Did the bot provide correct facts — pricing, product specs, store information, EMI details, offers — without making things up?",
                "failure_modes": ["Gave wrong or hallucinated information"]
            },
            {
                "key": "conversation_coherence",
                "name": "Conversation Coherence",
                "max_points": 10,
                "check": "Logical conversation flow — no repetitive loops, no contradictions between turns, no nonsensical topic jumps.",
                "failure_modes": ["Got stuck in repetitive loops or nonsensical topic jumps"]
            },
            {
                "key": "ownership_resolution",
                "name": "No Next Step / Wrong Product/Price",
                "max_points": 10,
                "check": "Did the bot push toward a clear next step appropriate to the customer's stage — callback from a human, store visit, link sent, or smooth handover to a human agent?",
                "failure_modes": ["No clear call-to-action or wrong product/price recommendation"]
            },
            {
                "key": "closing_confirmation",
                "name": "Closing & Confirmation",
                "max_points": 5,
                "check": "Warm close, confirmation of next steps and any commitments made.",
                "failure_modes": ["Abrupt hang up or poor closing"]
            },
            {
                "key": "complaint",
                "name": "Complaint",
                "max_points": 0,
                "check": "If customer raised a complaint, was it acknowledged and addressed? (NA in virtually all bot calls)",
                "failure_modes": ["Complaint not acknowledged or addressed"]
            }
        ]
    }
}


def get_rubric(lead_source: str) -> dict:
    """Fetch rubric by lead source key. Raises if unknown source."""
    if lead_source not in RUBRICS:
        valid = ", ".join(RUBRICS.keys())
        raise ValueError(
            f"Unknown lead source '{lead_source}'. Must be one of: {valid}"
        )
    return RUBRICS[lead_source]


def format_rubric_for_prompt(lead_source: str) -> str:
    """Turn the rubric into a readable string we can paste into a Gemini prompt."""
    r = get_rubric(lead_source)
    lines = [
        f"LEAD SOURCE: {r['name']}",
        f"CUSTOMER INTENT: {r['description']}",
        "",
        "SCORING PARAMETERS (total 100 points):",
    ]
    for p in r["parameters"]:
        lines.append(f"- {p['key']} ({p['max_points']} pts): {p['check']}")
        if p.get("failure_modes"):
            lines.append(f"  Failure modes: {', '.join(p['failure_modes'])}")
    return "\n".join(lines)