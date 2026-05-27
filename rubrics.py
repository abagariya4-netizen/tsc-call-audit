"""
The 7 scoring rubrics for TSC call audits.
One rubric per lead source. Each is worth 100 points.
"""

RUBRIC_VERSION = "v2"

# Common parameters shared by all 6 human agent rubrics
COMMON_PARAMETERS = [
    {
        "key": "greeting_introduction",
        "name": "Greeting & Introduction",
        "max_points": 5,
        "fatal": False,
        "check": "Did agent greet customer appropriately, introduce themselves, and start without delay (under 3 seconds)?",
        "failure_modes": ["Delay opening (>3 sec)", "Did not greet appropriately"]
    },
    {
        "key": "advisor_behaviour",
        "name": "Advisor Behaviour",
        "max_points": 5,
        "fatal": True,
        "check": "Did agent maintain professional, polite, non-abrupt behaviour?",
        "failure_modes": ["Rude and abrupt behaviour"]
    },
    {
        "key": "complaint",
        "name": "Complaint Handling",
        "max_points": 10,
        "fatal": False,
        "check": "If customer raised a complaint, was it acknowledged and addressed? If NO complaint was raised → score NA",
        "failure_modes": ["Complaint not acknowledged/addressed"]
    },
    {
        "key": "ownership_resolution",
        "name": "Ownership – Resolution & Assistance",
        "max_points": 10,
        "fatal": True,
        "check": "Did agent take ownership — provide correct/complete info, note conversation properly, tag the call per query?",
        "failure_modes": ["Did not note conversation", "Incorrect/incomplete info", "Did not tag the call"]
    },
    {
        "key": "communication",
        "name": "Communication Quality",
        "max_points": 10,
        "fatal": False,
        "check": "Was tone professional, grammar correct, pace appropriate, and did agent avoid speaking over customer?",
        "failure_modes": ["Tone/grammar issues", "Spoke too fast/unclear", "Spoke over caller"]
    },
    {
        "key": "hold_mute",
        "name": "Hold & Mute",
        "max_points": 5,
        "fatal": False,
        "check": "Did agent avoid dead air (>10 sec) and long holds/mutes (>120 sec)?",
        "failure_modes": ["Dead air >10 sec", "Long hold/mute >120 sec"]
    }
]

RUBRICS = {
    "find_a_store": {
        "name": "Find a Store",
        "description": "High-intent customer search for nearest store. Confirm store and push for visit, do not over-pitch product.",
        "parameters": COMMON_PARAMETERS + [
            {
                "key": "store_visit_confirmation",
                "name": "Store Visit Confirmation",
                "max_points": 25,
                "fatal": False,
                "check": "Did agent confirm a specific store visit?",
                "failure_modes": ["Did not confirm store visit"]
            },
            {
                "key": "visit_date_locking",
                "name": "Visit Date Locking",
                "max_points": 20,
                "fatal": False,
                "check": "Did agent lock a specific date for the visit?",
                "failure_modes": ["Did not lock a specific date"]
            },
            {
                "key": "closing_with_urgency",
                "name": "Closing with Urgency",
                "max_points": 10,
                "fatal": False,
                "check": "Did agent close the call creating urgency (limited offer, etc.)?",
                "failure_modes": ["Did not create urgency at close"]
            }
        ]
    },

    "arrange_callback": {
        "name": "Arrange Callback",
        "description": "Exploratory request from website. Qualify customer sleep needs first, recommend product, push next step.",
        "parameters": COMMON_PARAMETERS + [
            {
                "key": "need_identification",
                "name": "Need Identification",
                "max_points": 20,
                "fatal": False,
                "check": "Did agent deeply qualify the customer's specific needs?",
                "failure_modes": ["Did not qualify sleep needs"]
            },
            {
                "key": "product_recommendation",
                "name": "Product Recommendation",
                "max_points": 20,
                "fatal": False,
                "check": "Did agent recommend the right product based on identified needs?",
                "failure_modes": ["Did not recommend appropriate product"]
            },
            {
                "key": "conversion_push",
                "name": "Conversion Push",
                "max_points": 15,
                "fatal": False,
                "check": "Did agent push the customer toward a conversion action (next step, callback time, store visit, purchase)?",
                "failure_modes": ["Did not push for conversion action"]
            }
        ]
    },

    "inbound": {
        "name": "Inbound Call",
        "description": "Highest intent. Customer initiated call. Listen actively, answer questions directly, recommend next step.",
        "parameters": COMMON_PARAMETERS + [
            {
                "key": "active_listening",
                "name": "Active Listening",
                "max_points": 25,
                "fatal": False,
                "check": "Did agent actively listen before jumping to a response (acknowledge customer's question, no interruptions)?",
                "failure_modes": ["Interrupted customer", "Did not actively listen"]
            },
            {
                "key": "need_based_response",
                "name": "Need-Based Response",
                "max_points": 20,
                "fatal": False,
                "check": "Did agent's response specifically address the customer's question?",
                "failure_modes": ["Response ignored customer's specific query"]
            },
            {
                "key": "resolution_or_recommendation",
                "name": "Resolution or Recommendation",
                "max_points": 10,
                "fatal": False,
                "check": "Did agent resolve the query or recommend a clear next step?",
                "failure_modes": ["Did not resolve query or recommend next step"]
            }
        ]
    },

    "shopflo_abandoned_cart": {
        "name": "Shopflo Abandoned Cart",
        "description": "High intent recovery. Customer abandoned cart. Probe to find the blocker (price, delivery, trust) and resolve it.",
        "parameters": COMMON_PARAMETERS + [
            {
                "key": "cart_context",
                "name": "Cart Context",
                "max_points": 20,
                "fatal": False,
                "check": "Did agent reference what customer had abandoned (specific product, what they were viewing)?",
                "failure_modes": ["Did not reference abandoned product/context"]
            },
            {
                "key": "blocker_resolution",
                "name": "Blocker Resolution",
                "max_points": 25,
                "fatal": False,
                "check": "Did agent identify and resolve the specific blocker that caused the abandonment (price, delivery, doubt, etc.)?",
                "failure_modes": ["Did not identify or resolve purchase blocker"]
            },
            {
                "key": "purchase_push",
                "name": "Purchase Push",
                "max_points": 10,
                "fatal": False,
                "check": "Did agent push customer to complete the purchase (offer, urgency, payment link, follow-up)?",
                "failure_modes": ["Did not push for purchase completion"]
            }
        ]
    },

    "next_day_delivery": {
        "name": "Next Day Delivery",
        "description": "Urgency focus. Confirm pincode eligibility, explain the free pillow guarantee, close quickly.",
        "parameters": COMMON_PARAMETERS + [
            {
                "key": "ndd_eligibility_check",
                "name": "NDD Eligibility Check",
                "max_points": 15,
                "fatal": False,
                "check": "Did agent verify customer's eligibility for next-day delivery (PIN code, product, etc.)?",
                "failure_modes": ["Did not verify NDD pincode eligibility"]
            },
            {
                "key": "ndd_guarantee_explanation",
                "name": "NDD Guarantee Explanation",
                "max_points": 20,
                "fatal": False,
                "check": "Did agent explain the ₹999 free pillow guarantee if NDD fails?",
                "failure_modes": ["Did not explain the free pillow guarantee"]
            },
            {
                "key": "immediate_purchase_close",
                "name": "Immediate Purchase Close",
                "max_points": 20,
                "fatal": False,
                "check": "Did agent push for immediate purchase before the NDD window closed?",
                "failure_modes": ["Did not push for immediate purchase"]
            }
        ]
    },

    "no_cost_emi": {
        "name": "No-Cost EMI",
        "description": "Price sensitive. Explain zero-interest EMI math clearly, do monthly breakdowns, mention bank offers.",
        "parameters": COMMON_PARAMETERS + [
            {
                "key": "emi_explanation_clarity",
                "name": "EMI Explanation Clarity",
                "max_points": 25,
                "fatal": False,
                "check": "Did agent explain the EMI math clearly (total amount, monthly payment, tenure, zero interest)?",
                "failure_modes": ["EMI details unclear or missing"]
            },
            {
                "key": "product_emi_breakdown",
                "name": "Product EMI Breakdown",
                "max_points": 20,
                "fatal": False,
                "check": "Did agent give a specific product + EMI breakdown for that customer's chosen product?",
                "failure_modes": ["Did not provide specific product EMI breakdown"]
            },
            {
                "key": "bank_offer_communication",
                "name": "Bank Offer Communication",
                "max_points": 10,
                "fatal": False,
                "check": "Did agent mention bank-specific offers (HDFC, ICICI, etc.)?",
                "failure_modes": ["Did not mention bank offers"]
            }
        ]
    },

    "sales_team": {
        "name": "Sales Team Audit Baseline",
        "description": "Baseline rubric used for standard sales team evaluation. Focuses on deep customer qualification, SmartGRID technology pitch, and professional closing.",
        "parameters": COMMON_PARAMETERS + [
            {
                "key": "understanding_customer_needs",
                "name": "Understanding Customer Needs",
                "max_points": 20,
                "fatal": False,
                "check": "Did agent deeply qualify customer sleep issues, posture, bed size, and requirements?",
                "failure_modes": ["Did not qualify sleep needs/posture"]
            },
            {
                "key": "sales_pitch",
                "name": "Sales Pitch",
                "max_points": 20,
                "fatal": False,
                "check": "Did agent explain SmartGRID technology, comfort, and state premium benefits?",
                "failure_modes": ["Weak or missing product pitch"]
            },
            {
                "key": "closing",
                "name": "Closing",
                "max_points": 15,
                "fatal": False,
                "check": "Did agent secure the booking or process payment with clear next steps?",
                "failure_modes": ["No clear call-to-action or closing attempt"]
            }
        ]
    },

    "ai_voice_bot": {
        "name": "AI Voice Bot",
        "description": "Automated call analysis. Evaluate conversation quality, understanding, latency, and loops.",
        "parameters": [
            {
                "key": "greeting_introduction",
                "name": "Opening",
                "max_points": 5,
                "fatal": False,
                "check": "Did the bot greet professionally, identify The Sleep Company (TSC), and transition smoothly into the conversation?",
                "failure_modes": ["Unprofessional or delayed opening"]
            },
            {
                "key": "advisor_behaviour",
                "name": "Bot Conduct / Robotic Delivery",
                "max_points": 15,
                "fatal": False,
                "check": "Voice quality — natural pacing, clear pronunciation, no robotic monotone, appropriate inflection and warmth.",
                "failure_modes": ["Robotic, scripted, or monotone delivery throughout"]
            },
            {
                "key": "customer_understanding",
                "name": "Customer Understanding",
                "max_points": 20,
                "fatal": False,
                "check": "Did the bot correctly understand what the customer said, including Hindi/English code-switching, accents, partial sentences, and follow-up clarifications?",
                "failure_modes": ["Misunderstood basic customer intent or responses"]
            },
            {
                "key": "active_listening",
                "name": "Active Listening",
                "max_points": 10,
                "fatal": False,
                "check": "Did the bot wait for the customer to finish speaking, avoid interrupting mid-sentence, and verbally acknowledge what was said before responding?",
                "failure_modes": ["Interrupted customer or failed to wait for speaking to end"]
            },
            {
                "key": "need_identification",
                "name": "Need Identification",
                "max_points": 15,
                "fatal": False,
                "check": "Did the bot ask relevant qualifying questions to understand the customer's specific need or stage?",
                "failure_modes": ["Failed to qualify customer needs or stage"]
            },
            {
                "key": "information_accuracy",
                "name": "Information Accuracy",
                "max_points": 10,
                "fatal": False,
                "check": "Did the bot provide correct facts — pricing, product specs, store information, EMI details, offers — without making things up?",
                "failure_modes": ["Gave wrong or hallucinated information"]
            },
            {
                "key": "conversation_coherence",
                "name": "Conversation Coherence",
                "max_points": 10,
                "fatal": False,
                "check": "Logical conversation flow — no repetitive loops, no contradictions between turns, no nonsensical topic jumps.",
                "failure_modes": ["Got stuck in repetitive loops or nonsensical topic jumps"]
            },
            {
                "key": "ownership_resolution",
                "name": "No Next Step / Wrong Product/Price",
                "max_points": 10,
                "fatal": False,
                "check": "Did the bot push toward a clear next step appropriate to the customer's stage — callback from a human, store visit, link sent, or smooth handover to a human agent?",
                "failure_modes": ["No clear call-to-action or wrong product/price recommendation"]
            },
            {
                "key": "closing_confirmation",
                "name": "Closing & Confirmation",
                "max_points": 5,
                "fatal": False,
                "check": "Warm close, confirmation of next steps and any commitments made.",
                "failure_modes": ["Abrupt hang up or poor closing"]
            },
            {
                "key": "complaint",
                "name": "Complaint",
                "max_points": 0,
                "fatal": False,
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