"""
The 6 scoring rubrics for TSC call audits.
One rubric per lead source. Each is worth 100 points.
Red flags apply on top (deductions).
"""

RUBRICS = {
    "find_a_store": {
        "name": "Find a Store - Store Found",
        "intent": "HIGH-INTENT customer searched for nearest store on WhatsApp. They already want to visit. Job is to confirm store + push for visit, NOT to sell on phone.",
        "parameters": [
            ("Opening & Introduction", 5, "Did agent greet professionally, identify self and TSC, confirm customer's name?"),
            ("Store Visit Confirmation", 25, "PRIMARY METRIC. Did agent confirm/share the nearest store address, store timings, and explicitly invite the customer to visit?"),
            ("Visit Date Locking", 20, "Did agent ask when the customer plans to visit and try to lock a specific day/time?"),
            ("Urgency Creation", 15, "Did agent create urgency — limited time offer, weekend rush, stock availability — to bring the visit forward?"),
            ("Brief Product Overview", 10, "Did agent give a SHORT product overview (Smart Grid tech, sizes, comfort levels) without over-pitching?"),
            ("Offer Communication", 10, "Did agent mention current store-exclusive offers, EMI options, or trial policy clearly?"),
            ("Call Control & Professionalism", 10, "Tone, pace, language clarity, empathy, no fillers or interrupting the customer."),
            ("Closing", 5, "Did agent recap the store address/time and end the call warmly?"),
        ],
        "red_flags": [
            ("Pushed online purchase instead of store visit", -15),
            ("Did not share store address/timings clearly", -10),
            ("Excessive product pitching that delayed store push", -5),
            ("Shared wrong information (store, offers, products)", -10),
            ("Rude or impatient with customer", -10),
            ("No self-introduction at start", -5),
        ],
    },

    "arrange_callback": {
        "name": "Arrange Call Back",
        "intent": "EXPLORATORY. Customer requested callback from website. Still exploring. Job is to ASK qualifying questions first, then recommend product, then push store visit.",
        "parameters": [
            ("Opening", 5, "Greet, introduce self + TSC, reference the callback request."),
            ("Need Identification", 20, "Did agent ask qualifying questions — sleep issues, current mattress, budget, family size, preferences — BEFORE pitching?"),
            ("Product Recommendation", 20, "Did agent recommend a specific TSC product matched to the customer's stated needs?"),
            ("Product Education", 15, "Clear explanation of features — Smart Grid, edge support, trial, warranty — relevant to the customer's pain points."),
            ("Pricing & Offers", 10, "Shared accurate pricing, current discounts, EMI, free pillow / free delivery offers."),
            ("Conversion Push", 15, "Pushed for store visit OR online purchase with clear next step."),
            ("Objection Handling", 5, "Addressed any pushback (price, brand, timing) with empathy + facts."),
            ("Call Control", 5, "Tone, pace, professionalism."),
            ("Closing", 5, "Recapped next step, warm close."),
        ],
        "red_flags": [
            ("Pitched product without qualifying customer's needs", -10),
            ("Recommended wrong product for stated need", -10),
            ("Quoted wrong pricing or offers", -10),
            ("Rude or dismissive", -10),
            ("No clear call-to-action at end", -5),
        ],
    },

    "inbound": {
        "name": "Inbound Phone Call",
        "intent": "HIGHEST INTENT. Customer called us. Job is to LISTEN FIRST, understand exactly what they need, then respond.",
        "parameters": [
            ("Opening", 5, "Picked up promptly, professional greeting, identified self + TSC."),
            ("Active Listening", 25, "Let customer speak first, did not interrupt, asked clarifying questions, acknowledged their query."),
            ("Need-Based Response", 20, "Response directly addressed what the customer actually asked (not a scripted pitch)."),
            ("Product Recommendation / Resolution", 15, "Provided the right product info OR resolved the query the customer had."),
            ("Pricing & Offers", 10, "Accurate pricing, EMI, current offers — only when relevant to the query."),
            ("Conversion", 10, "Pushed for next step — store visit, online purchase, or callback — appropriately."),
            ("Call Control", 10, "Tone, empathy, no rushing, no jargon."),
            ("Closing", 5, "Recapped resolution, thanked customer, warm close."),
        ],
        "red_flags": [
            ("Launched into scripted pitch without listening", -15),
            ("Rushed the customer or interrupted", -10),
            ("Could not answer customer's question", -5),
            ("Shared wrong information", -10),
            ("Rude or impatient", -10),
        ],
    },

    "shopflo_abandoned_cart": {
        "name": "Shopflo Abandoned Cart",
        "intent": "HIGH INTENT. Customer added to cart but didn't buy. Something blocked them. Job is to find the blocker, remove it, close the sale.",
        "parameters": [
            ("Opening", 5, "Greet, introduce, reference the abandoned cart."),
            ("Cart Context & Reason", 25, "Did agent ask why the customer didn't complete the purchase? Probed for the real blocker?"),
            ("Blocker Resolution", 25, "Did agent address the specific blocker — price, doubt about product, delivery, EMI, trust — with a concrete answer?"),
            ("Offer", 15, "Used an appropriate offer (discount, free pillow, EMI, trial) to nudge the close."),
            ("Purchase Push", 15, "Asked for the sale, guided to checkout or store visit with clarity."),
            ("Call Control", 10, "Tone, patience, non-pushy professionalism."),
            ("Closing", 5, "Recapped agreement and next step."),
        ],
        "red_flags": [
            ("Aggressive or pushy sales tactics", -10),
            ("Did not ask why customer abandoned cart", -10),
            ("Generic pitch ignoring the cart context", -5),
            ("Wrong product, pricing, or offer info", -10),
            ("Rude or dismissive", -10),
        ],
    },

    "next_day_delivery": {
        "name": "Next Day Delivery",
        "intent": "URGENCY. Customer wants next-day delivery. Job is to confirm pincode eligibility, explain the free pillow guarantee, close fast.",
        "parameters": [
            ("Opening", 5, "Greet, introduce, reference NDD enquiry."),
            ("NDD Eligibility", 20, "Did agent confirm the customer's pincode is eligible for next-day delivery?"),
            ("NDD Guarantee Explanation", 20, "Did agent clearly explain: if not delivered next day, customer gets a FREE ₹999 pillow?"),
            ("Need ID + Product", 15, "Identified product fit briefly (size, comfort level) without over-pitching."),
            ("Pricing & Offers", 15, "Accurate pricing, payment options, NDD-related offers."),
            ("Purchase Closure", 10, "Pushed for immediate purchase given the urgency angle."),
            ("Call Control", 10, "Fast but warm; no rushing the customer."),
            ("Closing", 5, "Recapped order details, delivery promise, and contact for follow-up."),
        ],
        "red_flags": [
            ("Could not confirm NDD pincode eligibility", -10),
            ("Did not explain the free ₹999 pillow guarantee", -10),
            ("Spent too long on general product education", -5),
            ("Wrong delivery information", -10),
            ("Rude or rushed", -10),
        ],
    },

    "no_cost_emi": {
        "name": "No-Cost EMI",
        "intent": "PRICE SENSITIVE. Customer wants EMI option. Job is to explain EMI clearly, calculate monthly amount, close.",
        "parameters": [
            ("Opening", 5, "Greet, introduce, reference EMI enquiry."),
            ("EMI Explanation", 25, "Did agent explain No-Cost EMI clearly — tenure options, zero interest, how it works?"),
            ("Need ID", 10, "Identified product fit and budget range."),
            ("Product + EMI Breakdown", 20, "Gave a concrete monthly EMI number for the recommended product across at least 2 tenures (e.g., 6 / 9 / 12 months)?"),
            ("Bank-Specific Offers", 10, "Asked which bank/card the customer has and shared any bank-specific cashback or instant discount?"),
            ("Payment Friction Removal", 10, "Addressed any concerns — application process, documentation, eligibility — clearly."),
            ("Purchase Push", 10, "Asked for the sale with a clear next step (store visit / online checkout)."),
            ("Call Control", 5, "Tone, clarity, patience with numbers."),
            ("Closing", 5, "Recapped EMI plan and next step."),
        ],
        "red_flags": [
            ("Could not explain EMI clearly", -15),
            ("Quoted wrong EMI numbers", -10),
            ("Dismissed customer's budget concerns", -10),
            ("Did not ask which bank/card customer has", -5),
            ("Rude or impatient", -10),
        ],
    },

    "ai_voice_bot": {
        "name": "AI Voice Bot",
        "intent": "AUTOMATED. An AI voice bot conducts the conversation with the lead, not a human agent. Evaluated on conversation quality (understanding, listening, need ID, accuracy, conversion) AND on bot-specific failure modes (getting stuck in loops, hallucinated answers, awkward latency, missed escalations) that don't apply to human agents.",
        "parameters": [
            ("Opening", 5, "Did the bot greet professionally, identify The Sleep Company (TSC), and transition smoothly into the conversation?"),
            ("Speech Naturalness & Clarity", 15, "Voice quality — natural pacing, clear pronunciation, no robotic monotone, appropriate inflection and warmth."),
            ("Customer Understanding", 20, "Did the bot correctly understand what the customer said, including Hindi/English code-switching, accents, partial sentences, and follow-up clarifications?"),
            ("Active Listening", 10, "Did the bot wait for the customer to finish speaking, avoid interrupting mid-sentence, and verbally acknowledge what was said before responding?"),
            ("Need Identification", 15, "Did the bot ask relevant qualifying questions to understand the customer's specific need or stage?"),
            ("Information Accuracy", 10, "Did the bot provide correct facts — pricing, product specs, store information, EMI details, offers — without making things up?"),
            ("Conversation Coherence", 10, "Logical conversation flow — no repetitive loops, no contradictions between turns, no nonsensical topic jumps."),
            ("Conversion / Next-Step Clarity", 10, "Did the bot push toward a clear next step appropriate to the customer's stage — callback from a human, store visit, link sent, or smooth handover to a human agent?"),
            ("Closing & Confirmation", 5, "Warm close, confirmation of next steps and any commitments made."),
        ],
        "red_flags": [
            ("Got stuck in a loop or repeated the same response multiple times", -15),
            ("Gave wrong or hallucinated information about products, prices, or store", -15),
            ("Hung up on the customer abruptly without warning", -15),
            ("Misunderstood basic customer intent and pushed wrong product or direction", -10),
            ("Failed to escalate to a human when the customer asked or got frustrated", -10),
            ("Long dead-air or awkward latency between responses", -10),
            ("Robotic, scripted, or monotone delivery throughout", -5),
        ],
    },
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
        f"CUSTOMER INTENT: {r['intent']}",
        "",
        "SCORING PARAMETERS (total 100 points):",
    ]
    for name, points, desc in r["parameters"]:
        lines.append(f"- {name} ({points} pts): {desc}")
    lines.append("")
    lines.append("RED FLAGS (apply as deductions on top of the score):")
    for desc, penalty in r["red_flags"]:
        lines.append(f"- {desc}: {penalty} pts")
    return "\n".join(lines)