"""Few-shot prompt for the LLM classifier.

The instruction lists every intent with its gloss so the model never guesses what
a label means, and includes a handful of bilingual examples so it locks onto the
output contract (a single JSON object) for both Arabic and English input.
"""

from __future__ import annotations

from src.schema import INTENT_DESCRIPTIONS

_LABEL_BLOCK = "\n".join(f"- {name}: {desc}" for name, desc in INTENT_DESCRIPTIONS.items())

SYSTEM_PROMPT = f"""You are an intent classifier for a bilingual (Arabic/English) \
customer-support queue. Classify the user's message into exactly ONE of these intents:

{_LABEL_BLOCK}

Rules:
- Respond with a JSON object of the form {{"intent": "<one_of_the_labels>"}} and nothing else.
- The value MUST be one of the labels above, lowercase, exactly as written.
- Judge the underlying need, not surface keywords. Works for both Arabic and English."""

# Few-shot exemplars (kept short; one AR + one EN per a few representative intents).
FEW_SHOT: list[tuple[str, str]] = [
    ("I was charged twice this month", "billing_issue"),
    ("عايز ألغي اشتراكي", "cancel_subscription"),
    ("Can I get my money back?", "refund_request"),
    ("التطبيق بيقفل أول ما يفتح", "technical_support"),
    ("عندكم باقة للفرق؟", "product_inquiry"),
    ("صباح الخير", "greeting"),
]


def build_messages(text: str) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example_text, label in FEW_SHOT:
        messages.append({"role": "user", "content": example_text})
        messages.append({"role": "assistant", "content": f'{{"intent": "{label}"}}'})
    messages.append({"role": "user", "content": text})
    return messages
