"""Shared contract for both classifiers: the intent label set and the prediction type.

Keeping the label set in one place means the encoder, the LLM prompt, the eval
harness, and the API can never drift out of sync.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Intent(str, Enum):
    """Customer-support intents. Bilingual (AR/EN) by design."""

    GREETING = "greeting"
    BILLING_ISSUE = "billing_issue"
    TECHNICAL_SUPPORT = "technical_support"
    CANCEL_SUBSCRIPTION = "cancel_subscription"
    REFUND_REQUEST = "refund_request"
    COMPLAINT = "complaint"
    PRODUCT_INQUIRY = "product_inquiry"
    ACCOUNT_UPDATE = "account_update"


# Stable, sorted label order. The encoder uses this for its id<->label mapping,
# so DO NOT reorder after a model is trained.
LABELS: list[str] = [i.value for i in Intent]

# Short natural-language gloss for each intent. Fed to the LLM as part of the
# instruction so it knows exactly what each label means (no guessing).
INTENT_DESCRIPTIONS: dict[str, str] = {
    Intent.GREETING.value: "A greeting, small talk, or opening message with no concrete request.",
    Intent.BILLING_ISSUE.value: "A problem with an invoice, charge, payment failure, or billing amount.",
    Intent.TECHNICAL_SUPPORT.value: "Something is broken or not working: errors, bugs, login/access problems.",
    Intent.CANCEL_SUBSCRIPTION.value: "The user wants to cancel, stop, or end their subscription/service.",
    Intent.REFUND_REQUEST.value: "The user wants money back for a charge or purchase.",
    Intent.COMPLAINT.value: "Dissatisfaction or a complaint about service/quality, without a specific fix request.",
    Intent.PRODUCT_INQUIRY.value: "A pre-sale question about features, plans, pricing, or how the product works.",
    Intent.ACCOUNT_UPDATE.value: "Updating account details: email, password, name, address, plan change.",
}


@dataclass
class Prediction:
    """Unified result type returned by both classifiers and the API."""

    intent: str
    confidence: float | None = None  # None when the backend does not expose one
    latency_ms: float | None = None
    method: str | None = None  # "encoder" | "llm"


def json_schema() -> dict:
    """JSON Schema used to constrain the LLM's structured output to a valid label."""
    return {
        "name": "intent_classification",
        "schema": {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": LABELS},
            },
            "required": ["intent"],
            "additionalProperties": False,
        },
        "strict": True,
    }
