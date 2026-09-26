import pytest

from src.llm_classifier.classifier import _extract_label


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"intent": "refund_request"}', "refund_request"),
        ('{"intent": "  Billing_Issue "}', "billing_issue"),
        ('Sure! {"intent": "cancel_subscription"} hope that helps', "cancel_subscription"),
        ("I think this is technical_support.", "technical_support"),
        ('{"intent": "not_a_label"}', "complaint"),
        ("no idea", "complaint"),
        ("", "complaint"),
        ("[]", "complaint"),
    ],
)
def test_extract_label_always_returns_a_valid_label(raw: str, expected: str) -> None:
    assert _extract_label(raw) == expected
