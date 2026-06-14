"""Generate a small bilingual (Arabic/English) intent dataset, deterministically.

Run:  python data/generate.py
Produces:  data/intents.jsonl (train) and data/eval_set.jsonl (held-out ground truth).

No randomness: the split is "last K utterances per (intent, language) go to eval",
so results are reproducible and the eval set never leaks into training.
"""

from __future__ import annotations

import json
from pathlib import Path

# Bilingual utterances per intent. Real-sounding, short, mixed dialect/MSA on the
# Arabic side to mirror what an Arabic-facing support queue actually receives.
UTTERANCES: dict[str, list[str]] = {
    "greeting": [
        "Hi there!", "Hello, good morning", "Hey, how are you?", "Good evening",
        "Hi, anyone online?", "Greetings", "Hello team", "Hey, just saying hi",
        "السلام عليكم", "صباح الخير", "أهلاً، إزيكم؟", "مرحبا",
        "هاي، في حد موجود؟", "مساء الخير", "أهلاً وسهلاً", "إزي الحال؟",
    ],
    "billing_issue": [
        "I was charged twice this month", "My invoice amount looks wrong",
        "Why was I billed 99 instead of 49?", "The payment failed but money was taken",
        "There is an extra charge on my card", "My bill is higher than my plan",
        "I got charged after I downgraded", "Double payment on my statement",
        "اتخصم مني المبلغ مرتين الشهر ده", "الفاتورة غلط ومش مظبوطة",
        "ليه اتحاسبت 99 بدل 49؟", "الدفع فشل بس الفلوس اتخصمت",
        "في رسوم زيادة على الكارت", "الفاتورة أعلى من باقتي",
        "اتحاسبت بعد ما نزّلت الباقة", "اتدفع مرتين في الكشف",
    ],
    "technical_support": [
        "The app keeps crashing on launch", "I can't log in to my account",
        "I get a 500 error when I save", "The page won't load",
        "Upload button does nothing", "Export feature is broken",
        "Login says wrong password but it's correct", "Dashboard is stuck loading",
        "التطبيق بيقفل أول ما يفتح", "مش قادر أعمل لوجين",
        "بييجي إيرور 500 لما بحفظ", "الصفحة مش بتفتح",
        "زرار الرفع مش شغال", "خاصية التصدير باظت",
        "بيقول الباسورد غلط وهو صح", "الداشبورد واقفة بتحمّل",
    ],
    "cancel_subscription": [
        "I want to cancel my subscription", "Please stop my plan",
        "How do I end my membership?", "Cancel my account renewal",
        "I'd like to terminate the service", "Stop charging me, cancel it",
        "End my subscription today", "I no longer want the subscription",
        "عايز ألغي اشتراكي", "من فضلك أوقف الباقة",
        "إزاي ألغي العضوية؟", "ألغوا تجديد الحساب",
        "عايز أنهي الخدمة", "بطلوا تخصموا، ألغوا الاشتراك",
        "اقفل اشتراكي النهاردة", "مش عايز الاشتراك تاني",
    ],
    "refund_request": [
        "I want a refund for last month", "Can I get my money back?",
        "Please refund the duplicate charge", "I need a refund, the product didn't work",
        "Refund me for the annual plan", "I'd like my payment returned",
        "Give me back the charge from yesterday", "Requesting a full refund",
        "عايز استرجاع فلوس الشهر اللي فات", "ممكن ترجعولي فلوسي؟",
        "رجّعوا الرسوم المكررة", "محتاج ريفاند، المنتج مشتغلش",
        "استرجعوا مبلغ الباقة السنوية", "عايز فلوسي ترجع",
        "رجّعوا الخصم بتاع امبارح", "بطلب استرجاع كامل",
    ],
    "complaint": [
        "Your support is terrible", "I'm very disappointed with the service",
        "This is the worst experience ever", "Nobody answers my tickets",
        "I'm unhappy with how slow you are", "Really bad service quality",
        "I've been waiting for days, unacceptable", "Your team is unhelpful",
        "الدعم بتاعكم سيء جداً", "أنا متضايق جداً من الخدمة",
        "أسوأ تجربة على الإطلاق", "محدش بيرد على التذاكر",
        "مش راضي عن البطء بتاعكم", "جودة الخدمة وحشة فعلاً",
        "مستني من كذا يوم، ده مرفوض", "فريقكم مش بيساعد",
    ],
    "product_inquiry": [
        "Do you offer a team plan?", "What features are in the pro tier?",
        "How much is the yearly subscription?", "Is there a free trial?",
        "Does it integrate with Slack?", "What's the difference between plans?",
        "Can I use it offline?", "Do you support single sign-on?",
        "عندكم باقة للفرق؟", "إيه المميزات في باقة البرو؟",
        "بكام الاشتراك السنوي؟", "في فترة تجربة مجانية؟",
        "بيتكامل مع سلاك؟", "إيه الفرق بين الباقات؟",
        "أقدر أستخدمه أوفلاين؟", "بتدعموا تسجيل الدخول الموحّد؟",
    ],
    "account_update": [
        "I need to change my email", "Please update my password",
        "Change my billing address", "Update the name on my account",
        "I want to switch to the annual plan", "Update my phone number",
        "Change my account email to the new one", "Upgrade me to the pro plan",
        "محتاج أغيّر الإيميل", "من فضلك حدّثوا الباسورد",
        "غيّروا عنوان الفوترة", "حدّثوا الاسم على الحساب",
        "عايز أحوّل للباقة السنوية", "حدّثوا رقم تليفوني",
        "غيّروا إيميل الحساب للجديد", "رقّوني لباقة البرو",
    ],
}

# Per (intent, language) we hold out the last EVAL_PER_GROUP utterances for the
# eval set. With 8 EN + 8 AR per intent, 2 per group -> 25% eval, balanced.
EVAL_PER_GROUP = 2


def _split(items: list[str]) -> tuple[list[str], list[str]]:
    """First half = EN, second half = AR. Hold out the tail of each as eval."""
    half = len(items) // 2
    en, ar = items[:half], items[half:]
    train = en[:-EVAL_PER_GROUP] + ar[:-EVAL_PER_GROUP]
    eval_ = en[-EVAL_PER_GROUP:] + ar[-EVAL_PER_GROUP:]
    return train, eval_


def main() -> None:
    out_dir = Path(__file__).parent
    train_rows, eval_rows = [], []

    for intent, items in UTTERANCES.items():
        train, eval_ = _split(items)
        train_rows += [{"text": t, "label": intent} for t in train]
        eval_rows += [{"text": t, "label": intent} for t in eval_]

    for name, rows in (("intents.jsonl", train_rows), ("eval_set.jsonl", eval_rows)):
        path = out_dir / name
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows):>3} rows -> {path}")


if __name__ == "__main__":
    main()
