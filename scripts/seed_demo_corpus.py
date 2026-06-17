"""Generate the 30-document synthetic golden corpus.

Run once to materialise `eval/golden/*.json`. The output is deterministic
under a fixed seed so the eval numbers are reproducible across runs.

Usage:
    uv run python scripts/seed_demo_corpus.py
    uv run python scripts/seed_demo_corpus.py --out eval/golden
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

from guardian_rag.eval.models import GoldenDocument, GoldenEntity, GoldenQA
from guardian_rag.pii.models import PiiEntityType

# ── Synthetic pool ──────────────────────────────────────────────────────────
# Names, emails, and IBANs are deliberately fake. IBANs are the standard
# example values from the official IBAN registry; credit cards are the
# IANA test numbers (e.g. 4111-1111-1111-1111). Nothing here links to a
# real person.

NAMES = [
    "Alice Bennett",
    "Bob Chen",
    "Carla Diaz",
    "David Owens",
    "Elena Petrov",
    "Faisal Khan",
    "Gloria Reyes",
    "Hiro Tanaka",
    "Iris Whyte",
    "Jack Murphy",
]

EMAILS = [
    "alice.bennett@example.com",
    "bob.chen@example.com",
    "carla.diaz@example.com",
    "david.owens@example.com",
    "elena.petrov@example.com",
    "faisal.khan@example.com",
    "gloria.reyes@example.com",
    "hiro.tanaka@example.com",
    "iris.whyte@example.com",
    "jack.murphy@example.com",
]

PHONES = [
    "+1-555-0142",
    "+44-7700-900123",
    "+34-91-555-0987",
    "+49-30-555-0234",
    "+81-3-5555-1234",
    "+353-1-555-0567",
    "+1-555-0199",
    "+33-1-55-55-01-23",
    "+39-06-555-0876",
    "+31-20-555-0345",
]

IBANS = [
    "BE71096123456769",
    "DE89370400440532013000",
    "FR1420041010050500013M02606",
    "ES9121000418450200051332",
    "GB29NWBK60161331926819",
    "IT60X0542811101000000123456",
    "NL91ABNA0417164300",
    "AT611904300234573201",
    "CH9300762011623852957",
    "IE29AIBK93115212345678",
]

CARDS = [
    "4111-1111-1111-1111",
    "4012-8888-8888-1881",
    "5555-5555-5555-4444",
    "3782-822463-10005",
    "6011-1111-1111-1117",
    "3056-9300-9025-9047",
    "5105-1051-0510-5100",
    "4222-2222-2222-2",
    "4242-4242-4242-4242",
    "5454-5454-5454-5454",
]

IPS = [
    "10.0.0.42",
    "192.168.1.105",
    "172.16.254.7",
    "203.0.113.45",
    "198.51.100.8",
    "10.10.10.10",
    "192.0.2.146",
    "203.0.113.99",
    "10.20.30.40",
    "172.31.255.1",
]

PERSON = PiiEntityType.PERSON
EMAIL = PiiEntityType.EMAIL
PHONE = PiiEntityType.PHONE
BANK = PiiEntityType.BANK_ACCOUNT
CARD = PiiEntityType.CREDIT_CARD
IP = PiiEntityType.IP_ADDRESS

Segment = tuple[str, PiiEntityType | None]


def _build(segments: Sequence[Segment]) -> tuple[str, list[GoldenEntity]]:
    """Concatenate `segments` into text and emit a GoldenEntity per labelled span."""
    text = ""
    entities: list[GoldenEntity] = []
    for seg_text, etype in segments:
        start = len(text)
        text += seg_text
        end = len(text)
        if etype is not None:
            entities.append(GoldenEntity(text=seg_text, entity_type=etype, start=start, end=end))
    return text, entities


# ── HR templates ────────────────────────────────────────────────────────────
def hr_onboarding(i: int) -> tuple[str, list[GoldenEntity], list[GoldenQA]]:
    text, ents = _build(
        [
            ("Welcome to the company, ", None),
            (NAMES[i % len(NAMES)], PERSON),
            (". Your manager will be ", None),
            (NAMES[(i + 3) % len(NAMES)], PERSON),
            (". For any HR question, write to ", None),
            (EMAILS[(i + 1) % len(EMAILS)], EMAIL),
            (" or call ", None),
            (PHONES[i % len(PHONES)], PHONE),
            (".", None),
        ]
    )
    qa = [
        GoldenQA(
            question="Who should I contact for HR questions?",
            reference_answer="The HR contact email is the address mentioned in the document.",
        )
    ]
    return text, ents, qa


def hr_performance_review(i: int) -> tuple[str, list[GoldenEntity], list[GoldenQA]]:
    text, ents = _build(
        [
            ("Performance review for ", None),
            (NAMES[i % len(NAMES)], PERSON),
            (". Reviewer: ", None),
            (NAMES[(i + 2) % len(NAMES)], PERSON),
            (". Overall rating: exceeds expectations. Recommended raise paid to ", None),
            (IBANS[i % len(IBANS)], BANK),
            (".", None),
        ]
    )
    return text, ents, []


def hr_salary_adjustment(i: int) -> tuple[str, list[GoldenEntity], list[GoldenQA]]:
    text, ents = _build(
        [
            ("Salary adjustment notice for ", None),
            (NAMES[i % len(NAMES)], PERSON),
            (". New net salary will be deposited to ", None),
            (IBANS[(i + 1) % len(IBANS)], BANK),
            (" starting next month. Contact payroll at ", None),
            (EMAILS[(i + 4) % len(EMAILS)], EMAIL),
            (".", None),
        ]
    )
    return text, ents, []


def hr_leave_request(i: int) -> tuple[str, list[GoldenEntity], list[GoldenQA]]:
    text, ents = _build(
        [
            ("Leave request approved for ", None),
            (NAMES[i % len(NAMES)], PERSON),
            (". Coverage will be provided by ", None),
            (NAMES[(i + 5) % len(NAMES)], PERSON),
            (". Forward any urgent matter to ", None),
            (EMAILS[(i + 2) % len(EMAILS)], EMAIL),
            (".", None),
        ]
    )
    return text, ents, []


def hr_disciplinary_notice(i: int) -> tuple[str, list[GoldenEntity], list[GoldenQA]]:
    text, ents = _build(
        [
            ("Disciplinary notice issued to ", None),
            (NAMES[i % len(NAMES)], PERSON),
            (". This notice is filed by HR. Appeals can be sent to ", None),
            (EMAILS[(i + 3) % len(EMAILS)], EMAIL),
            (" within 14 days.", None),
        ]
    )
    return text, ents, []


def hr_promotion(i: int) -> tuple[str, list[GoldenEntity], list[GoldenQA]]:
    text, ents = _build(
        [
            ("We are pleased to announce the promotion of ", None),
            (NAMES[i % len(NAMES)], PERSON),
            (" to senior engineer. Reporting line shifts to ", None),
            (NAMES[(i + 4) % len(NAMES)], PERSON),
            (". HR updates will go through ", None),
            (EMAILS[(i + 6) % len(EMAILS)], EMAIL),
            (".", None),
        ]
    )
    return text, ents, []


def hr_termination(i: int) -> tuple[str, list[GoldenEntity], list[GoldenQA]]:
    text, ents = _build(
        [
            ("Termination notice for ", None),
            (NAMES[i % len(NAMES)], PERSON),
            (". Final settlement will be wired to ", None),
            (IBANS[(i + 5) % len(IBANS)], BANK),
            (" within 30 days. Inquiries: ", None),
            (PHONES[(i + 2) % len(PHONES)], PHONE),
            (".", None),
        ]
    )
    return text, ents, []


def hr_benefits(i: int) -> tuple[str, list[GoldenEntity], list[GoldenQA]]:
    text, ents = _build(
        [
            ("Benefits enrollment confirmation: ", None),
            (NAMES[i % len(NAMES)], PERSON),
            (". Spouse listed: ", None),
            (NAMES[(i + 7) % len(NAMES)], PERSON),
            (". Pension contribution will be debited from ", None),
            (IBANS[(i + 2) % len(IBANS)], BANK),
            (".", None),
        ]
    )
    return text, ents, []


def hr_training(i: int) -> tuple[str, list[GoldenEntity], list[GoldenQA]]:
    text, ents = _build(
        [
            ("Training completion certificate awarded to ", None),
            (NAMES[i % len(NAMES)], PERSON),
            (". Trainer: ", None),
            (NAMES[(i + 1) % len(NAMES)], PERSON),
            (". Certificate copy sent to ", None),
            (EMAILS[(i + 8) % len(EMAILS)], EMAIL),
            (".", None),
        ]
    )
    return text, ents, []


def hr_transfer(i: int) -> tuple[str, list[GoldenEntity], list[GoldenQA]]:
    text, ents = _build(
        [
            ("Internal transfer authorised for ", None),
            (NAMES[i % len(NAMES)], PERSON),
            (" effective next quarter. New manager: ", None),
            (NAMES[(i + 6) % len(NAMES)], PERSON),
            (". Onboarding email: ", None),
            (EMAILS[(i + 5) % len(EMAILS)], EMAIL),
            (". Office phone: ", None),
            (PHONES[(i + 4) % len(PHONES)], PHONE),
            (".", None),
        ]
    )
    return text, ents, []


# ── Contracts templates ─────────────────────────────────────────────────────
def contract_service_agreement(i: int) -> tuple[str, list[GoldenEntity], list[GoldenQA]]:
    text, ents = _build(
        [
            ("Service agreement signed between Acme Corp and the client ", None),
            (NAMES[i % len(NAMES)], PERSON),
            (". Invoices will be wired to ", None),
            (IBANS[i % len(IBANS)], BANK),
            (". Escalations: ", None),
            (EMAILS[(i + 3) % len(EMAILS)], EMAIL),
            (".", None),
        ]
    )
    qa = [
        GoldenQA(
            question="Where will invoices be wired?",
            reference_answer="Invoices will be wired to the bank account listed in the agreement.",
        )
    ]
    return text, ents, qa


def contract_nda(i: int) -> tuple[str, list[GoldenEntity], list[GoldenQA]]:
    text, ents = _build(
        [
            ("Non-disclosure agreement between Globex Industries and ", None),
            (NAMES[i % len(NAMES)], PERSON),
            (". Counterpart contact: ", None),
            (EMAILS[(i + 2) % len(EMAILS)], EMAIL),
            (". Phone: ", None),
            (PHONES[(i + 1) % len(PHONES)], PHONE),
            (".", None),
        ]
    )
    return text, ents, []


def contract_sales(i: int) -> tuple[str, list[GoldenEntity], list[GoldenQA]]:
    text, ents = _build(
        [
            ("Sales contract: buyer ", None),
            (NAMES[i % len(NAMES)], PERSON),
            (" purchases licences from Initech Solutions. Payment card on file ends with ", None),
            (CARDS[i % len(CARDS)], CARD),
            (". Notifications: ", None),
            (EMAILS[(i + 7) % len(EMAILS)], EMAIL),
            (".", None),
        ]
    )
    return text, ents, []


def contract_lease(i: int) -> tuple[str, list[GoldenEntity], list[GoldenQA]]:
    text, ents = _build(
        [
            ("Lease agreement for office space signed by ", None),
            (NAMES[i % len(NAMES)], PERSON),
            (". Rent transfers will land in ", None),
            (IBANS[(i + 3) % len(IBANS)], BANK),
            (". Landlord contact: ", None),
            (PHONES[(i + 6) % len(PHONES)], PHONE),
            (".", None),
        ]
    )
    return text, ents, []


def contract_vendor_onboarding(i: int) -> tuple[str, list[GoldenEntity], list[GoldenQA]]:
    text, ents = _build(
        [
            ("Vendor onboarding for Wayne Enterprises. Primary contact: ", None),
            (NAMES[i % len(NAMES)], PERSON),
            (" at ", None),
            (EMAILS[(i + 4) % len(EMAILS)], EMAIL),
            (". Bank for net-30 payment: ", None),
            (IBANS[(i + 8) % len(IBANS)], BANK),
            (".", None),
        ]
    )
    return text, ents, []


def contract_consulting(i: int) -> tuple[str, list[GoldenEntity], list[GoldenQA]]:
    text, ents = _build(
        [
            ("Consulting agreement: ", None),
            (NAMES[i % len(NAMES)], PERSON),
            (" engaged by Stark Industries. Fees deposited to ", None),
            (IBANS[(i + 4) % len(IBANS)], BANK),
            (". Project lead: ", None),
            (NAMES[(i + 9) % len(NAMES)], PERSON),
            (".", None),
        ]
    )
    return text, ents, []


def contract_partnership(i: int) -> tuple[str, list[GoldenEntity], list[GoldenQA]]:
    text, ents = _build(
        [
            ("Partnership memorandum: Pied Piper and counterpart ", None),
            (NAMES[i % len(NAMES)], PERSON),
            (". Joint account for shared expenses: ", None),
            (IBANS[(i + 6) % len(IBANS)], BANK),
            (". Steering committee email: ", None),
            (EMAILS[(i + 5) % len(EMAILS)], EMAIL),
            (".", None),
        ]
    )
    return text, ents, []


def contract_licensing(i: int) -> tuple[str, list[GoldenEntity], list[GoldenQA]]:
    text, ents = _build(
        [
            ("Licensing terms for Hooli Inc software granted to ", None),
            (NAMES[i % len(NAMES)], PERSON),
            (". Renewal payment card: ", None),
            (CARDS[(i + 2) % len(CARDS)], CARD),
            (". Support: ", None),
            (EMAILS[(i + 9) % len(EMAILS)], EMAIL),
            (".", None),
        ]
    )
    return text, ents, []


def contract_supplier(i: int) -> tuple[str, list[GoldenEntity], list[GoldenQA]]:
    text, ents = _build(
        [
            ("Supplier contract with Aperture Labs. Counterpart signer: ", None),
            (NAMES[i % len(NAMES)], PERSON),
            (". Wire transfers go to ", None),
            (IBANS[(i + 7) % len(IBANS)], BANK),
            (". Counterpart phone: ", None),
            (PHONES[(i + 3) % len(PHONES)], PHONE),
            (".", None),
        ]
    )
    return text, ents, []


def contract_subcontract(i: int) -> tuple[str, list[GoldenEntity], list[GoldenQA]]:
    text, ents = _build(
        [
            ("Subcontracting agreement: prime ", None),
            (NAMES[i % len(NAMES)], PERSON),
            (" engages Tyrell Corporation. Sub-tier payments to ", None),
            (IBANS[(i + 9) % len(IBANS)], BANK),
            (". Notifications: ", None),
            (EMAILS[(i + 6) % len(EMAILS)], EMAIL),
            (".", None),
        ]
    )
    return text, ents, []


# ── Support templates ───────────────────────────────────────────────────────
def support_login(i: int) -> tuple[str, list[GoldenEntity], list[GoldenQA]]:
    text, ents = _build(
        [
            ("Customer ", None),
            (NAMES[i % len(NAMES)], PERSON),
            (" reports they cannot log in. Login attempts came from IP ", None),
            (IPS[i % len(IPS)], IP),
            (". Account email: ", None),
            (EMAILS[i % len(EMAILS)], EMAIL),
            (".", None),
        ]
    )
    qa = [
        GoldenQA(
            question="What IP address did the login attempts come from?",
            reference_answer="The login attempts came from the IP address listed in the ticket.",
        )
    ]
    return text, ents, qa


def support_billing(i: int) -> tuple[str, list[GoldenEntity], list[GoldenQA]]:
    text, ents = _build(
        [
            ("Billing dispute opened by ", None),
            (NAMES[i % len(NAMES)], PERSON),
            (". Disputed charge applied to card ", None),
            (CARDS[i % len(CARDS)], CARD),
            (". Followup contact: ", None),
            (EMAILS[(i + 2) % len(EMAILS)], EMAIL),
            (".", None),
        ]
    )
    return text, ents, []


def support_refund(i: int) -> tuple[str, list[GoldenEntity], list[GoldenQA]]:
    text, ents = _build(
        [
            ("Refund request for ", None),
            (NAMES[i % len(NAMES)], PERSON),
            (". Refund will be returned to card ", None),
            (CARDS[(i + 1) % len(CARDS)], CARD),
            (". Confirmation sent to ", None),
            (EMAILS[(i + 3) % len(EMAILS)], EMAIL),
            (".", None),
        ]
    )
    return text, ents, []


def support_product_complaint(i: int) -> tuple[str, list[GoldenEntity], list[GoldenQA]]:
    text, ents = _build(
        [
            ("Product complaint logged by ", None),
            (NAMES[i % len(NAMES)], PERSON),
            (". Customer reachable at ", None),
            (PHONES[i % len(PHONES)], PHONE),
            (" or ", None),
            (EMAILS[(i + 4) % len(EMAILS)], EMAIL),
            (".", None),
        ]
    )
    return text, ents, []


def support_shipping(i: int) -> tuple[str, list[GoldenEntity], list[GoldenQA]]:
    text, ents = _build(
        [
            ("Shipping inquiry for ", None),
            (NAMES[i % len(NAMES)], PERSON),
            (". Tracking call placed to ", None),
            (PHONES[(i + 1) % len(PHONES)], PHONE),
            (". Replies go to ", None),
            (EMAILS[(i + 5) % len(EMAILS)], EMAIL),
            (".", None),
        ]
    )
    return text, ents, []


def support_password_reset(i: int) -> tuple[str, list[GoldenEntity], list[GoldenQA]]:
    text, ents = _build(
        [
            ("Password reset link sent to ", None),
            (NAMES[i % len(NAMES)], PERSON),
            (" at ", None),
            (EMAILS[i % len(EMAILS)], EMAIL),
            (". Original request came from ", None),
            (IPS[(i + 1) % len(IPS)], IP),
            (".", None),
        ]
    )
    return text, ents, []


def support_cancellation(i: int) -> tuple[str, list[GoldenEntity], list[GoldenQA]]:
    text, ents = _build(
        [
            ("Subscription cancellation requested by ", None),
            (NAMES[i % len(NAMES)], PERSON),
            (". Card on file was ", None),
            (CARDS[(i + 4) % len(CARDS)], CARD),
            (". Confirmation: ", None),
            (EMAILS[(i + 6) % len(EMAILS)], EMAIL),
            (".", None),
        ]
    )
    return text, ents, []


def support_feature_request(i: int) -> tuple[str, list[GoldenEntity], list[GoldenQA]]:
    text, ents = _build(
        [
            ("Feature request from ", None),
            (NAMES[i % len(NAMES)], PERSON),
            (". Replies will go to ", None),
            (EMAILS[(i + 7) % len(EMAILS)], EMAIL),
            (". Account region matches IP ", None),
            (IPS[(i + 2) % len(IPS)], IP),
            (".", None),
        ]
    )
    return text, ents, []


def support_bug_report(i: int) -> tuple[str, list[GoldenEntity], list[GoldenQA]]:
    text, ents = _build(
        [
            ("Bug report filed by ", None),
            (NAMES[i % len(NAMES)], PERSON),
            (". Crash recorded from IP ", None),
            (IPS[(i + 3) % len(IPS)], IP),
            (". Reach back at ", None),
            (PHONES[(i + 2) % len(PHONES)], PHONE),
            (".", None),
        ]
    )
    return text, ents, []


def support_verification(i: int) -> tuple[str, list[GoldenEntity], list[GoldenQA]]:
    text, ents = _build(
        [
            ("Account verification requested for ", None),
            (NAMES[i % len(NAMES)], PERSON),
            (". Verification code sent to ", None),
            (EMAILS[(i + 8) % len(EMAILS)], EMAIL),
            (". Backup phone: ", None),
            (PHONES[(i + 5) % len(PHONES)], PHONE),
            (".", None),
        ]
    )
    return text, ents, []


# ── Master template list ────────────────────────────────────────────────────
HR_TEMPLATES: list[Callable[[int], tuple[str, list[GoldenEntity], list[GoldenQA]]]] = [
    hr_onboarding,
    hr_performance_review,
    hr_salary_adjustment,
    hr_leave_request,
    hr_disciplinary_notice,
    hr_promotion,
    hr_termination,
    hr_benefits,
    hr_training,
    hr_transfer,
]

CONTRACT_TEMPLATES: list[Callable[[int], tuple[str, list[GoldenEntity], list[GoldenQA]]]] = [
    contract_service_agreement,
    contract_nda,
    contract_sales,
    contract_lease,
    contract_vendor_onboarding,
    contract_consulting,
    contract_partnership,
    contract_licensing,
    contract_supplier,
    contract_subcontract,
]

SUPPORT_TEMPLATES: list[Callable[[int], tuple[str, list[GoldenEntity], list[GoldenQA]]]] = [
    support_login,
    support_billing,
    support_refund,
    support_product_complaint,
    support_shipping,
    support_password_reset,
    support_cancellation,
    support_feature_request,
    support_bug_report,
    support_verification,
]

CATEGORIES = {
    "hr": HR_TEMPLATES,
    "contracts": CONTRACT_TEMPLATES,
    "support": SUPPORT_TEMPLATES,
}


def generate() -> list[GoldenDocument]:
    docs: list[GoldenDocument] = []
    for category, templates in CATEGORIES.items():
        for i, template in enumerate(templates):
            text, entities, qa = template(i)
            docs.append(
                GoldenDocument(
                    document_id=f"{category}-{i + 1:03d}",
                    category=category,
                    text=text,
                    entities=entities,
                    qa_pairs=qa,
                )
            )
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("eval/golden"),
        help="Output directory.",
    )
    args = parser.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    docs = generate()
    for doc in docs:
        path = out / f"{doc.document_id}.json"
        path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
    print(f"wrote {len(docs)} documents to {out}/")


if __name__ == "__main__":
    main()
