"""Presidio Stage-1 detector tests.

These tests do exercise the real Presidio recognisers (pattern matchers,
no NLP engine). They run offline and never touch the network. If Presidio's
default thresholds change upstream, the assertions stay valid because we
check entity types rather than raw scores.
"""

from __future__ import annotations

from guardian_rag.pii.models import PiiEntityType, PiiStage
from guardian_rag.pii.presidio_engine import PresidioEngine


def test_email_detection():
    engine = PresidioEngine()
    text = "Please email john.smith@example.com for details."
    entities = engine.detect(text)
    emails = [e for e in entities if e.entity_type is PiiEntityType.EMAIL]
    assert len(emails) == 1
    assert emails[0].text == "john.smith@example.com"
    assert emails[0].stage is PiiStage.PRESIDIO


def test_iban_detection():
    engine = PresidioEngine()
    text = "Transfer to BE71096123456769 by Friday."
    entities = engine.detect(text)
    ibans = [e for e in entities if e.entity_type is PiiEntityType.BANK_ACCOUNT]
    assert len(ibans) == 1
    assert "BE71096123456769" in ibans[0].text


def test_credit_card_detection():
    engine = PresidioEngine()
    text = "Card 4111-1111-1111-1111 expires soon."
    entities = engine.detect(text)
    cards = [e for e in entities if e.entity_type is PiiEntityType.CREDIT_CARD]
    assert len(cards) >= 1


def test_ip_address_detection():
    engine = PresidioEngine()
    text = "Origin IP 192.168.1.42 reported at 09:00."
    entities = engine.detect(text)
    ips = [e for e in entities if e.entity_type is PiiEntityType.IP_ADDRESS]
    assert len(ips) >= 1
    assert ips[0].text == "192.168.1.42"


def test_empty_text_returns_empty():
    assert PresidioEngine().detect("") == []


def test_entities_returned_in_source_order():
    engine = PresidioEngine()
    text = "Send to alice@ex.com from 10.0.0.1 by Friday."
    entities = engine.detect(text)
    starts = [e.start for e in entities]
    assert starts == sorted(starts)
