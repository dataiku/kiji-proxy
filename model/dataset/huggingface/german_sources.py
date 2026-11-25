"""Convert permissively licensed German PII datasets to Kiji training samples.

Sources (both allow commercial use):

- Wismut/nym-pii-multilingual-data (MIT). Rows carry no language field, so
  German rows are picked with a function-word heuristic. Fine-grained labels
  map 1:1 to Kiji; "street + number" spans are split into STREET/BUILDINGNUM.
- ai4privacy/pii-masking-openpii-1m (CC-BY-4.0, credit "Ai4Privacy / Ai Suisse
  SA"). German rows only. It has a single DATE label; dates with a birth cue
  just before them become DATEOFBIRTH, and dates after an event word
  ("bis zum", "am", "vom") with a year from 1990 on stay unlabelled. Rows with
  any other date are skipped as ambiguous.

Every converter returns the internal training format used by
``labelstudio_format.convert_to_labelstudio``::

    {"text", "privacy_mask": [{"value", "label", "start", "end"}], "language",
     "country", "source"}

or ``None`` when the row should not be used.
"""

from __future__ import annotations

import re

NYM_REPO = "Wismut/nym-pii-multilingual-data"
OPENPII_REPO = "ai4privacy/pii-masking-openpii-1m"

# Wismut/nym-pii-multilingual-data -> Kiji. None = no Kiji label (left as O).
NYM_TO_KIJI: dict[str, str | None] = {
    "GIVEN_NAME": "FIRSTNAME",
    "SURNAME": "SURNAME",
    "EMAIL": "EMAIL",
    "PHONE": "PHONENUMBER",
    "FAX_NUMBER": "PHONENUMBER",
    "USERNAME": "USERNAME",
    "URL": "URL",
    "COMPANY_NAME": "COMPANYNAME",
    "STREET_ADDRESS": "STREET",  # split into STREET + BUILDINGNUM
    "STREET_NAME": "STREET",
    "SECONDARY_ADDRESS": "STREET",
    "BUILDING_NUMBER": "BUILDINGNUM",
    "CITY": "CITY",
    "STATE": "STATE",
    "ZIP_CODE": "ZIP",
    "COUNTRY": "COUNTRY",
    "DATE_OF_BIRTH": "DATEOFBIRTH",
    "AGE": "AGE",
    "SSN": "SSN",
    "GOVERNMENT_ID": "NATIONALID",
    "PASSPORT": "PASSPORTID",
    "DRIVERS_LICENSE": "DRIVERLICENSENUM",
    "TAX_ID": "TAXNUM",
    "LICENSE_PLATE": "LICENSEPLATENUM",
    "IBAN": "IBAN",
    "CREDIT_DEBIT_CARD": "CREDITCARDNUMBER",
    "PASSWORD": "PASSWORD",
    "API_KEY": "SECURITYTOKEN",
    "DATE": None,
    "TIME": None,
    "GENDER": None,
}

# nym labels for sensitive values Kiji has no label for. Rows containing them
# are skipped so the model is not trained to treat e.g. a CVV as non-PII.
NYM_SKIP_ROW_LABELS = {
    "CUSTOMER_ID",
    "EMPLOYEE_ID",
    "ACCOUNT_NUMBER",
    "ROUTING_NUMBER",
    "SWIFT_BIC",
    "CVV",
    "PIN",
    "MEDICAL_RECORD_NUMBER",
    "MAC_ADDRESS",
}

# ai4privacy/pii-masking-openpii-1m -> Kiji. None = no Kiji label (left as O).
OPENPII_TO_KIJI: dict[str, str | None] = {
    "GIVENNAME": "FIRSTNAME",
    "SURNAME": "SURNAME",
    "EMAIL": "EMAIL",
    "TELEPHONENUM": "PHONENUMBER",
    "STREET": "STREET",
    "BUILDINGNUM": "BUILDINGNUM",
    "ZIPCODE": "ZIP",
    "CITY": "CITY",
    "AGE": "AGE",
    "SOCIALNUM": "SSN",
    "TAXNUM": "TAXNUM",
    "DRIVERLICENSENUM": "DRIVERLICENSENUM",
    "IDCARDNUM": "IDCARDNUM",
    "PASSPORTNUM": "PASSPORTID",
    "CREDITCARDNUMBER": "CREDITCARDNUMBER",
    "DATE": None,  # DATEOFBIRTH when preceded by a birth cue, see below
    "TITLE": None,
    "GENDER": None,
    "SEX": None,
}

OPENPII_REGION_TO_COUNTRY = {"DE": "Germany", "AT": "Austria", "CH": "Switzerland"}

# A birth cue directly before a date ("Geburtsdatum: ", "geboren am ").
_BIRTH_CUE = re.compile(
    r"(geb\.|geboren|geburt\w*|jahrgang)\W{0,3}((am|den|ist|lautet|von|vom|:)\W{0,3}){0,2}$",
    re.IGNORECASE,
)
# A birth word somewhere nearby, but not directly before the date: ambiguous.
_BIRTH_WORD = re.compile(r"geb\.|geboren|geburt|jahrgang|\bborn\b|birth", re.IGNORECASE)
# A birth cue directly after a date ("03.04.2001 ... geboren").
_BIRTH_CUE_AFTER = re.compile(
    r"(geb\.|geboren|geburt\w*|zur\s+welt\s+gekommen)", re.IGNORECASE
)
# An event word directly before the date ("bis zum", "am", "vom", "Datum:").
_EVENT_CUE = re.compile(
    r"\b(bis|ab|seit|spätestens|zum|am|vom|den|datum|stand|termin|frist|datiert)\W{0,3}$",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"[.!?]")
_YEAR = re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)")
# openpii fills date slots with random dates, so "bis zum 28/12/1943" is
# common. An event date that old is more likely a misplaced birth date.
_MIN_EVENT_YEAR = 1990


def _is_event_date(before: str, value: str) -> bool:
    """True when an unlabelled openpii DATE can safely stay unlabelled."""
    if not _EVENT_CUE.search(before[-30:]):
        return False
    year = _YEAR.search(value)
    return year is None or int(year.group()) >= _MIN_EVENT_YEAR


_STREET_THEN_NUMBER = re.compile(
    r"^(?P<street>.*\S)\s+(?P<num>\d+\s?[a-zA-Z]?(?:\s?[-/]\s?\d+\s?[a-zA-Z]?)?)$"
)
_NUMBER_THEN_STREET = re.compile(r"^(?P<num>\d+[a-zA-Z]?)\s+(?P<street>\S.*)$")

_DE_WORDS = re.compile(
    r"\b(und|der|die|das|nicht|ist|mit|für|sie|ich|bitte|wir|ihre|ihr|auf|den|dem"
    r"|von|zu|eine|ein|einen|bei|unter|wurde|sind|haben|hat|wird|können|kann"
    r"|meine|mein|zum|zur|ihnen|herr|frau|sehr|geehrte|grüße|werden|auch|oder)\b",
    re.IGNORECASE,
)
_OTHER_LANG_WORDS = [
    re.compile(rf"\b({words})\b", re.IGNORECASE)
    for words in (
        "the|and|is|for|with|please|your|of|to|this|that|are|was|be",  # en
        "het|een|niet|voor|met|wij|uw|zijn|van|naar|ik|je|deze|dat|wordt",  # nl
        "og|ikke|jeg|er|til|af|på|med|det|den|som|har",  # da
        "och|inte|jag|är|till|av|på|med|det|som|har|för",  # sv
        "le|la|les|et|est|pour|avec|vous|nous|une|des|du|au",  # fr
        "il|la|che|è|per|con|non|una|sono|di|del|della",  # it
        "el|la|que|es|para|con|por|una|los|las|del",  # es
        "o|que|não|para|com|uma|os|do|da|em|é",  # pt
    )
]


def looks_german(text: str) -> bool:
    """Function-word heuristic (same as the nym-de benchmark loader)."""
    de_words = [w.lower() for w in _DE_WORDS.findall(text)]
    other = max(len(p.findall(text)) for p in _OTHER_LANG_WORDS)
    return len(set(de_words)) >= 2 and len(de_words) >= 3 and len(de_words) > 2 * other


def split_street_address(text: str, start: int, end: int) -> list[tuple[int, int, str]]:
    """Split a "street + house number" span into STREET and BUILDINGNUM."""
    value = text[start:end]
    for pattern in (_STREET_THEN_NUMBER, _NUMBER_THEN_STREET):
        m = pattern.match(value)
        if m:
            return sorted(
                [
                    (start + m.start("street"), start + m.end("street"), "STREET"),
                    (start + m.start("num"), start + m.end("num"), "BUILDINGNUM"),
                ]
            )
    return [(start, end, "STREET")]


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _to_sample(
    text: str,
    spans: list[tuple[int, int, str]],
    country: str | None,
    source: str,
) -> dict | None:
    mask = []
    last_end = -1
    for start, end, label in sorted(spans):
        start, end = _trim(text, start, end)
        if start >= end:
            continue
        if start < last_end:
            # Overlapping gold spans: the row is unreliable.
            return None
        mask.append(
            {"value": text[start:end], "label": label, "start": start, "end": end}
        )
        last_end = end
    if not mask:
        return None
    return {
        "text": text,
        "privacy_mask": mask,
        "coreferences": [],
        "language": "German",
        "country": country,
        "source": source,
    }


def convert_nym_row(row: dict) -> dict | None:
    """German nym row -> sample, or None (not German, skipped label, no PII)."""
    text = row["text"]
    if not looks_german(text):
        return None
    spans: list[tuple[int, int, str]] = []
    for ent in row["entities"]:
        label = ent["label"]
        if label in NYM_SKIP_ROW_LABELS:
            return None
        kiji = NYM_TO_KIJI.get(label)
        if kiji is None:
            if label not in NYM_TO_KIJI:
                return None  # unknown label: skip rather than guess
            continue
        start, end = ent["start"], ent["end"]
        if label == "STREET_ADDRESS":
            spans.extend(split_street_address(text, start, end))
        else:
            spans.append((start, end, kiji))
    return _to_sample(text, spans, None, NYM_REPO)


def convert_openpii_row(row: dict) -> dict | None:
    """German openpii-1m row -> sample, or None."""
    if row.get("language") != "de":
        return None
    text = row["source_text"]
    spans: list[tuple[int, int, str]] = []
    prev_end = 0
    for ent in sorted(row["privacy_mask"], key=lambda e: e["start"]):
        label = ent["label"]
        start, end = ent["start"], ent["end"]
        if text[start:end] != ent["value"]:
            return None
        window_start, prev_end = max(prev_end, start - 60), max(prev_end, end)
        if label == "DATE":
            # Only text after the previous entity counts as this date's context.
            before = text[max(0, window_start) : start]
            after = text[end : min(len(text), end + 40)]
            sentence_end = _SENTENCE_END.search(after)
            if sentence_end:
                after = after[: sentence_end.start()]
            if _BIRTH_CUE.search(before[-35:]) or _BIRTH_CUE_AFTER.search(after):
                spans.append((start, end, "DATEOFBIRTH"))
            elif _BIRTH_WORD.search(before) or not _is_event_date(before, ent["value"]):
                # May be a birth date (birth word nearby, no event word, or an
                # implausibly old event date): skip the row rather than teach
                # that a date without a birth word is never PII.
                return None
            continue
        if label not in OPENPII_TO_KIJI:
            return None
        kiji = OPENPII_TO_KIJI[label]
        if kiji is not None:
            spans.append((start, end, kiji))
    return _to_sample(
        text, spans, OPENPII_REGION_TO_COUNTRY.get(row.get("region")), OPENPII_REPO
    )
