#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Generate synthetic contact-card / resume / email-signature training samples.

Targets a distribution gap: the existing training corpus has near-zero coverage
of structured multi-line layouts (resume headers, email signatures, business
cards, form fields). Bidirectional encoders propagate that OOD context back to
otherwise easy tokens (e.g. a leading "John Smith"), which causes the smoke
test in `tests/benchmark/smoke_test.py` to drop FIRSTNAME/SURNAME on the
fourth case.

Output is Label Studio JSON, one file per sample, written to the same
training_samples directory the AI4Privacy import uses — so the next training
run picks them up automatically.

Usage:
    uv run python model/dataset/synthetic/generate_contact_cards.py
    uv run python model/dataset/synthetic/generate_contact_cards.py --num 2000
    uv run python model/dataset/synthetic/generate_contact_cards.py --output-dir /tmp/test --num 5
"""

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.dataset.labelstudio.labelstudio_format import (  # noqa: E402
    convert_to_labelstudio,
)

# ---------------------------------------------------------------------------
# Pools — chosen to be diverse and to avoid mutual-substring collisions
# (the Label Studio converter does first-occurrence substring matching).
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "John",
    "Maria",
    "Liam",
    "Aisha",
    "Kenji",
    "Ananya",
    "Marcus",
    "Priya",
    "Elena",
    "Jamal",
    "Wei",
    "Sofia",
    "Diego",
    "Fatima",
    "Aroha",
    "Samir",
    "Olivia",
    "Hiroshi",
    "Naomi",
    "Ethan",
    "Chen",
    "Lucia",
    "Mateo",
    "Zara",
    "Noah",
    "Yuki",
    "Imani",
    "Tomas",
    "Anika",
    "Rashid",
    "Camille",
    "Rohan",
    "Esme",
    "Tariq",
    "Beatriz",
    "Felix",
    "Saoirse",
    "Dmitri",
    "Naledi",
    "Arjun",
    "Sienna",
    "Hassan",
    "Mira",
    "Ezra",
    "Lina",
    "Pablo",
    "Amara",
    "Ivan",
    "Jin",
    "Cleo",
    "Nadia",
    "Owen",
    "Mei",
    "Theo",
    "Hana",
    "Idris",
    "Kira",
    "Luca",
    "Ren",
    "Selma",
    "Vikram",
]

SURNAMES = [
    "Smith",
    "Gonzalez",
    "OBrien",
    "Hassan",
    "Watanabe",
    "Iyer",
    "Johnson",
    "Patel",
    "Petrova",
    "Williams",
    "Nguyen",
    "Garcia",
    "Khan",
    "Lopez",
    "Andersen",
    "Kowalski",
    "Tanaka",
    "Diallo",
    "Rossi",
    "Park",
    "Mendoza",
    "Eriksen",
    "Bauer",
    "Singh",
    "Chowdhury",
    "Silva",
    "Mwangi",
    "Hoffman",
    "Akande",
    "Yamamoto",
    "Cohen",
    "Marchetti",
    "Vargas",
    "Holm",
    "Banerjee",
    "Hernandez",
    "Okafor",
    "Lindgren",
    "Cruz",
    "Sato",
    "Klein",
    "Romero",
    "Ahmed",
    "Schneider",
    "Santos",
    "Petersen",
    "Dubois",
    "Reyes",
    "Walsh",
    "Karimov",
    "Olsen",
    "Ferreira",
    "Nakamura",
    "Kowalczyk",
    "MacIntyre",
    "Boateng",
    "Costa",
    "Vasquez",
    "Engelbert",
    "Kobayashi",
]

# (city, state, zip) — all mutually non-substring within each tuple.
CITIES_STATES_ZIPS = [
    ("Boston", "MA", "02110"),
    ("San Diego", "CA", "92101"),
    ("Austin", "TX", "78701"),
    ("Portland", "OR", "97201"),
    ("Asheville", "NC", "28801"),
    ("Madison", "WI", "53703"),
    ("Boise", "ID", "83702"),
    ("Tulsa", "OK", "74103"),
    ("Tacoma", "WA", "98402"),
    ("Albuquerque", "NM", "87102"),
    ("Burlington", "VT", "05401"),
    ("Worcester", "MA", "01608"),
    ("Provo", "UT", "84601"),
    ("Lincoln", "NE", "68508"),
    ("Bozeman", "MT", "59715"),
    ("Savannah", "GA", "31401"),
    ("Anchorage", "AK", "99501"),
    ("Knoxville", "TN", "37902"),
    ("Reno", "NV", "89501"),
    ("Charleston", "SC", "29401"),
    ("Spokane", "WA", "99201"),
    ("Boulder", "CO", "80302"),
    ("Pasadena", "CA", "91101"),
    ("Hartford", "CT", "06103"),
    ("Tallahassee", "FL", "32301"),
    ("Wichita", "KS", "67202"),
    ("Birmingham", "AL", "35203"),
    ("Akron", "OH", "44308"),
    ("Eugene", "OR", "97401"),
    ("Lubbock", "TX", "79401"),
]

INTL_CITIES = [
    ("Berlin", "Germany"),
    ("Lisbon", "Portugal"),
    ("Cape Town", "South Africa"),
    ("Auckland", "New Zealand"),
    ("Helsinki", "Finland"),
    ("Singapore", "Singapore"),
    ("Bangalore", "India"),
    ("Edinburgh", "United Kingdom"),
    ("Montreal", "Canada"),
    ("Buenos Aires", "Argentina"),
    ("Stockholm", "Sweden"),
    ("Kyoto", "Japan"),
    ("Marrakech", "Morocco"),
    ("Reykjavik", "Iceland"),
    ("Krakow", "Poland"),
]

STREETS = [
    "Cranbrook Lane",
    "Linden Avenue",
    "Birch Hollow Road",
    "Sycamore Drive",
    "Magnolia Court",
    "Foxglove Way",
    "Juniper Trail",
    "Catalpa Boulevard",
    "Acacia Mews",
    "Holly Ridge",
    "Cypress Walk",
    "Marigold Crescent",
    "Hawthorn Close",
    "Larkspur Place",
    "Yarrow Path",
    "Spruce Hollow",
    "Wisteria Lane",
    "Cedar Bluff Drive",
    "Tamarack Court",
    "Quince Row",
    "Periwinkle Trail",
    "Goldenrod Way",
    "Heather Glen",
    "Mulberry Park",
    "Ironwood Pass",
    "Bayberry Hill",
    "Driftwood Avenue",
    "Sequoia Ridge",
]

EMAIL_DOMAINS = [
    "protonmail.com",
    "fastmail.com",
    "outlook.com",
    "icloud.com",
    "mailbox.org",
    "tutanota.com",
    "yandex.com",
    "zoho.com",
    "hey.com",
    "duck.com",
    "posteo.net",
    "gmx.com",
]

# Company names — kept distinctive so they don't substring into other tokens.
COMPANIES = [
    "Crescent Robotics",
    "BlueWave Analytics",
    "Northwind Software",
    "Acme Corporation",
    "Polaris Bioscience",
    "Lighthouse Capital",
    "Mosaic Industries",
    "Veridian Systems",
    "Helios Therapeutics",
    "Driftwood Studios",
    "Talon Engineering",
    "Halcyon Networks",
    "Cobalt Strategies",
    "Larkfield Partners",
    "Meridian Logistics",
    "Pinecrest Holdings",
    "Granite Peak Advisors",
    "Solstice Media",
    "Verdant Foods",
    "Riverstone Consulting",
    "Emberglow Design",
    "Marlowe Industries",
    "Bramblewood Press",
    "Quill & Compass",
    "Sable Stream Labs",
    "Tideline Pharmaceuticals",
    "Stonebridge Research",
    "Nimbus Cloud Group",
    "Foxglen Ventures",
    "Aurelia Robotics",
]

TITLES = [
    "Senior Engineer",
    "Product Manager",
    "Marketing Director",
    "Data Scientist",
    "Software Architect",
    "Account Executive",
    "Operations Lead",
    "Research Scientist",
    "VP of Engineering",
    "Head of Sales",
    "Designer",
    "Engineering Manager",
    "Customer Success Manager",
    "Principal Engineer",
    "Founder",
    "Chief of Staff",
    "Solutions Architect",
    "Backend Developer",
    "Frontend Engineer",
    "Site Reliability Engineer",
    "Recruiter",
    "Technical Writer",
    "Program Manager",
    "Consultant",
]

URL_TEMPLATES = [
    "linkedin.com/in/{handle}",
    "github.com/{handle}",
    "twitter.com/{handle}",
    "{handle}.dev",
    "{company_slug}.com",
    "www.{company_slug}.com",
    "https://linkedin.com/in/{handle}",
    "https://github.com/{handle}",
    "https://www.{company_slug}.com",
]

SIGNOFFS = [
    "Best,",
    "Best regards,",
    "Cheers,",
    "Thanks,",
    "Kind regards,",
    "Warmly,",
    "All the best,",
    "Sincerely,",
]


# ---------------------------------------------------------------------------
# Field generators
# ---------------------------------------------------------------------------


def _slug(s: str) -> str:
    return "".join(c.lower() for c in s if c.isalnum())


def _email_for(first: str, last: str, rng: random.Random) -> str:
    fn, ln = first.lower(), _slug(last)
    forms = [
        f"{fn}.{ln}",
        f"{fn[0]}{ln}",
        f"{fn}{ln}",
        f"{fn}_{ln}",
        f"{fn}.{ln}{rng.randint(1, 99)}",
    ]
    return f"{rng.choice(forms)}@{rng.choice(EMAIL_DOMAINS)}"


def _phone_us(rng: random.Random) -> str:
    area = rng.randint(201, 989)
    pre = rng.randint(200, 989)
    line = rng.randint(1000, 9999)
    fmt = rng.choice(
        [
            "({a}) {p}-{l}",
            "{a}-{p}-{l}",
            "+1 {a} {p} {l}",
            "+1-{a}-{p}-{l}",
            "({a}) {p}.{l}",
            "{a}.{p}.{l}",
        ]
    )
    return fmt.format(a=area, p=pre, l=line)


def _phone_intl(rng: random.Random) -> str:
    return rng.choice(
        [
            f"+44 20 {rng.randint(1000, 9999)} {rng.randint(1000, 9999)}",
            f"+49 30 {rng.randint(10000000, 99999999)}",
            f"+33 1 {rng.randint(10, 99)} {rng.randint(10, 99)} {rng.randint(10, 99)} {rng.randint(10, 99)}",
            f"+81 3 {rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}",
            f"+61 2 {rng.randint(1000, 9999)} {rng.randint(1000, 9999)}",
        ]
    )


def _url(first: str, last: str, company: str, rng: random.Random) -> str:
    handle = rng.choice(
        [
            f"{first.lower()}{_slug(last)}",
            f"{first.lower()}-{_slug(last)}",
            f"{_slug(last)}{first.lower()[0]}",
            f"{first.lower()}.{_slug(last)}",
            f"{first.lower()}{rng.randint(10, 999)}",
        ]
    )
    template = rng.choice(URL_TEMPLATES)
    return template.format(handle=handle, company_slug=_slug(company.split()[0]))


# ---------------------------------------------------------------------------
# Templates: each returns (text, privacy_mask). Each value in privacy_mask
# must appear exactly once in `text` and not be a substring of another value.
# A post-build validator drops samples that violate this — so a slightly
# noisy template is fine.
# ---------------------------------------------------------------------------


def _pick_person(rng: random.Random) -> tuple[str, str]:
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(SURNAMES)
    return first, last


def _pick_address(rng: random.Random):
    city, state, zip_ = rng.choice(CITIES_STATES_ZIPS)
    bnum = str(rng.randint(2, 9999))
    street = rng.choice(STREETS)
    return bnum, street, city, state, zip_


def template_resume_header(rng):
    first, last = _pick_person(rng)
    bnum, street, city, state, zip_ = _pick_address(rng)
    email = _email_for(first, last, rng)
    phone = _phone_us(rng)
    company = rng.choice(COMPANIES)
    url = _url(first, last, company, rng)

    text = (
        f"{first} {last}\n"
        f"{bnum} {street}\n"
        f"{city}, {state} {zip_}\n"
        f"Email: {email}\n"
        f"Phone: {phone}\n"
        f"LinkedIn: {url}"
    )
    pm = [
        {"value": first, "label": "FIRSTNAME"},
        {"value": last, "label": "SURNAME"},
        {"value": bnum, "label": "BUILDINGNUM"},
        {"value": street, "label": "STREET"},
        {"value": city, "label": "CITY"},
        {"value": state, "label": "STATE"},
        {"value": zip_, "label": "ZIP"},
        {"value": email, "label": "EMAIL"},
        {"value": phone, "label": "PHONENUMBER"},
        {"value": url, "label": "URL"},
    ]
    return text, pm


def template_resume_header_minimal(rng):
    first, last = _pick_person(rng)
    city, state, zip_ = rng.choice(CITIES_STATES_ZIPS)
    email = _email_for(first, last, rng)
    phone = _phone_us(rng)

    text = f"{first} {last}\n{city}, {state} {zip_}\n{email} | {phone}"
    pm = [
        {"value": first, "label": "FIRSTNAME"},
        {"value": last, "label": "SURNAME"},
        {"value": city, "label": "CITY"},
        {"value": state, "label": "STATE"},
        {"value": zip_, "label": "ZIP"},
        {"value": email, "label": "EMAIL"},
        {"value": phone, "label": "PHONENUMBER"},
    ]
    return text, pm


def template_email_signature(rng):
    first, last = _pick_person(rng)
    company = rng.choice(COMPANIES)
    title = rng.choice(TITLES)
    email = _email_for(first, last, rng)
    phone = _phone_us(rng)
    url = _url(first, last, company, rng)
    signoff = rng.choice(SIGNOFFS)

    text = f"{signoff}\n{first} {last}\n{title} | {company}\n{email} | {phone}\n{url}"
    pm = [
        {"value": first, "label": "FIRSTNAME"},
        {"value": last, "label": "SURNAME"},
        {"value": company, "label": "COMPANYNAME"},
        {"value": email, "label": "EMAIL"},
        {"value": phone, "label": "PHONENUMBER"},
        {"value": url, "label": "URL"},
    ]
    return text, pm


def template_email_signature_compact(rng):
    first, last = _pick_person(rng)
    company = rng.choice(COMPANIES)
    email = _email_for(first, last, rng)
    phone = _phone_us(rng)

    text = f"--\n{first} {last}\n{company}\n{email}\n{phone}"
    pm = [
        {"value": first, "label": "FIRSTNAME"},
        {"value": last, "label": "SURNAME"},
        {"value": company, "label": "COMPANYNAME"},
        {"value": email, "label": "EMAIL"},
        {"value": phone, "label": "PHONENUMBER"},
    ]
    return text, pm


def template_business_card(rng):
    first, last = _pick_person(rng)
    title = rng.choice(TITLES)
    company = rng.choice(COMPANIES)
    bnum, street, city, state, zip_ = _pick_address(rng)
    email = _email_for(first, last, rng)
    phone = _phone_us(rng)
    url = _url(first, last, company, rng)

    text = (
        f"{first} {last}\n"
        f"{title}\n"
        f"{company}\n"
        f"{bnum} {street}\n"
        f"{city}, {state} {zip_}\n"
        f"{email}\n"
        f"{phone}\n"
        f"{url}"
    )
    pm = [
        {"value": first, "label": "FIRSTNAME"},
        {"value": last, "label": "SURNAME"},
        {"value": company, "label": "COMPANYNAME"},
        {"value": bnum, "label": "BUILDINGNUM"},
        {"value": street, "label": "STREET"},
        {"value": city, "label": "CITY"},
        {"value": state, "label": "STATE"},
        {"value": zip_, "label": "ZIP"},
        {"value": email, "label": "EMAIL"},
        {"value": phone, "label": "PHONENUMBER"},
        {"value": url, "label": "URL"},
    ]
    return text, pm


def template_form_fields(rng):
    first, last = _pick_person(rng)
    bnum, street, city, state, zip_ = _pick_address(rng)
    email = _email_for(first, last, rng)
    phone = _phone_us(rng)
    company = rng.choice(COMPANIES)

    text = (
        f"Name: {first} {last}\n"
        f"Address: {bnum} {street}\n"
        f"City: {city}\n"
        f"State: {state}\n"
        f"Zip: {zip_}\n"
        f"Email: {email}\n"
        f"Phone: {phone}\n"
        f"Company: {company}"
    )
    pm = [
        {"value": first, "label": "FIRSTNAME"},
        {"value": last, "label": "SURNAME"},
        {"value": bnum, "label": "BUILDINGNUM"},
        {"value": street, "label": "STREET"},
        {"value": city, "label": "CITY"},
        {"value": state, "label": "STATE"},
        {"value": zip_, "label": "ZIP"},
        {"value": email, "label": "EMAIL"},
        {"value": phone, "label": "PHONENUMBER"},
        {"value": company, "label": "COMPANYNAME"},
    ]
    return text, pm


def template_form_fields_caps(rng):
    first, last = _pick_person(rng)
    company = rng.choice(COMPANIES)
    title = rng.choice(TITLES)
    email = _email_for(first, last, rng)
    phone = _phone_us(rng)

    text = (
        f"NAME: {first} {last}\n"
        f"TITLE: {title}\n"
        f"COMPANY: {company}\n"
        f"EMAIL: {email}\n"
        f"PHONE: {phone}"
    )
    pm = [
        {"value": first, "label": "FIRSTNAME"},
        {"value": last, "label": "SURNAME"},
        {"value": company, "label": "COMPANYNAME"},
        {"value": email, "label": "EMAIL"},
        {"value": phone, "label": "PHONENUMBER"},
    ]
    return text, pm


def template_linkedin_profile(rng):
    first, last = _pick_person(rng)
    title = rng.choice(TITLES)
    company = rng.choice(COMPANIES)
    city, state, _ = rng.choice(CITIES_STATES_ZIPS)
    url = _url(first, last, company, rng)
    email = _email_for(first, last, rng)

    text = (
        f"{first} {last}\n"
        f"{title} at {company}\n"
        f"{city}, {state}\n"
        f"Contact: {email}\n"
        f"Profile: {url}"
    )
    pm = [
        {"value": first, "label": "FIRSTNAME"},
        {"value": last, "label": "SURNAME"},
        {"value": company, "label": "COMPANYNAME"},
        {"value": city, "label": "CITY"},
        {"value": state, "label": "STATE"},
        {"value": email, "label": "EMAIL"},
        {"value": url, "label": "URL"},
    ]
    return text, pm


def template_signature_intl(rng):
    first, last = _pick_person(rng)
    title = rng.choice(TITLES)
    company = rng.choice(COMPANIES)
    city, country = rng.choice(INTL_CITIES)
    email = _email_for(first, last, rng)
    phone = _phone_intl(rng)
    url = _url(first, last, company, rng)
    signoff = rng.choice(SIGNOFFS)

    text = (
        f"{signoff}\n"
        f"{first} {last}\n"
        f"{title}\n"
        f"{company}\n"
        f"{city}, {country}\n"
        f"{email} • {phone}\n"
        f"{url}"
    )
    pm = [
        {"value": first, "label": "FIRSTNAME"},
        {"value": last, "label": "SURNAME"},
        {"value": company, "label": "COMPANYNAME"},
        {"value": city, "label": "CITY"},
        {"value": country, "label": "COUNTRY"},
        {"value": email, "label": "EMAIL"},
        {"value": phone, "label": "PHONENUMBER"},
        {"value": url, "label": "URL"},
    ]
    return text, pm


def template_directory_listing(rng):
    first, last = _pick_person(rng)
    title = rng.choice(TITLES)
    company = rng.choice(COMPANIES)
    email = _email_for(first, last, rng)
    phone = _phone_us(rng)

    text = f"{last}, {first}\n{title}, {company}\n{email}\n{phone}"
    pm = [
        {"value": first, "label": "FIRSTNAME"},
        {"value": last, "label": "SURNAME"},
        {"value": company, "label": "COMPANYNAME"},
        {"value": email, "label": "EMAIL"},
        {"value": phone, "label": "PHONENUMBER"},
    ]
    return text, pm


def template_intake_form(rng):
    first, last = _pick_person(rng)
    bnum, street, city, state, zip_ = _pick_address(rng)
    email = _email_for(first, last, rng)
    phone = _phone_us(rng)
    dob = f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/{rng.randint(1950, 2005)}"

    text = (
        f"Patient Information\n"
        f"Name: {first} {last}\n"
        f"DOB: {dob}\n"
        f"Address: {bnum} {street}\n"
        f"{city}, {state} {zip_}\n"
        f"Phone: {phone}\n"
        f"Email: {email}"
    )
    pm = [
        {"value": first, "label": "FIRSTNAME"},
        {"value": last, "label": "SURNAME"},
        {"value": dob, "label": "DATEOFBIRTH"},
        {"value": bnum, "label": "BUILDINGNUM"},
        {"value": street, "label": "STREET"},
        {"value": city, "label": "CITY"},
        {"value": state, "label": "STATE"},
        {"value": zip_, "label": "ZIP"},
        {"value": phone, "label": "PHONENUMBER"},
        {"value": email, "label": "EMAIL"},
    ]
    return text, pm


def template_short_bio_block(rng):
    first, last = _pick_person(rng)
    title = rng.choice(TITLES)
    company = rng.choice(COMPANIES)
    city, state, _ = rng.choice(CITIES_STATES_ZIPS)
    url = _url(first, last, company, rng)

    text = f"{first} {last}\n{title}\n{company} — {city}, {state}\n{url}"
    pm = [
        {"value": first, "label": "FIRSTNAME"},
        {"value": last, "label": "SURNAME"},
        {"value": company, "label": "COMPANYNAME"},
        {"value": city, "label": "CITY"},
        {"value": state, "label": "STATE"},
        {"value": url, "label": "URL"},
    ]
    return text, pm


def template_just_name_then_address(rng):
    """Bare 'name on its own line' + minimal trailing context — the hardest
    pattern from the smoke test."""
    first, last = _pick_person(rng)
    bnum, street, city, state, zip_ = _pick_address(rng)

    text = f"{first} {last}\n{bnum} {street}\n{city}, {state} {zip_}"
    pm = [
        {"value": first, "label": "FIRSTNAME"},
        {"value": last, "label": "SURNAME"},
        {"value": bnum, "label": "BUILDINGNUM"},
        {"value": street, "label": "STREET"},
        {"value": city, "label": "CITY"},
        {"value": state, "label": "STATE"},
        {"value": zip_, "label": "ZIP"},
    ]
    return text, pm


TEMPLATES = [
    template_resume_header,
    template_resume_header_minimal,
    template_email_signature,
    template_email_signature_compact,
    template_business_card,
    template_form_fields,
    template_form_fields_caps,
    template_linkedin_profile,
    template_signature_intl,
    template_directory_listing,
    template_intake_form,
    template_short_bio_block,
    template_just_name_then_address,
]


# ---------------------------------------------------------------------------
# Validation: drop samples whose values aren't cleanly findable in the text
# ---------------------------------------------------------------------------


def _is_valid(text: str, privacy_mask: list[dict]) -> bool:
    values = [item["value"] for item in privacy_mask]
    # Each value must appear at least once
    for v in values:
        if v not in text:
            return False
    # No value can be a substring of another (would cause first-match collisions)
    for i, v1 in enumerate(values):
        for j, v2 in enumerate(values):
            if i != j and v1 != v2 and v1 in v2:
                return False
    return True


def _validate_annotations(ls_sample: dict) -> bool:
    text = ls_sample["data"]["text"]
    results = ls_sample["predictions"][0]["result"]
    seen_labels = set()
    for ann in results:
        if ann.get("type") != "labels":
            continue
        v = ann["value"]
        if not v.get("labels"):
            continue
        start, end = v["start"], v["end"]
        if text[start:end] != v["text"]:
            return False
        seen_labels.update(v["labels"])
    # Sanity: the sample must contain at least one entity
    return bool(seen_labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate_samples(
    num: int,
    output_dir: str,
    seed: int = 42,
    language: str = "English",
    country: str = "United States",
) -> int:
    rng = random.Random(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written = 0
    attempts = 0
    max_attempts = num * 5

    while written < num and attempts < max_attempts:
        attempts += 1
        template = rng.choice(TEMPLATES)
        text, privacy_mask = template(rng)

        if not _is_valid(text, privacy_mask):
            continue

        sample = {
            "text": text,
            "privacy_mask": privacy_mask,
            "language": language,
            "country": country,
        }
        ls = convert_to_labelstudio(sample)

        if not _validate_annotations(ls):
            continue

        text_hash = hashlib.sha256(text.encode()).hexdigest()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = f"{timestamp}_synth_{text_hash}.json"
        ls["data"]["file_name"] = file_name

        out_file = output_path / file_name
        if out_file.exists():
            # Hash collision (same text generated twice) — skip without counting
            continue

        with out_file.open("w", encoding="utf-8") as f:
            json.dump(ls, f, indent=2, ensure_ascii=False)
        written += 1

    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic contact-card / resume / signature training samples"
    )
    parser.add_argument(
        "--num", type=int, default=1000, help="Number of samples to generate"
    )
    parser.add_argument(
        "--output-dir",
        default="model/dataset/data_samples/training_samples",
        help="Directory to write JSON files (default: training_samples)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--language", default="English")
    parser.add_argument("--country", default="United States")
    args = parser.parse_args()

    written = generate_samples(
        num=args.num,
        output_dir=args.output_dir,
        seed=args.seed,
        language=args.language,
        country=args.country,
    )
    print(f"Wrote {written} samples to {args.output_dir}")
    print("Ready for training with: uv run python model/flows/training_pipeline.py run")
