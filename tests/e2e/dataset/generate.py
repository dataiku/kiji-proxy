"""Generate a labeled PII evaluation dataset for the e2e suite.

Run: uv run python tests/e2e/dataset/generate.py

The output `samples.jsonl` is committed as the stable evaluation baseline.
Values are synthesized via Faker + custom generators under a fixed seed, so
regenerating with the same Faker version produces identical output.

This dataset is intentionally separate from `model/dataset/data_samples/`
(the training corpus) — it gives the detector unseen inputs.
"""

from __future__ import annotations

import json
import random
import re
import string
from pathlib import Path

from faker import Faker

SEED = 42
OUT_NUM = 750
OUT_PATH = Path(__file__).parent / "samples.jsonl"

LABELS = [
    "SURNAME",
    "FIRSTNAME",
    "BUILDINGNUM",
    "DATEOFBIRTH",
    "EMAIL",
    "PHONENUMBER",
    "CITY",
    "URL",
    "COMPANYNAME",
    "STATE",
    "ZIP",
    "STREET",
    "COUNTRY",
    "SSN",
    "DRIVERLICENSENUM",
    "PASSPORTID",
    "NATIONALID",
    "IDCARDNUM",
    "TAXNUM",
    "LICENSEPLATENUM",
    "PASSWORD",
    "IBAN",
    "AGE",
    "SECURITYTOKEN",
    "CREDITCARDNUMBER",
    "USERNAME",
]

faker = Faker("en_US")


def _alnum(rand: random.Random, n: int) -> str:
    return "".join(rand.choice(string.ascii_uppercase + string.digits) for _ in range(n))


GENERATORS = {
    "SURNAME": lambda rand: faker.last_name(),
    "FIRSTNAME": lambda rand: faker.first_name(),
    "BUILDINGNUM": lambda rand: str(rand.randint(1, 9999)),
    "DATEOFBIRTH": lambda rand: faker.date_of_birth(minimum_age=18, maximum_age=85).strftime("%Y-%m-%d"),
    "EMAIL": lambda rand: faker.email(),
    "PHONENUMBER": lambda rand: faker.phone_number(),
    "CITY": lambda rand: faker.city(),
    "URL": lambda rand: faker.url(),
    "COMPANYNAME": lambda rand: faker.company(),
    "STATE": lambda rand: faker.state(),
    "ZIP": lambda rand: faker.zipcode(),
    "STREET": lambda rand: faker.street_name(),
    "COUNTRY": lambda rand: faker.country(),
    "SSN": lambda rand: faker.ssn(),
    "DRIVERLICENSENUM": lambda rand: f"D{rand.randint(1000000, 9999999)}",
    "PASSPORTID": lambda rand: f"{rand.choice(string.ascii_uppercase)}{rand.randint(10000000, 99999999)}",
    "NATIONALID": lambda rand: f"{rand.randint(100, 999)}-{rand.randint(10, 99)}-{rand.randint(1000, 9999)}",
    "IDCARDNUM": lambda rand: f"ID-{_alnum(rand, 8)}",
    "TAXNUM": lambda rand: f"{rand.randint(10, 99)}-{rand.randint(1000000, 9999999)}",
    "LICENSEPLATENUM": lambda rand: faker.license_plate(),
    "PASSWORD": lambda rand: faker.password(length=rand.randint(10, 16)),
    "IBAN": lambda rand: faker.iban(),
    "AGE": lambda rand: str(rand.randint(1, 99)),
    "SECURITYTOKEN": lambda rand: _alnum(rand, rand.randint(24, 40)),
    "CREDITCARDNUMBER": lambda rand: faker.credit_card_number(),
    "USERNAME": lambda rand: faker.user_name(),
}

TEMPLATES = [
    "Please email {EMAIL} about the {COMPANYNAME} contract signed by {FIRSTNAME} {SURNAME} on {DATEOFBIRTH}.",
    "Please contact {FIRSTNAME} {SURNAME} at {PHONENUMBER} or {EMAIL} for the {COMPANYNAME} account.",
    "The applicant {FIRSTNAME} {SURNAME}, age {AGE}, was born on {DATEOFBIRTH} in {CITY}, {COUNTRY}.",
    "Please update {FIRSTNAME} {SURNAME} address to {BUILDINGNUM} {STREET}, {CITY}, {STATE} {ZIP}.",
    "{FIRSTNAME} {SURNAME} can be reached at {EMAIL}; profile: {URL}.",
    "The SSN on file for {FIRSTNAME} is {SSN}; tax ID: {TAXNUM}.",
    "Driver license {DRIVERLICENSENUM} belongs to {FIRSTNAME} {SURNAME}.",
    "Passport number {PASSPORTID} issued to {FIRSTNAME} {SURNAME} from {COUNTRY}.",
    "National ID {NATIONALID} registered in {STATE}.",
    "Employee ID card {IDCARDNUM} grants access to {COMPANYNAME}.",
    "License plate {LICENSEPLATENUM} is registered to {FIRSTNAME} {SURNAME} at {BUILDINGNUM} {STREET}.",
    "The IBAN for {COMPANYNAME} is {IBAN}.",
    "Please bill card {CREDITCARDNUMBER} for the order placed by {FIRSTNAME} {SURNAME}.",
    "Wire transfer to IBAN {IBAN} referencing invoice from {COMPANYNAME}.",
    "Tax number {TAXNUM} appears on the {COMPANYNAME} filing.",
    "Login as {USERNAME} with password {PASSWORD}; docs at {URL}.",
    "API token {SECURITYTOKEN} was issued to user {USERNAME}.",
    "Reset password {PASSWORD} for {USERNAME} registered to {EMAIL}.",
    "Please add {EMAIL} to the mailing list for {COMPANYNAME} updates.",
    "Visit {URL} to activate the account for {USERNAME}.",
    "Ship to {FIRSTNAME} {SURNAME}, {BUILDINGNUM} {STREET}, {CITY}, {STATE} {ZIP}, {COUNTRY}.",
    "The headquarters of {COMPANYNAME} is at {BUILDINGNUM} {STREET}, {CITY}, {STATE} {ZIP}.",
    "Call our {CITY} office at {PHONENUMBER} for questions about {COMPANYNAME}.",
    "Deliveries are handled from {BUILDINGNUM} {STREET} in {CITY}, {STATE}.",
    "Dr. {SURNAME} ({EMAIL}) owns license plate {LICENSEPLATENUM}.",
    "{FIRSTNAME} date of birth is {DATEOFBIRTH} and SSN is {SSN}.",
    "Please verify identity for {FIRSTNAME} {SURNAME}, passport {PASSPORTID}.",
    "New hire {FIRSTNAME} {SURNAME}, age {AGE}, starts at {COMPANYNAME} next month.",
    "Customer {USERNAME} (email {EMAIL}) reports an issue on card {CREDITCARDNUMBER}.",
    "Security token {SECURITYTOKEN} was issued to {USERNAME} at {EMAIL}.",
]

SLOT_RE = re.compile(r"\{([A-Z]+)\}")


def generate_one(template: str, rand: random.Random) -> dict:
    # Resolve one value per distinct label, in sorted order so iteration is deterministic.
    labels_in_template = SLOT_RE.findall(template)
    values = {lbl: GENERATORS[lbl](rand) for lbl in sorted(set(labels_in_template))}

    parts: list[str] = []
    entities: list[dict] = []
    cursor = 0
    for match in SLOT_RE.finditer(template):
        parts.append(template[cursor : match.start()])
        label = match.group(1)
        value = values[label]
        start = sum(len(p) for p in parts)
        parts.append(value)
        end = start + len(value)
        entities.append({"start": start, "end": end, "label": label, "text": value})
        cursor = match.end()
    parts.append(template[cursor:])
    return {"text": "".join(parts), "entities": entities}


def main() -> None:
    random.seed(SEED)
    Faker.seed(SEED)
    rand = random.Random(SEED)

    samples: list[dict] = []
    for i in range(OUT_NUM):
        template = rand.choice(TEMPLATES)
        sample = generate_one(template, rand)
        sample["id"] = f"s{i:04d}"
        samples.append(sample)

    label_counts: dict[str, int] = {}
    for s in samples:
        for e in s["entities"]:
            label_counts[e["label"]] = label_counts.get(e["label"], 0) + 1
    missing = [lbl for lbl in LABELS if label_counts.get(lbl, 0) < 5]
    if missing:
        print(f"WARNING: labels with <5 occurrences: {missing}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False))
            f.write("\n")
    print(f"Wrote {len(samples)} samples to {OUT_PATH}")
    print("Label coverage:")
    for lbl in LABELS:
        print(f"  {lbl}: {label_counts.get(lbl, 0)}")


if __name__ == "__main__":
    main()
