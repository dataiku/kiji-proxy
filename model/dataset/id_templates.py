"""English (UK and US) template rows with personal ID numbers in context.

The Kiji English data labels IDs by the cue word in front of them ("national
ID", "tax number", "ID card number"), but its values are mostly placeholders
(987654321, 123-45-6789, AB123456C), and the same format carries different
labels in different rows. The German-mix models therefore confuse ID types
(a driving licence called a passport) and miss country names outside the six
English-speaking countries of the Kiji data. These rows teach:

- the label from the cue word, with realistic, varied formats: UK National
  Insurance number (NATIONALID), passport, driving licence (derived from the
  surname and birth date like a real DVLA number), Unique Taxpayer Reference
  (TAXNUM); US Social Security number, EIN and ITIN (TAXNUM),
  passport, state driving licence and state ID card (IDCARDNUM);
- that the same shapes are not PII after a non-ID cue: order, case, booking,
  confirmation, e-ticket and form numbers, flight numbers, issue and expiry
  dates; and that a cue word without a number ("bring your passport") labels
  nothing;
- country names from the whole world (COUNTRY), in addresses, travel
  documents and money transfers.

Settings: bank account opening (identity checks), HR onboarding (payroll,
right to work), air travel and hotels, car rental, money transfers, and
key/value records (forms, CSV, JSON). About a third of the rows are PII-free.

Usage:
    from model.dataset.id_templates import generate
    rows = generate(1500, seed=13)
"""

from __future__ import annotations

import random
import string
from collections.abc import Callable
from dataclasses import dataclass

from model.dataset.synthetic_negatives import (
    _PROBE_RE,
    EN_ORDINALS,
    ENGLISH,
    Part,
    _ascii,
    _assemble,
    _Gen,
    _l,
)

COUNTRIES = _l(
    "Afghanistan|Albania|Algeria|Andorra|Angola|Argentina|Armenia|Australia|"
    "Austria|Azerbaijan|Bahrain|Bangladesh|Barbados|Belarus|Belgium|Belize|Benin|"
    "Bhutan|Bolivia|Bosnia and Herzegovina|Botswana|Brazil|Brunei|Bulgaria|"
    "Burkina Faso|Burundi|Cambodia|Cameroon|Canada|Cape Verde|Chad|Chile|China|"
    "Colombia|Costa Rica|Croatia|Cuba|Cyprus|Czech Republic|Denmark|Djibouti|"
    "Dominican Republic|Ecuador|Egypt|El Salvador|Eritrea|Estonia|Eswatini|"
    "Ethiopia|Fiji|Finland|France|Gabon|Gambia|Georgia|Germany|Ghana|Greece|"
    "Guatemala|Guinea|Guyana|Haiti|Honduras|Hungary|Iceland|India|Indonesia|Iran|"
    "Iraq|Ireland|Israel|Italy|Jamaica|Japan|Jordan|Kazakhstan|Kenya|Kuwait|"
    "Kyrgyzstan|Laos|Latvia|Lebanon|Lesotho|Liberia|Libya|Liechtenstein|"
    "Lithuania|Luxembourg|Madagascar|Malawi|Malaysia|Maldives|Mali|Malta|"
    "Mauritania|Mauritius|Mexico|Moldova|Monaco|Mongolia|Montenegro|Morocco|"
    "Mozambique|Myanmar|Namibia|Nepal|Netherlands|New Zealand|Nicaragua|Niger|"
    "Nigeria|North Macedonia|Norway|Oman|Pakistan|Panama|Papua New Guinea|"
    "Paraguay|Peru|Philippines|Poland|Portugal|Qatar|Romania|Rwanda|Saint Lucia|"
    "Samoa|San Marino|Saudi Arabia|Senegal|Serbia|Seychelles|Sierra Leone|"
    "Singapore|Slovakia|Slovenia|Somalia|South Africa|South Korea|Spain|"
    "Sri Lanka|Sudan|Suriname|Sweden|Switzerland|Syria|Taiwan|Tajikistan|"
    "Tanzania|Thailand|Togo|Tonga|Trinidad and Tobago|Tunisia|Turkey|"
    "Turkmenistan|Uganda|Ukraine|United Arab Emirates|Uruguay|Uzbekistan|"
    "Vanuatu|Venezuela|Vietnam|Yemen|Zambia|Zimbabwe"
)
HOME_COUNTRIES = {
    "uk": _l(
        "United Kingdom|United Kingdom|UK|England|Scotland|Wales|Northern Ireland"
    ),
    "us": _l("United States|United States|USA|U.S."),
}

# NI number prefixes: neither letter D, F, I, Q, U or V, second letter not O,
# and never BG, GB, KN, NK, NT, TN or ZZ.
_NI_FIRST = [c for c in string.ascii_uppercase if c not in "DFIQUV"]
_NI_SECOND = [c for c in string.ascii_uppercase if c not in "DFIQUVO"]
_NI_BAD = {"BG", "GB", "KN", "NK", "NT", "TN", "ZZ"}


@dataclass
class Country:
    code: str  # "uk" / "us"
    country: str  # as in the Kiji data
    first_names: list[str]
    surnames: list[str]
    streets: list[str]
    cities: list[tuple[str, str]]  # (ZIP pattern, city) or (state, city) for US
    phone_formats: list[str]
    email_domains: list[str]
    months: list[str]
    # label -> cue phrases written in front of the number
    cues: dict[str, list[str]]
    # key/value field names per label, for forms, CSV and JSON
    keys: dict[str, list[str]]
    currency: str
    tax_forms: list[str]


UK = Country(
    code="uk",
    country="United Kingdom",
    first_names=ENGLISH.first_names,
    surnames=ENGLISH.surnames,
    streets=ENGLISH.streets,
    cities=ENGLISH.cities,
    phone_formats=ENGLISH.phone_formats,
    email_domains=ENGLISH.email_domains,
    months=ENGLISH.months,
    cues={
        "NATIONALID": _l(
            "National Insurance number|NI number|NI no.|National Insurance no.|"
            "NINO|National Insurance (NI) number"
        ),
        "PASSPORTID": _l("passport number|passport no.|UK passport number|passport"),
        "DRIVERLICENSENUM": _l(
            "driving licence number|driving licence no.|licence number|"
            "DVLA licence number|photocard licence number"
        ),
        "TAXNUM": _l(
            "UTR|Unique Taxpayer Reference|UTR number|tax reference (UTR)|"
            "self-assessment UTR"
        ),
    },
    keys={
        "NATIONALID": _l("NI number|NI No.|National Insurance No.|ni_number|nino"),
        "PASSPORTID": _l("Passport No.|Passport number|passport_no|passport"),
        "DRIVERLICENSENUM": _l(
            "Driving licence no.|Licence number|driving_licence|dl_number"
        ),
        "TAXNUM": _l("UTR|Tax reference|utr"),
    },
    currency="£",
    tax_forms=_l("P45|P60|P11D|SA100|SA302|form P46|starter checklist"),
)

US = Country(
    code="us",
    country="United States",
    first_names=_l(
        "James|Mary|Michael|Patricia|Robert|Jennifer|David|Linda|John|Elizabeth|"
        "Carlos|Maria|Jose|Guadalupe|Tyrone|Keisha|DeShawn|Aaliyah|Kevin|Ashley|"
        "Brandon|Brittany|Tyler|Madison|Nguyen|Mai|Hiroshi|Yuki|Rajesh|Ananya|"
        "Omar|Fatima|Dmitri|Svetlana|Cody|Kayla|Hunter|Savannah|Luis|Ximena|"
        "Jamal|Imani|Connor|Molly|Ethan|Olivia|Noah|Emma"
    ),
    surnames=_l(
        "Johnson|Williams|Brown|Garcia|Miller|Davis|Rodriguez|Martinez|Hernandez|"
        "Lopez|Gonzalez|Anderson|Thomas|Moore|Jackson|Martin|Lee|Thompson|White|"
        "Harris|Clark|Lewis|Robinson|Walker|Young|Allen|Ramirez|Nguyen|Kim|Patel|"
        "Washington|Jefferson|O'Connor|McAllister|Kowalski|Schneider|Yamamoto|"
        "Okonkwo|Delgado|Fitzpatrick|Van der Berg|Abernathy|Sullivan|Rivera"
    ),
    streets=_l(
        "Main Street|Oak Avenue|Maple Drive|Elm Street|Washington Avenue|"
        "Lakeview Drive|Pine Street|Cedar Lane|Sunset Boulevard|Park Place|"
        "Highland Avenue|Willow Court|River Road|Hillcrest Drive|Jefferson Street|"
        "Lincoln Way|Magnolia Lane|Ridge Road"
    ),
    cities=[
        ("TX", "Austin"),
        ("TX", "Houston"),
        ("CA", "Sacramento"),
        ("CA", "San Diego"),
        ("NY", "Albany"),
        ("NY", "Buffalo"),
        ("FL", "Tampa"),
        ("FL", "Orlando"),
        ("IL", "Springfield"),
        ("IL", "Naperville"),
        ("WA", "Spokane"),
        ("OH", "Columbus"),
        ("GA", "Savannah"),
        ("AZ", "Tucson"),
        ("CO", "Boulder"),
        ("PA", "Pittsburgh"),
        ("MI", "Ann Arbor"),
        ("NC", "Raleigh"),
        ("MN", "Duluth"),
        ("OR", "Eugene"),
    ],
    phone_formats=_l(
        "({D3}) {D3}-{d4}|{D3}-{D3}-{d4}|+1 {D3} {D3} {d4}|{D3}.{D3}.{d4}"
    ),
    email_domains=_l("gmail.com|yahoo.com|outlook.com|icloud.com|aol.com|example.com"),
    months=ENGLISH.months,
    cues={
        "SSN": _l(
            "Social Security number|SSN|social security no.|SS#|Social Security "
            "Number (SSN)"
        ),
        "TAXNUM": _l(
            "EIN|Employer Identification Number|federal tax ID (EIN)|ITIN|"
            "Individual Taxpayer Identification Number"
        ),
        "PASSPORTID": _l(
            "passport number|U.S. passport no.|passport book number|passport"
        ),
        "DRIVERLICENSENUM": _l(
            "driver's license number|driver license no.|DL #|license number|"
            "driver's license"
        ),
        "IDCARDNUM": _l("state ID number|state ID card number|ID card no."),
    },
    keys={
        "SSN": _l("SSN|Social Security No.|ssn|social_security_number"),
        "TAXNUM": _l("Tax ID|TIN|tax_id"),
        "PASSPORTID": _l("Passport No.|Passport number|passport_no"),
        "DRIVERLICENSENUM": _l(
            "Driver's License #|DL Number|dl_number|drivers_license"
        ),
        "IDCARDNUM": _l("State ID #|ID card number|state_id"),
    },
    currency="$",
    tax_forms=_l("W-4|W-9|I-9|1099-NEC|W-2|form 1040|Form SS-5"),
)
COUNTRIES_BY_CODE = {"uk": UK, "us": US}

# Labels each country's rows can carry, besides names, dates and addresses.
ID_LABELS = {
    "uk": ["NATIONALID", "PASSPORTID", "DRIVERLICENSENUM", "TAXNUM"],
    "us": ["SSN", "TAXNUM", "PASSPORTID", "DRIVERLICENSENUM", "IDCARDNUM"],
}


REFERENCES: dict[str, list[tuple[str, str]]] = {
    "bank": [
        ("application reference", "APP-{year}-{d5}"),
        ("application number", "{D9}"),
        ("case number", "{D9}"),
        ("case reference", "CAS-{d6}-{C1}{d1}{C1}{d1}"),
        ("reference", "{D10}"),
        ("ticket", "#{D7}"),
        ("secure message ref", "{C1}{d8}"),
    ],
    "rental": [
        ("confirmation number", "{C1}{d7}"),
        ("confirmation number", "{D9}"),
        ("reservation number", "{C1}{d8}"),
        ("booking reference", "{C2}{d6}"),
        ("quote reference", "Q{d9}"),
    ],
    "general": [
        ("order number", "{D9}"),
        ("order no.", "{C1}{d8}"),
        ("invoice number", "INV-{year}-{d5}"),
        ("tracking number", "1Z{C3}{d3}{d10}"),
        ("ticket", "#{D7}"),
        ("case", "{D9}"),
    ],
}


class _IdGen(_Gen):
    """``_Gen`` with ID numbers and the UK/US pools."""

    def __init__(self, rng: random.Random, c: Country):
        super().__init__(rng, ENGLISH)
        self.c = c

    # --- people --------------------------------------------------------
    def person(self) -> dict:
        d, m = self.r.randint(1, 28), self.r.randint(1, 12)
        y = self.r.randint(1945, 2005)
        first = self.pick(self.c.first_names)
        last = self.pick(self.c.surnames)
        return {
            "first": first,
            "last": last,
            "female": self.r.random() < 0.5,
            "dob_parts": (d, m, y),
            "dob": self.us_or_uk_date(d, m, y),
            "email": f"{_ascii(first)}{self.pick(['.', '_', ''])}{_ascii(last)}"
            f"@{self.pick(self.c.email_domains)}".replace(" ", "").replace("'", ""),
            "phone": self.expand(self.pick(self.c.phone_formats)),
        }

    def us_or_uk_date(self, d: int, m: int, y: int) -> str:
        month = self.c.months[m - 1]
        if self.c.code == "us":
            return self.pick(
                [f"{m:02d}/{d:02d}/{y}", f"{month} {d}, {y}", f"{y}-{m:02d}-{d:02d}"]
            )
        day = f"{d}{EN_ORDINALS.get(d, 'th')}"
        return self.pick(
            [f"{d:02d}/{m:02d}/{y}", f"{d} {month} {y}", f"{day} {month} {y}"]
        )

    def event_date(self, years: tuple[int, int] = (2024, 2027)) -> str:
        d, m = self.r.randint(1, 28), self.r.randint(1, 12)
        return self.us_or_uk_date(d, m, self.r.randint(*years))

    def issue_and_expiry(self) -> tuple[str, str]:
        d, m, y = (
            self.r.randint(1, 28),
            self.r.randint(1, 12),
            self.r.randint(2016, 2025),
        )
        return self.us_or_uk_date(d, m, y), self.us_or_uk_date(d, m, y + 10)

    def address_parts(self) -> list[Part]:
        num = str(self.r.randint(1, 4800 if self.c.code == "us" else 180))
        street = self.pick(self.c.streets)
        parts: list[Part] = [(num, "BUILDINGNUM"), (" ", None), (street, "STREET")]
        if self.c.code == "us":
            state, city = self.pick(self.c.cities)
            parts += [(", ", None), (city, "CITY"), (", ", None), (state, "STATE")]
            parts += [(" ", None), (self.expand("{D5}"), "ZIP")]
        else:
            zip_, city = self.pick(self.c.cities)
            parts += [(", ", None), (city, "CITY"), (" ", None)]
            parts.append((self.expand(zip_, "ABDEFGHJLNPQRSTUWXYZ"), "ZIP"))
        if self.r.random() < 0.3:
            parts += [(", ", None), (self.pick(HOME_COUNTRIES[self.c.code]), "COUNTRY")]
        return parts

    def country(self) -> str:
        return self.pick(COUNTRIES)

    # --- ID numbers ----------------------------------------------------
    def ni_number(self) -> str:
        while True:
            prefix = self.pick(_NI_FIRST) + self.pick(_NI_SECOND)
            if prefix not in _NI_BAD:
                break
        digits, suffix = self.digits(6), self.pick("ABCD")
        return self.pick(
            [
                f"{prefix}{digits}{suffix}",
                f"{prefix} {digits[:2]} {digits[2:4]} {digits[4:]} {suffix}",
                f"{prefix} {digits} {suffix}",
            ]
        )

    def uk_driving_licence(self, p: dict) -> str:
        """DVLA format: surname (5, padded with 9), decade digit, month (+50
        for women), day, year digit, two initials (padded), 9, two letters."""
        name = "".join(ch for ch in _ascii(p["last"]).upper() if ch.isalpha())
        d, m, y = p["dob_parts"]
        month = m + 50 if p["female"] else m
        initials = _ascii(p["first"])[:1].upper() + self.pick(["", self.letters(1)])
        number = (
            f"{name[:5]:9<5}{(y // 10) % 10}{month:02d}{d:02d}{y % 10}"
            f"{initials:9<2}9{self.letters(2)}"
        )
        if self.r.random() < 0.25:
            return f"{number[:5]} {number[5:11]} {number[11:]}"
        return number

    def utr(self) -> str:
        n = self.expand("{D10}")
        return self.pick([n, f"{n[:5]} {n[5:]}"])

    def ssn(self) -> str:
        while True:
            area = self.r.randint(1, 899)
            if area != 666:
                break
        group, serial = self.r.randint(1, 99), self.r.randint(1, 9999)
        sep = "-" if self.r.random() < 0.85 else self.pick([" ", ""])
        return f"{area:03d}{sep}{group:02d}{sep}{serial:04d}"

    def us_tax_id(self) -> tuple[str, str]:
        """(cue, number): EIN 12-3456789 or ITIN 9XX-7X-XXXX."""
        if self.r.random() < 0.6:
            cue = self.pick(
                ["EIN", "Employer Identification Number", "federal tax ID (EIN)"]
            )
            return cue, f"{self.r.randint(10, 99)}-{self.digits(7)}"
        cue = self.pick(["ITIN", "Individual Taxpayer Identification Number"])
        return cue, f"9{self.digits(2)}-{self.r.randint(70, 88)}-{self.digits(4)}"

    def passport(self) -> str:
        if self.c.code == "uk":
            return self.expand("{D9}")
        return self.expand(self.pick(["{D9}", "{C1}{d8}"]))

    def us_state_number(self) -> str:
        """Driving licence / state ID formats of a few states."""
        return self.expand(
            self.pick(
                [
                    "{C1}{d7}",  # California
                    "{d3} {d3} {d3}",  # New York
                    "{d8}",  # Texas
                    "{C1}{d3}-{d3}-{d2}-{d3}-{d1}",  # Florida
                    "{C1}{d3}-{d4}-{d4}",  # Illinois
                    "WDL{C2}{d5}{C2}",  # Washington
                ]
            )
        )

    def id_number(self, label: str, p: dict) -> tuple[str, str]:
        """(cue, number) for ``label``; the cue matches the number's type."""
        c = self.c
        if label == "TAXNUM" and c.code == "us":
            return self.us_tax_id()
        number = {
            "NATIONALID": self.ni_number,
            "PASSPORTID": self.passport,
            "DRIVERLICENSENUM": (
                (lambda: self.uk_driving_licence(p))
                if c.code == "uk"
                else self.us_state_number
            ),
            "TAXNUM": self.utr,
            "SSN": self.ssn,
            "IDCARDNUM": self.us_state_number,
        }[label]()
        return self.pick(c.cues[label]), number

    # --- look-alikes that are not PII ----------------------------------
    def reference(self, setting: str) -> tuple[str, str]:
        """(cue, code): case, booking, order ... numbers shaped like IDs (nine
        digits, a letter and eight digits, ten digits) after a non-ID cue."""
        cue, pattern = self.pick(REFERENCES[setting])
        year = self.pick(["2025", "2026"])
        return cue, self.expand(pattern.replace("{year}", year))

    def flight(self) -> str:
        airline = self.pick(["BA", "VS", "EI", "UA", "AA", "DL", "LH", "KL", "EK"])
        return f"{airline} {self.r.randint(10, 2999)}"

    def eticket(self) -> str:
        return self.expand(self.pick(["125-{d10}", "{d3}{d10}", "016 {d10}"]))

    def money(self) -> str:
        return f"{self.c.currency}{self.r.randint(20, 98000):,}.{self.digits(2)}"


# ---------------------------------------------------------------------------
# Templates: each returns a list of (text, label) parts
# ---------------------------------------------------------------------------


def _name(p: dict) -> list[Part]:
    return [(p["first"], "FIRSTNAME"), (" ", None), (p["last"], "SURNAME")]


def _id_phrase(g: _IdGen, label: str, p: dict) -> list[Part]:
    cue, number = g.id_number(label, p)
    joiner = g.pick([" ", " ", ": ", " is ", " "])
    return [(cue + joiner, None), (number, label)]


def _ref_phrase(g: _IdGen, setting: str) -> list[Part]:
    cue, code = g.reference(setting)
    return [(f"{cue} {code}", None)]


def _pick_ids(g: _IdGen, k: int) -> list[str]:
    return g.r.sample(ID_LABELS[g.c.code], k)


def _licence_word(g: _IdGen) -> str:
    return "driving licence" if g.c.code == "uk" else "driver's license"


def t_account_opening(g: _IdGen, pii: bool) -> list[Part]:
    product = g.pick(
        [
            "current account",
            "savings account",
            "credit card",
            "ISA" if g.c.code == "uk" else "IRA",
            "checking account" if g.c.code == "us" else "joint account",
            "brokerage account",
            "mortgage",
        ]
    )
    if not pii:
        cue, code = g.reference("bank")
        doc = g.pick(["passport", _licence_word(g), "photo ID", "proof of address"])
        return [
            (
                g.pick(
                    [
                        f"Thank you for your {product} application ({cue} {code}). "
                        f"To finish the identity check, please upload a clear photo of "
                        f"your {doc} and a bank statement dated after "
                        f"{g.event_date()}. Uploads are reviewed within 2 working "
                        f"days; quote {code} if you call us.",
                        f"We could not verify your identity yet ({cue} {code}). The "
                        f"photo of your {doc} was blurred. Please upload it again "
                        f"before {g.event_date()}; your application stays open until "
                        f"then.",
                        f"Reminder: for your {product}, we still need a copy of your "
                        f"{doc}. You can bring it to any branch and mention "
                        f"{cue} {code}. Opening hours: Mon-Fri 9:00-17:00.",
                    ]
                ),
                None,
            )
        ]
    p = g.person()
    parts: list[Part] = [
        (g.pick(["Dear ", "Hi ", "Hello "]), None),
        (p["first"], "FIRSTNAME"),
    ]
    parts.append(
        (
            g.pick(
                [
                    f",\n\nthanks for applying for a {product}. We have the following "
                    "on file: ",
                    f",\n\nbefore we can open your {product}, please check these "
                    "details: ",
                ]
            ),
            None,
        )
    )
    parts += _name(p)
    parts += [(", born ", None), (p["dob"], "DATEOFBIRTH"), (", living at ", None)]
    parts += g.address_parts()
    parts.append((". ", None))
    for label in _pick_ids(g, g.r.randint(1, 2)):
        parts.append((g.pick(["Your ", "The ", "We verified your "]), None))
        parts += _id_phrase(g, label, p)
        parts.append(
            (g.pick([" matches our records. ", " has been checked. ", ". "]), None)
        )
    parts.append(("Your ", None))
    parts += _ref_phrase(g, "bank")
    parts.append(
        (g.pick([" stays the same.", " is unchanged.", " is still valid."]), None)
    )
    return parts


def t_onboarding(g: _IdGen, pii: bool) -> list[Part]:
    forms = g.r.sample(g.c.tax_forms, 2)
    start = g.event_date()
    if not pii:
        return [
            (
                g.pick(
                    [
                        f"Onboarding checklist (starts {start}): sign the contract, "
                        f"return the {forms[0]} and {forms[1]}, complete the security "
                        f"training (module {g.expand('SEC-{d3}')}) and collect your "
                        f"laptop, asset tag {g.expand('{C2}{d6}')}. Payroll cut-off is "
                        f"the {g.r.randint(15, 25)}th; cost centre {g.expand('{d4}')}.",
                        f"Hi team, three new starters join on {start}. Payroll needs "
                        f"their {forms[0]} by Friday; HR will check passports for the "
                        f"right-to-work file on day one. Desk numbers "
                        f"{g.expand('{d1}.{d2}')} to {g.expand('{d1}.{d2}')}, "
                        f"requisition {g.expand('REQ-{d6}')}.",
                        f"Please do not send your {forms[0]} or {forms[1]} by email. "
                        f"Upload them in the HR portal under case {g.expand('{D9}')}; "
                        f"the portal accepts PDF files up to 10 MB.",
                    ]
                ),
                None,
            )
        ]
    p = g.person()
    parts: list[Part] = [
        (g.pick(["New starter: ", "Employee: ", "Starter form for "]), None)
    ]
    parts += _name(p)
    parts += [
        (f", start date {start}. Date of birth ", None),
        (p["dob"], "DATEOFBIRTH"),
    ]
    parts += [(". Home address: ", None)]
    parts += g.address_parts()
    parts += [(". Mobile ", None), (p["phone"], "PHONENUMBER"), (". ", None)]
    labels = ["NATIONALID", "TAXNUM"] if g.c.code == "uk" else ["SSN", "TAXNUM"]
    labels = labels[: g.r.randint(1, 2)]
    if g.r.random() < 0.5:
        labels.append("PASSPORTID")  # right to work / I-9
    for label in labels:
        parts += _id_phrase(g, label, p)
        parts.append(("; ", None))
    parts.append(
        (f"{forms[0]} received, {forms[1]} outstanding (ticket #{g.digits(6)}).", None)
    )
    return parts


def t_travel(g: _IdGen, pii: bool) -> list[Part]:
    flight, ref = g.flight(), g.expand("{C6}")
    seat = f"{g.r.randint(1, 45)}{g.pick('ABCDEF')}"
    if not pii:
        return [
            (
                g.pick(
                    [
                        f"Flight {flight} (booking reference {ref}) has been "
                        f"rescheduled to {g.event_date()}. Your seat {seat} and "
                        f"e-ticket {g.eticket()} stay valid. Please bring the passport "
                        f"you booked with; online check-in opens 24 hours before "
                        f"departure.",
                        f"Your hotel booking {g.expand('{D9}')} is confirmed for "
                        f"{g.event_date()}, room type Double, rate code "
                        f"{g.expand('{C3}{d2}')}. Guests from outside the EU are asked "
                        f"to show a passport at check-in.",
                        f"Baggage claim {g.expand('{C3}{d5}')} for flight {flight}: "
                        f"we located your bag and will deliver it by {g.event_date()}. "
                        f"Keep your e-ticket {g.eticket()} for reference.",
                    ]
                ),
                None,
            )
        ]
    p = g.person()
    issue = g.pick([g.c.country, g.pick(HOME_COUNTRIES[g.c.code]), g.country()])
    issued, expires = g.issue_and_expiry()
    _, number = g.id_number("PASSPORTID", p)
    if g.r.random() < 0.4:  # sentence
        parts: list[Part] = [("Guest ", None)] + _name(p)
        parts += [(" checked in with passport ", None), (number, "PASSPORTID")]
        parts += [(" issued by ", None), (issue, "COUNTRY")]
        parts += [(f" (expires {expires}), arriving on flight {flight} from ", None)]
        parts += [
            (g.country(), "COUNTRY"),
            (f". Booking {ref}, room {g.r.randint(101, 940)}.", None),
        ]
        return parts
    parts = [("Passenger: ", None)] + _name(p)
    parts += [(f"\nFlight: {flight} to ", None), (g.country(), "COUNTRY")]
    parts += [(f", seat {seat}\nBooking ref: {ref}\nE-ticket: {g.eticket()}\n", None)]
    parts += [
        (g.pick(["Passport No.: ", "Passport number: ", "Passport: "]), None),
        (number, "PASSPORTID"),
    ]
    parts += [("\nCountry of issue: ", None), (issue, "COUNTRY")]
    parts += [(f"\nIssued {issued}, expires {expires}", None)]
    parts += [("\nDate of birth: ", None), (p["dob"], "DATEOFBIRTH")]
    return parts


def t_car_rental(g: _IdGen, pii: bool) -> list[Part]:
    agreement = g.expand("RA-{d7}")
    licence = _licence_word(g)
    if not pii:
        cue, code = g.reference("rental")
        return [
            (
                g.pick(
                    [
                        f"Your reservation ({cue} {code}) for a compact car is "
                        f"confirmed: pick-up {g.event_date()} at 10:00. The main driver "
                        f"must show a full {licence} held for at least a year and a "
                        f"credit card in their own name. Rental agreement {agreement} "
                        f"will be printed at the desk.",
                        f"Thanks for returning the car. Rental agreement {agreement} is "
                        f"closed; fuel level full, {g.number(g.r.randint(120, 2400))} "
                        f"miles driven. Your deposit is released within 5 days "
                        f"({cue} {code}).",
                        f"Can I add a second driver to {cue} {code}? They will bring "
                        f"their {licence} to the counter on {g.event_date()}.",
                    ]
                ),
                None,
            )
        ]
    p = g.person()
    cue, number = g.id_number("DRIVERLICENSENUM", p)
    if g.r.random() < 0.4:  # sentence
        parts: list[Part] = [("Please add ", None)] + _name(p)
        parts += [
            (f" as a second driver on agreement {agreement}. Their {cue} is ", None)
        ]
        parts += [(number, "DRIVERLICENSENUM"), (", date of birth ", None)]
        parts += [(p["dob"], "DATEOFBIRTH"), (".", None)]
        return parts
    parts = [(f"Rental agreement {agreement}\nMain driver: ", None)] + _name(p)
    parts += [("\nDate of birth: ", None), (p["dob"], "DATEOFBIRTH")]
    parts += [(f"\n{cue[0].upper() + cue[1:]}: ", None), (number, "DRIVERLICENSENUM")]
    if g.c.code == "us":
        state, _ = g.pick(g.c.cities)
        parts += [(", issued in ", None), (state, "STATE")]
    else:
        parts += [(", issued in ", None), (g.pick(HOME_COUNTRIES["uk"]), "COUNTRY")]
    parts += [("\nAddress: ", None)] + g.address_parts()
    cue2, code = g.reference("rental")
    parts += [(f"\nPick-up {g.event_date()}, {cue2} {code}", None)]
    return parts


def t_transfer(g: _IdGen, pii: bool) -> list[Part]:
    """Country names in money transfers and travel; always has PII."""
    amount = g.money()
    c1, c2 = g.country(), g.country()
    p = g.person()
    style = g.r.randrange(3)
    if style == 0:
        parts: list[Part] = [(f"Please confirm the transfer of {amount} to ", None)]
        parts += _name(p) + [(" in ", None), (c1, "COUNTRY")]
        parts += [
            (f", reference INV-{g.digits(5)}, submitted on {g.event_date()}.", None)
        ]
        return parts
    if style == 1:
        parts = [("I did not authorise the payment of ", None)]
        parts += [
            (f"{amount} sent from my account to a bank in ", None),
            (c1, "COUNTRY"),
        ]
        parts += [
            (f" on {g.event_date()}. I was travelling in ", None),
            (c2, "COUNTRY"),
        ]
        parts += [(" at the time. ", None)] + _name(p)
        return parts
    parts = _name(p) + [(" relocated from ", None), (c1, "COUNTRY")]
    parts += [(" to ", None), (c2, "COUNTRY")]
    parts += [
        (
            f" in {g.r.randint(2015, 2026)}; the relocation allowance of {amount} was paid "
            f"under case {g.expand('{D9}')}.",
            None,
        )
    ]
    return parts


def t_record(g: _IdGen, pii: bool) -> list[Part]:
    """Key/value records (form, CSV, JSON) with ID fields; always has PII."""
    people = [g.person() for _ in range(g.r.randint(1, 3))]
    labels = _pick_ids(g, g.r.randint(1, 3))
    keys = {lab: g.pick(g.c.keys[lab]) for lab in labels}
    style = g.r.randrange(3)
    parts: list[Part] = []
    if style == 0:  # form
        p = people[0]
        parts += [(g.pick(["Full name: ", "Name: "]), None)] + _name(p)
        parts += [("\nDate of birth: ", None), (p["dob"], "DATEOFBIRTH")]
        for lab in labels:
            parts += [(f"\n{keys[lab]}: ", None), (g.id_number(lab, p)[1], lab)]
        parts += [
            ("\nNationality: ", None),
            (g.pick([g.c.country, g.country()]), "COUNTRY"),
        ]
        parts += [(f"\nForm ref: {g.expand('{C2}-{d6}')}", None)]
    elif style == 1:  # CSV
        cols = ["first_name", "last_name", "dob"] + [keys[lab] for lab in labels]
        parts.append((",".join(cols) + ",record_id\n", None))
        for p in people:
            parts += [(p["first"], "FIRSTNAME"), (",", None), (p["last"], "SURNAME")]
            d, m, y = p["dob_parts"]
            dob = p["dob"] if "," not in p["dob"] else f"{y}-{m:02d}-{d:02d}"
            parts += [(",", None), (dob, "DATEOFBIRTH")]
            for lab in labels:
                parts += [(",", None), (g.id_number(lab, p)[1], lab)]
            parts.append((f",{g.expand('{d7}')}\n", None))
    else:  # JSON
        parts.append(("[\n", None))
        for i, p in enumerate(people):
            parts += [('  {"name": "', None)] + _name(p)
            parts += [('", "dob": "', None), (p["dob"], "DATEOFBIRTH"), ('"', None)]
            for lab in labels:
                parts += [
                    (f', "{keys[lab]}": "', None),
                    (g.id_number(lab, p)[1], lab),
                    ('"', None),
                ]
            parts.append(
                (
                    f', "id": {g.expand("{D6}")}}}'
                    + (",\n" if i < len(people) - 1 else "\n"),
                    None,
                )
            )
        parts.append(("]", None))
    while parts and parts[-1][1] is None and not parts[-1][0].strip():
        parts.pop()
    if parts and parts[-1][1] is None:
        parts[-1] = (parts[-1][0].rstrip(), None)
    return parts


TEMPLATES: list[tuple[Callable[[_IdGen, bool], list[Part]], bool]] = [
    # (template, can be PII-free)
    (t_account_opening, True),
    (t_onboarding, True),
    (t_travel, True),
    (t_car_rental, True),
    (t_transfer, False),
    (t_record, False),
]
# Four of six templates can be PII-free; 0.5 of those makes about a third of
# all rows PII-free.
DEFAULT_PII_FREE_SHARE = 0.5


def generate(
    num_rows: int, seed: int = 13, pii_free_share: float = DEFAULT_PII_FREE_SHARE
) -> list[dict]:
    """Return ``num_rows`` English rows (half UK, half US) in the internal
    training format. PII-free rows carry ``"pii_free": True``."""
    rows: list[dict] = []
    seen: set[str] = set()
    for i, c in enumerate((UK, US)):
        target = num_rows // 2 + (num_rows % 2 if i == 0 else 0)
        g = _IdGen(random.Random(f"{seed}:ids:{c.code}"), c)
        made, attempts = 0, 0
        while made < target and attempts < target * 20:
            attempts += 1
            template, can_be_free = g.pick(TEMPLATES)
            pii = not (can_be_free and g.r.random() < pii_free_share)
            text, mask = _assemble(template(g, pii))
            if text in seen or _PROBE_RE.search(text):
                continue
            seen.add(text)
            made += 1
            rows.append(
                {
                    "text": text,
                    "privacy_mask": mask,
                    "coreferences": [],
                    "language": "English",
                    "country": c.country,
                    "source": f"synthetic_ids_{c.code}",
                    "pii_free": not mask,
                }
            )
    return rows
