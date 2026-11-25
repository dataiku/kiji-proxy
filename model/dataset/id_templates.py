"""English (UK and US) template rows with personal ID numbers in context.

The Kiji English data labels IDs by the cue word in front of them ("national
ID", "tax number", "ID card number"), but its values are mostly placeholders
(987654321, 123-45-6789, AB123456C), and the same format carries different
labels in different rows. The German-mix models therefore confuse ID types
(a driving licence called a passport) and miss country names outside the six
English-speaking countries of the Kiji data. These rows teach:

- the label from the cue word, with realistic, varied formats: UK National
  Insurance number (NATIONALID), passport, driving licence (derived from the
  name and birth date like a real DVLA number), Unique Taxpayer Reference
  (TAXNUM); US Social Security number, ITIN and EIN (TAXNUM), passport,
  state driving licence and state ID card (IDCARDNUM) in that state's format;
- that the same shapes are not PII after a non-ID cue: order, case, booking,
  confirmation, e-ticket and form numbers, flight numbers, issue and expiry
  dates; and that a cue word without a number ("bring your passport") labels
  nothing;
- country names from the whole world (COUNTRY), in addresses, travel
  documents and money transfers.

Settings: bank account opening (identity checks), HR onboarding (payroll,
right to work), air travel and hotels, car rental, money transfers, and
key/value records (forms, CSV, JSON). About a third of the rows are PII-free;
those are put together from interchangeable sentences, so they do not repeat
a handful of fixed frames.

Conventions follow the Kiji English rows: a UK National Insurance number is
NATIONALID, UK rows name the country only as "United Kingdom" or "UK" (Kiji
labels Scotland, Wales and Northern Ireland as STATE), and about 40% of the
birth dates have no birth word in front of them, so the cue-heavy ID rows do
not teach "no birth word, no birth date" (the v2 regression).

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
    "uk": _l("United Kingdom|United Kingdom|UK"),
    "us": _l("United States|United States|USA"),
}

# NI number prefixes: neither letter D, F, I, Q, U or V, second letter not O,
# and never BG, GB, KN, NK, NT, TN or ZZ.
_NI_FIRST = [c for c in string.ascii_uppercase if c not in "DFIQUV"]
_NI_SECOND = [c for c in string.ascii_uppercase if c not in "DFIQUVO"]
_NI_BAD = {"BG", "GB", "KN", "NK", "NT", "TN", "ZZ"}

# EIN prefixes the IRS assigns (00, 07-09, 17-19, 28, 29, 49, 69, 70, 78, 79,
# 89, 96 and 97 are never used).
EIN_PREFIXES = [
    f"{n:02d}"
    for n in range(1, 100)
    if n not in {7, 8, 9, 17, 18, 19, 28, 29, 49, 69, 70, 78, 79, 89, 96, 97}
]

# Driving licence / state ID card number format per state.
STATE_FORMATS = {
    "CA": "{C1}{d7}",
    "NY": "{d3} {d3} {d3}",
    "TX": "{d8}",
    "FL": "{C1}{d3}-{d3}-{d2}-{d3}-{d1}",
    "IL": "{C1}{d3}-{d4}-{d4}",
    "WA": "WDL{C2}{d5}{C2}",
}

FEMALE_NAMES = set(
    _l(
        "Amelia|Isla|Ava|Mia|Grace|Freya|Lily|Sophie|Evie|Poppy|Siân|Eilidh|Niamh|"
        "Priya|Aisha|Chloe|Ellie|Bethan|Zainab|Fiona|Megan|Rosie|Mary-Kate|"
        "Mary|Patricia|Jennifer|Linda|Elizabeth|Maria|Guadalupe|Keisha|Aaliyah|"
        "Ashley|Brittany|Madison|Mai|Yuki|Ananya|Fatima|Svetlana|Kayla|Savannah|"
        "Ximena|Imani|Molly|Olivia|Emma"
    )
)


@dataclass
class Country:
    code: str  # "uk" / "us"
    country: str  # as in the Kiji data
    first_names: list[str]
    surnames: list[str]
    streets: list[str]
    # UK: (postcode pattern, city); US: (state, city, first three ZIP digits)
    cities: list[tuple[str, ...]]
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
        "Brandon|Brittany|Tyler|Madison|Mai|Hiroshi|Yuki|Rajesh|Ananya|"
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
        ("TX", "Austin", "787"),
        ("TX", "Houston", "770"),
        ("TX", "El Paso", "799"),
        ("CA", "Sacramento", "958"),
        ("CA", "San Diego", "921"),
        ("CA", "Fresno", "937"),
        ("NY", "Albany", "122"),
        ("NY", "Buffalo", "142"),
        ("NY", "Rochester", "146"),
        ("FL", "Tampa", "336"),
        ("FL", "Orlando", "328"),
        ("FL", "Tallahassee", "323"),
        ("IL", "Springfield", "627"),
        ("IL", "Naperville", "605"),
        ("IL", "Peoria", "616"),
        ("WA", "Spokane", "992"),
        ("WA", "Tacoma", "984"),
        ("WA", "Olympia", "985"),
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
        # A person's tax ID without an SSN; EINs (EIN_CUES) only for businesses.
        "TAXNUM": _l("ITIN|Individual Taxpayer Identification Number|ITIN number"),
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
        "TAXNUM": _l("ITIN|Tax ID|TIN|tax_id"),
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
EIN_CUES = _l("EIN|Employer Identification Number|federal tax ID (EIN)")

# Labels each country's rows can carry, besides names, dates and addresses.
ID_LABELS = {
    "uk": ["NATIONALID", "PASSPORTID", "DRIVERLICENSENUM", "TAXNUM"],
    "us": ["SSN", "TAXNUM", "PASSPORTID", "DRIVERLICENSENUM", "IDCARDNUM"],
}

# Keys / column names for a birth date in records. Half carry no birth word.
DOB_KEYS = {
    "form": _l("Date of birth|DOB|Date of birth|Date|Born"),
    "column": _l("dob|date_of_birth|dob|date|d"),
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
        email_user = f"{_ascii(first)}{self.pick(['.', '_', ''])}{_ascii(last)}"
        return {
            "first": first,
            "last": last,
            "female": first in FEMALE_NAMES,
            "dob_parts": (d, m, y),
            "dob": self.date_str(d, m, y),
            "email": email_user.replace(" ", "").replace("'", "")
            + f"@{self.pick(self.c.email_domains)}",
            "phone": self.expand(self.pick(self.c.phone_formats)),
            "home": self.pick(self.c.cities),
        }

    def date_str(self, d: int, m: int, y: int, style: int | None = None) -> str:
        month = self.c.months[m - 1]
        if self.c.code == "us":
            styles = [
                f"{m:02d}/{d:02d}/{y}",
                f"{month} {d}, {y}",
                f"{y}-{m:02d}-{d:02d}",
            ]
        else:
            day = f"{d}{EN_ORDINALS.get(d, 'th')}"
            styles = [f"{d:02d}/{m:02d}/{y}", f"{d} {month} {y}", f"{day} {month} {y}"]
        return styles[self.r.randrange(3) if style is None else style]

    def event_date(self, years: tuple[int, int] = (2024, 2027)) -> str:
        d, m = self.r.randint(1, 28), self.r.randint(1, 12)
        return self.date_str(d, m, self.r.randint(*years))

    def issue_and_expiry(self) -> tuple[str, str]:
        """Issue and expiry date ten years apart, in the same format."""
        d, m = self.r.randint(1, 28), self.r.randint(1, 12)
        y, style = self.r.randint(2016, 2025), self.r.randrange(3)
        return self.date_str(d, m, y, style), self.date_str(d, m, y + 10, style)

    def address_parts(self, p: dict) -> list[Part]:
        """The person's home address; the city, state and ZIP agree."""
        num = str(self.r.randint(1, 4800 if self.c.code == "us" else 180))
        street = self.pick(self.c.streets)
        parts: list[Part] = [(num, "BUILDINGNUM"), (" ", None), (street, "STREET")]
        if self.c.code == "us":
            state, city, zip3 = p["home"]
            parts += [(", ", None), (city, "CITY"), (", ", None), (state, "STATE")]
            parts += [(" ", None), (zip3 + self.digits(2), "ZIP")]
        else:
            zip_, city = p["home"]
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
        """DVLA format: surname (5, Mac -> MC, padded with 9), decade digit,
        month (+50 for women), day, year digit, two initials (padded), 9,
        two letters."""
        name = "".join(ch for ch in _ascii(p["last"]).upper() if ch.isalpha())
        if name.startswith("MAC"):
            name = "MC" + name[3:]
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
            # 000, 666 and 900-999 are invalid SSN area numbers; 078 is the
            # SSA's "dummy" area used in test data and must not look like a
            # real person's number either.
            if area not in (666, 678) and area < 900:
                break
        while True:
            group = self.r.randint(1, 99)
            if group != 0:
                break
        while True:
            serial = self.r.randint(1, 9999)
            if serial != 0:
                break
        sep = "-" if self.r.random() < 0.85 else self.pick([" ", ""])
        return f"{area:03d}{sep}{group:02d}{sep}{serial:04d}"

    def itin(self) -> str:
        return f"9{self.digits(2)}-{self.r.randint(70, 88)}-{self.digits(4)}"

    def ein(self) -> str:
        return f"{self.pick(EIN_PREFIXES)}-{self.digits(7)}"

    def passport(self) -> str:
        if self.c.code == "uk":
            return self.expand("{D9}")
        return self.expand(self.pick(["{D9}", "{C1}{d8}"]))

    def state_number(self, p: dict) -> str:
        """Driving licence / state ID number in the home state's format."""
        return self.expand(STATE_FORMATS[p["home"][0]])

    def id_number(self, label: str, p: dict) -> tuple[str, str]:
        """(cue, number) for ``label``; the cue matches the number's type."""
        c = self.c
        number = {
            "NATIONALID": self.ni_number,
            "PASSPORTID": self.passport,
            "DRIVERLICENSENUM": (
                (lambda: self.uk_driving_licence(p))
                if c.code == "uk"
                else (lambda: self.state_number(p))
            ),
            "TAXNUM": self.utr if c.code == "uk" else self.itin,
            "SSN": self.ssn,
            "IDCARDNUM": lambda: self.state_number(p),
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


def _pick_ids(g: _IdGen, k: int) -> list[str]:
    """``k`` ID labels for one person; a US person has an SSN or an ITIN
    (TAXNUM), never both."""
    labels = g.r.sample(ID_LABELS[g.c.code], k)
    if "SSN" in labels and "TAXNUM" in labels:
        labels.remove(g.pick(["SSN", "TAXNUM"]))
    return labels


def _licence_word(g: _IdGen) -> str:
    return "driving licence" if g.c.code == "uk" else "driver's license"


def _dob_after_name(g: _IdGen, p: dict) -> list[Part]:
    """ ", born <date>" or, without a birth word, ", <date>" / " (<date>)"."""
    style = g.r.random()
    if style < 0.55:
        return [
            (g.pick([", born ", ", date of birth ", ", DOB "]), None),
            (p["dob"], "DATEOFBIRTH"),
        ]
    if style < 0.8:
        return [(", ", None), (p["dob"], "DATEOFBIRTH")]
    return [(" (", None), (p["dob"], "DATEOFBIRTH"), (")", None)]


def _sentences(
    g: _IdGen, openers: list[str], middles: list[str], closers: list[str]
) -> str:
    """One opener, one or two middles and maybe a closer, for PII-free rows."""
    picked = [g.pick(openers)] + g.r.sample(middles, g.r.randint(1, 2))
    if g.r.random() < 0.6:
        picked.append(g.pick(closers))
    return " ".join(picked)


def t_account_opening(g: _IdGen, pii: bool) -> list[Part]:
    business = g.c.code == "us" and g.r.random() < 0.2
    product = (
        g.pick(["business checking account", "sole proprietor account"])
        if business
        else g.pick(
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
    )
    if not pii:
        cue, code = g.reference("bank")
        doc = g.pick(["passport", _licence_word(g), "photo ID", "proof of address"])
        text = _sentences(
            g,
            [
                f"Thank you for your {product} application ({cue} {code}).",
                f"We could not verify your identity yet ({cue} {code}).",
                f"Reminder about your {product} ({cue} {code}).",
                f"Your {product} application is almost complete.",
                f"Re: {cue} {code}.",
            ],
            [
                f"Please upload a clear photo of your {doc}.",
                f"The photo of your {doc} was blurred, so please upload it again.",
                f"We also need a bank statement dated after {g.event_date()}.",
                f"You can bring your {doc} to any branch instead.",
                f"Uploads are reviewed within {g.r.randint(2, 5)} working days.",
                f"Your application stays open until {g.event_date()}.",
                "We never ask for your PIN or full password.",
            ],
            [
                f"Quote {code} if you call us.",
                "Opening hours: Mon-Fri 9:00-17:00.",
                "Thank you for banking with us.",
            ],
        )
        return [(text, None)]
    p = g.person()
    parts: list[Part] = [
        (g.pick(["Dear ", "Hi ", "Hello "]), None),
        (p["first"], "FIRSTNAME"),
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
        ),
    ]
    parts += _name(p) + _dob_after_name(g, p)
    parts += [(", living at ", None)] + g.address_parts(p) + [(". ", None)]
    ids: list[tuple[str, str, str]] = []  # (cue, number, label)
    if business:
        ids.append((g.pick(EIN_CUES), g.ein(), "TAXNUM"))
    for label in _pick_ids(g, g.r.randint(1, 2)):
        if business and label == "TAXNUM":
            continue
        cue, number = g.id_number(label, p)
        ids.append((cue, number, label))
    for cue, number, label in ids:
        before, after = g.pick(
            [
                (f"Your {cue} ", " matches our records. "),
                (f"We verified your {cue}: ", ". "),
                (f"The {cue} you gave us (", ") has been checked. "),
                (f"Your {cue} is ", ". "),
            ]
        )
        parts += [(before, None), (number, label), (after, None)]
    cue, code = g.reference("bank")
    parts.append(
        (f"Your {cue} {code} " + g.pick(["stays the same.", "is unchanged."]), None)
    )
    return parts


def t_onboarding(g: _IdGen, pii: bool) -> list[Part]:
    forms = g.r.sample(g.c.tax_forms, 2)
    start = g.event_date()
    if not pii:
        text = _sentences(
            g,
            [
                f"Onboarding checklist (starts {start}):",
                f"Hi team, {g.pick(['two', 'three', 'four'])} new starters join on {start}.",
                f"Welcome aboard! Your first day is {start}.",
                f"HR update for requisition {g.expand('REQ-{d6}')}.",
            ],
            [
                f"Please return the {forms[0]} and {forms[1]} before payroll cut-off "
                f"on the {g.r.randint(15, 25)}th.",
                "HR will check passports for the right-to-work file on day one.",
                f"Complete the security training (module {g.expand('SEC-{d3}')}).",
                f"Collect your laptop, asset tag {g.expand('{C2}{d6}')}.",
                f"Upload documents in the HR portal under case {g.expand('{D9}')}.",
                f"Please do not send your {forms[0]} by email.",
                f"Desk {g.expand('{d1}.{d2}')}, cost centre {g.expand('{d4}')}.",
            ],
            [
                "The portal accepts PDF files up to 10 MB.",
                "See you soon!",
                f"Questions? Ticket #{g.expand('{D6}')}.",
            ],
        )
        return [(text, None)]
    p = g.person()
    parts: list[Part] = [
        (g.pick(["New starter: ", "Employee: ", "Starter form for "]), None)
    ]
    parts += _name(p)
    if g.r.random() < 0.5:
        parts += [
            (f", start date {start}. Date of birth ", None),
            (p["dob"], "DATEOFBIRTH"),
        ]
    else:  # no birth word
        parts += [
            (" (", None),
            (p["dob"], "DATEOFBIRTH"),
            (f"), start date {start}", None),
        ]
    parts += [(". Home address: ", None)] + g.address_parts(p)
    parts += [(". Mobile ", None), (p["phone"], "PHONENUMBER"), (".", None)]
    if g.c.code == "uk":
        labels = ["NATIONALID", "TAXNUM"][: g.r.randint(1, 2)]
    else:
        labels = ["SSN"] if g.r.random() < 0.8 else ["TAXNUM"]  # SSN or ITIN
    if g.r.random() < 0.5:
        labels.append("PASSPORTID")  # right to work / I-9
    for label in labels:
        cue, number = g.id_number(label, p)
        parts += [(f" {cue}{g.pick([': ', ' '])}", None), (number, label), (";", None)]
    parts.append(
        (f" {forms[0]} received, {forms[1]} outstanding (ticket #{g.digits(6)}).", None)
    )
    return parts


def t_travel(g: _IdGen, pii: bool) -> list[Part]:
    flight, ref = g.flight(), g.expand("{C6}")
    seat = f"{g.r.randint(1, 45)}{g.pick('ABCDEF')}"
    if not pii:
        text = _sentences(
            g,
            [
                f"Flight {flight} (booking reference {ref}) has been rescheduled to "
                f"{g.event_date()}.",
                f"Your hotel booking {g.expand('{D9}')} is confirmed for "
                f"{g.event_date()}.",
                f"Baggage claim {g.expand('{C3}{d5}')} for flight {flight}:",
                f"Check-in for {flight} is now open.",
            ],
            [
                f"Your seat {seat} and e-ticket {g.eticket()} stay valid.",
                "Please bring the passport you booked with.",
                "Guests from outside the EU are asked to show a passport at check-in.",
                f"We located your bag and will deliver it by {g.event_date()}.",
                f"Room type Double, rate code {g.expand('{C3}{d2}')}.",
                f"Boarding closes 30 minutes before departure at gate {g.expand('{C1}{d2}')}.",
            ],
            [
                "Online check-in opens 24 hours before departure.",
                f"Keep your e-ticket {g.eticket()} for reference.",
                "Have a pleasant trip.",
            ],
        )
        return [(text, None)]
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
        text = _sentences(
            g,
            [
                f"Your reservation ({cue} {code}) for a compact car is confirmed.",
                f"Thanks for returning the car. Rental agreement {agreement} is closed.",
                f"Can I add a second driver to {cue} {code}?",
                f"Pick-up reminder for {cue} {code}.",
            ],
            [
                f"The main driver must show a full {licence} held for at least a year.",
                f"They will bring their {licence} to the counter on {g.event_date()}.",
                f"Pick-up is on {g.event_date()} at {g.r.randint(8, 17)}:00.",
                f"Fuel level full, {g.number(g.r.randint(120, 2400))} miles driven.",
                "Please bring a credit card in the main driver's name.",
                f"Rental agreement {agreement} will be printed at the desk.",
            ],
            [
                "Your deposit is released within 5 days.",
                "Drive safely!",
                "Late returns are charged per hour.",
            ],
        )
        return [(text, None)]
    p = g.person()
    cue, number = g.id_number("DRIVERLICENSENUM", p)
    if g.r.random() < 0.4:  # sentence
        parts: list[Part] = [("Please add ", None)] + _name(p)
        parts += _dob_after_name(g, p)
        parts += [
            (f" as a second driver on agreement {agreement}. Their {cue} is ", None),
            (number, "DRIVERLICENSENUM"),
            (".", None),
        ]
        return parts
    parts = [(f"Rental agreement {agreement}\nMain driver: ", None)] + _name(p)
    parts += [("\nDate of birth: ", None), (p["dob"], "DATEOFBIRTH")]
    parts += [(f"\n{cue[0].upper() + cue[1:]}: ", None), (number, "DRIVERLICENSENUM")]
    if g.c.code == "us":
        parts += [(", issued in ", None), (p["home"][0], "STATE")]
    else:
        parts += [(", issued in ", None), (g.pick(HOME_COUNTRIES["uk"]), "COUNTRY")]
    parts += [("\nAddress: ", None)] + g.address_parts(p)
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
        parts += [(f"\n{g.pick(DOB_KEYS['form'])}: ", None), (p["dob"], "DATEOFBIRTH")]
        for lab in labels:
            parts += [(f"\n{keys[lab]}: ", None), (g.id_number(lab, p)[1], lab)]
        parts += [
            ("\nNationality: ", None),
            (g.pick([g.c.country, g.country()]), "COUNTRY"),
        ]
        parts += [(f"\nForm ref: {g.expand('{C2}-{d6}')}", None)]
    elif style == 1:  # CSV
        dob_key = g.pick(DOB_KEYS["column"])
        cols = ["first_name", "last_name", dob_key] + [keys[lab] for lab in labels]
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
        dob_key = g.pick(DOB_KEYS["column"])
        parts.append(("[\n", None))
        for i, p in enumerate(people):
            parts += [('  {"name": "', None)] + _name(p)
            parts += [
                (f'", "{dob_key}": "', None),
                (p["dob"], "DATEOFBIRTH"),
                ('"', None),
            ]
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
