"""Template generator for German negative and contrast training rows.

The LLM-generated Kiji data has almost no text where codes, numbers or dates
are *not* PII: 97.9% of German 6+-digit strings and 98.5% of letter+digit codes
carry a PII label, car vocabulary appears in 0 rows, and there are no PII-free
rows. This module produces rows whose labels are correct by construction:

- PII-free rows: workshop notes, part numbers, fault codes, tyre sizes, order /
  invoice / ticket numbers, appointment and invoice dates, prices, versions.
- Contrast rows: the same kind of text plus real PII next to it (a plate after
  "Kennzeichen", a birth date after "geboren am", a sign-off name, an address),
  so the model has to use context instead of shape.

Values used by the 22 hand-written German probes (tests/benchmark German eval)
are deliberately not in the pools, so the probes stay a held-out check.

Usage:
    from model.dataset.synthetic_german import generate
    rows = generate(num_rows=6000, seed=13)
"""

from __future__ import annotations

import random
import string
from collections.abc import Callable

Part = tuple[str, str | None]  # (text, Kiji label or None)

# ---------------------------------------------------------------------------
# Value pools (no values from the German probes)
# ---------------------------------------------------------------------------

CAR_MODELS = [
    "VW Passat B8 (3G)",
    "VW Polo 6R",
    "VW Touran 1T",
    "Skoda Octavia III (5E)",
    "Seat Leon 5F",
    "Audi A3 8V",
    "Audi A6 C7",
    "BMW E90 320d",
    "BMW F10 520d",
    "BMW G30",
    "BMW E87 118i",
    "BMW E39 525i",
    "Mercedes W211 E 220 CDI",
    "Mercedes W212",
    "Mercedes W176 A 180",
    "Mercedes Sprinter 906",
    "Opel Astra J",
    "Opel Astra K",
    "Opel Insignia A",
    "Ford Focus Mk3",
    "Ford Fiesta Mk7",
    "Ford Transit Custom",
    "Toyota Corolla E12",
    "Toyota Yaris XP13",
    "Honda Civic FK2",
    "Mazda 6 GJ",
    "Mazda 3 BM",
    "Renault Clio IV",
    "Renault Kangoo II",
    "Peugeot 308 T9",
    "Citroën C4 Picasso",
    "Hyundai i30 PD",
    "Kia Ceed JD",
    "Fiat Ducato 250",
    "Volvo V70 III",
]
ENGINE_CODES = [
    "N52",
    "N57",
    "N20",
    "M271",
    "OM642",
    "OM651 EVO",
    "CAXA",
    "CBAB",
    "CHZB",
    "DFGA",
    "1KR-FE",
    "2ZR-FXE",
    "K9K",
    "R18A",
    "DV6TED4",
    "B47D20",
    "EA211",
    "Z18XER",
    "A20DTH",
    "M9R",
]
PART_PREFIXES = ["OE-Nr.", "Teilenummer", "Art.-Nr.", "Ersatzteil", "Best.-Nr."]
WORKSHOP_JOBS = [
    "Bremsbeläge HA erneuert",
    "Zahnriemen inkl. Wasserpumpe getauscht",
    "Ölwechsel mit Filter durchgeführt",
    "Klimaanlage befüllt (R1234yf, 480 g)",
    "Koppelstange VA rechts ersetzt",
    "AGR-Ventil gereinigt",
    "Glühkerzen 1-4 erneuert",
    "Batterie geprüft, Ruhestrom 35 mA",
    "Spur und Sturz vermessen",
    "Domlager VA links getauscht",
    "Thermostat ersetzt, Kühlsystem entlüftet",
    "Radlager HA links erneuert",
]
OIL_SPECS = [
    "5W-30 nach VW 504 00",
    "0W-30 nach MB 229.52",
    "5W-40 nach BMW LL-04",
    "0W-16 nach Toyota-Freigabe",
    "5W-30 nach Ford WSS-M2C913-D",
]
TYRE_LOADS = ["88H", "91H", "94V", "95W", "97Y", "99V", "103T"]
PROBE_OBD_CODES = {"P0420", "P0171", "P0016"}  # used by the German probes

FIRST_NAMES = [
    "Lukas",
    "Hannah",
    "Leon",
    "Mia",
    "Finn",
    "Emilia",
    "Paul",
    "Lea",
    "Jonas",
    "Lina",
    "Elias",
    "Marie",
    "Noah",
    "Johanna",
    "Felix",
    "Clara",
    "Moritz",
    "Greta",
    "Henrik",
    "Ida",
    "Ole",
    "Frieda",
    "Mehmet",
    "Ayşe",
    "Emre",
    "Elif",
    "Piotr",
    "Agnieszka",
    "Dragan",
    "Jelena",
    "Luca",
    "Giulia",
    "Ahmad",
    "Fatma",
    "Karl-Heinz",
    "Hans-Peter",
    "Anne-Marie",
    "Eva-Maria",
    "Sven",
    "Ute",
    "Dirk",
    "Heike",
    "Torsten",
    "Silke",
    "Matthias",
    "Birgit",
    "Ralf",
    "Kerstin",
    "Uwe",
    "Sabine",
    "Gülşen",
    "Zoë",
    "Malte",
    "Wiebke",
    "Detlef",
    "Gisela",
    "Bastian",
    "Svenja",
    "Jannik",
    "Merle",
    "Thi Lan",
    "Nguyen",
    "Oksana",
]
SURNAMES = [
    "Schmidt",
    "Schneider",
    "Fischer",
    "Weber",
    "Meyer",
    "Wagner",
    "Becker",
    "Hoffmann",
    "Schulz",
    "Koch",
    "Richter",
    "Klein",
    "Wolf",
    "Neumann",
    "Schwarz",
    "Zimmermann",
    "Braun",
    "Hofmann",
    "Hartmann",
    "Krüger",
    "Lange",
    "Werner",
    "Lehmann",
    "Köhler",
    "Jäger",
    "Graf",
    "Vogel",
    "Fuchs",
    "Kaiser",
    "Peters",
    "Scholz",
    "Möller",
    "Brandt",
    "Haas",
    "Schuster",
    "Yılmaz",
    "Kaya",
    "Demir",
    "Şahin",
    "Kowalski",
    "Nowak",
    "Wiśniewski",
    "Petrović",
    "Rossi",
    "Ferrari",
    "Nguyen",
    "Tran",
    "Popescu",
    "Ivanova",
    "Müller-Lüdenscheidt",
    "Schmidt-Rottluff",
    "von Hagen",
    "de Vries",
    "Overbeck",
    "Thießen",
    "Bäumer",
    "Großkopf",
    "Kleinschmidt",
    "Öztürk",
    "Brückner",
    "Weißenborn",
    "Lührs",
    "Janßen",
    "Heß",
]
STREETS = [
    "Bahnhofstraße",
    "Schillerstraße",
    "Gartenweg",
    "Lindenallee",
    "Am Mühlbach",
    "Kirchplatz",
    "Goethering",
    "Friedrich-Ebert-Straße",
    "Industriestraße",
    "Ringstraße",
    "Waldweg",
    "Marktplatz",
    "Hafenstraße",
    "Uhlandstraße",
    "Kastanienweg",
    "An der Alster",
    "Rosenstraße",
    "Zum Sportplatz",
]
CITIES = [  # (ZIP, city)
    ("10115", "Berlin"),
    ("50667", "Köln"),
    ("60311", "Frankfurt am Main"),
    ("70173", "Stuttgart"),
    ("40213", "Düsseldorf"),
    ("04109", "Leipzig"),
    ("90402", "Nürnberg"),
    ("30159", "Hannover"),
    ("28195", "Bremen"),
    ("44135", "Dortmund"),
    ("45127", "Essen"),
    ("33602", "Bielefeld"),
    ("53111", "Bonn"),
    ("48143", "Münster"),
    ("76133", "Karlsruhe"),
    ("68159", "Mannheim"),
    ("86150", "Augsburg"),
    ("65183", "Wiesbaden"),
    ("24103", "Kiel"),
    ("18055", "Rostock"),
    ("99084", "Erfurt"),
    ("93047", "Regensburg"),
    ("79098", "Freiburg im Breisgau"),
    ("26122", "Oldenburg"),
]
PLATE_DISTRICTS = [
    "B",
    "K",
    "F",
    "S",
    "D",
    "DO",
    "E",
    "DU",
    "BO",
    "HB",
    "H",
    "L",
    "DD",
    "N",
    "A",
    "KA",
    "MA",
    "WI",
    "MZ",
    "KI",
    "HL",
    "OS",
    "BI",
    "MS",
    "AC",
    "BN",
    "GE",
    "RE",
    "GT",
    "PB",
    "LU",
    "FR",
    "UL",
    "R",
    "PA",
    "RO",
    "OL",
    "WOB",
    "BS",
    "MD",
]
EMAIL_DOMAINS = [
    "web.de",
    "gmx.de",
    "t-online.de",
    "posteo.de",
    "mailbox.org",
    "example.de",
]

GREETINGS_NAMED = [
    "Hallo Herr {s},",
    "Hallo Frau {s},",
    "Guten Tag Herr {s},",
    "Sehr geehrte Frau {s},",
    "Sehr geehrter Herr {s},",
    "Moin {f},",
    "Hi {f},",
    "Liebe {f},",
    "Lieber {f},",
]
GREETINGS_PLAIN = [
    "Hallo zusammen,",
    "Sehr geehrte Damen und Herren,",
    "Guten Morgen,",
    "Hallo Team,",
    "Moin,",
]
SIGNOFFS = [
    "Viele Grüße",
    "Mit freundlichen Grüßen",
    "Beste Grüße",
    "Gruß",
    "LG",
    "Freundliche Grüße",
    "Herzliche Grüße",
    "VG",
    "Schöne Grüße aus der Werkstatt",
]
ROLES = [
    "Serviceberater",
    "Werkstattleitung",
    "Teiledienst",
    "Kundendienst",
    "Disposition",
    "Buchhaltung",
    "",
]
MONTHS = [
    "Januar",
    "Februar",
    "März",
    "April",
    "Mai",
    "Juni",
    "Juli",
    "August",
    "September",
    "Oktober",
    "November",
    "Dezember",
]
WEEKDAYS = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag"]


# ---------------------------------------------------------------------------
# Value makers
# ---------------------------------------------------------------------------


class _Gen:
    def __init__(self, rng: random.Random):
        self.r = rng

    def pick(self, seq):
        return self.r.choice(seq)

    def digits(self, n: int) -> str:
        return "".join(self.r.choice(string.digits) for _ in range(n))

    def letters(self, n: int) -> str:
        return "".join(self.r.choice(string.ascii_uppercase) for _ in range(n))

    # --- not PII -------------------------------------------------------
    def obd_code(self) -> str:
        while True:
            code = (
                self.pick("PPPPBCU")
                + self.pick("0012")
                + self.digits(1)
                + self.pick(string.digits + "ABCDEF")
                + self.digits(1)
            )
            if code not in PROBE_OBD_CODES:
                return code

    def tyre(self) -> str:
        width = self.pick(["185", "195", "215", "225", "235", "245", "255"])
        ratio = self.pick(["40", "45", "50", "60", "65"])
        rim = self.pick(["R15", "R17", "R18", "R19"])
        return f"{width}/{ratio} {rim} {self.pick(TYRE_LOADS)}"

    def part_number(self) -> str:
        style = self.r.randrange(5)
        if style == 0:  # VAG
            return f"{self.digits(1)}{self.letters(1)}{self.digits(1)} {self.digits(3)} {self.digits(3)} {self.letters(self.r.choice([0, 1, 2]))}".strip()
        if style == 1:  # BMW
            return f"{self.digits(2)} {self.digits(2)} {self.digits(1)} {self.digits(3)} {self.digits(3)}"
        if style == 2:  # Mercedes
            return (
                f"A {self.digits(3)} {self.digits(3)} {self.digits(2)} {self.digits(2)}"
            )
        if style == 3:  # Bosch
            return f"0 {self.digits(3)} {self.digits(3)} {self.digits(3)}"
        return f"{self.letters(2)}{self.digits(2)}-{self.digits(2)}-{self.digits(3)}"

    def order_ref(self) -> tuple[str, str]:
        kind = self.r.randrange(7)
        year = self.pick(["2024", "2025", "2026"])
        if kind == 0:
            return "Werkstattauftrag", f"WA-{year}-{self.digits(5)}"
        if kind == 1:
            return "Rechnungsnummer", f"RE-{year}-{self.digits(5)}"
        if kind == 2:
            return "Bestellnummer", f"45{self.digits(8)}"
        if kind == 3:
            return "Ticket", f"#INC-{self.digits(6)}"
        if kind == 4:
            return "Lieferschein", f"LS {self.digits(7)}"
        if kind == 5:
            return "Angebot", f"AN-{self.digits(4)}/{year[2:]}"
        return "Vorgangsnummer", f"{self.letters(2)}{self.digits(6)}"

    def date(self) -> str:
        d, m, y = (
            self.r.randint(1, 28),
            self.r.randint(1, 12),
            self.pick([2025, 2026, 2027]),
        )
        style = self.r.randrange(3)
        if style == 0:
            return f"{d:02d}.{m:02d}.{y}"
        if style == 1:
            return f"{d}. {MONTHS[m - 1]} {y}"
        return f"{d:02d}.{m:02d}."

    def price(self) -> str:
        return f"{self.r.randint(19, 2400)},{self.digits(2)} EUR"

    def version(self) -> str:
        return f"{self.r.randint(1, 9)}.{self.r.randint(0, 20)}.{self.r.randint(0, 99)}"

    def serial(self) -> str:
        return f"SN {self.letters(2)}{self.digits(2)}-{self.digits(4)}"

    def mileage(self) -> str:
        return f"{self.r.randint(12, 289)}.{self.digits(3)} km"

    # --- PII -----------------------------------------------------------
    def plate(self) -> str:
        district = self.pick(PLATE_DISTRICTS)
        mid = self.letters(self.r.choice([1, 2]))
        num = str(self.r.randint(1, 9999 if len(district) + len(mid) <= 3 else 999))
        suffix = self.pick(["", "", "", "", "E", "H"])
        sep = self.pick(["-", "-", " "])
        return f"{district}{sep}{mid} {num}{suffix}"

    def dob(self) -> str:
        d, m, y = (
            self.r.randint(1, 28),
            self.r.randint(1, 12),
            self.r.randint(1940, 2004),
        )
        return self.pick([f"{d:02d}.{m:02d}.{y}", f"{d}. {MONTHS[m - 1]} {y}"])

    def phone(self) -> str:
        if self.r.random() < 0.5:
            return f"01{self.pick(['5', '6', '7'])}{self.digits(1)} {self.digits(7)}"
        return f"0{self.digits(self.r.choice([2, 3]))} {self.digits(self.r.choice([5, 6, 7]))}"

    def email(self, first: str, last: str) -> str:
        def ascii_(s: str) -> str:
            table = str.maketrans(
                {
                    "ä": "ae",
                    "ö": "oe",
                    "ü": "ue",
                    "ß": "ss",
                    "ı": "i",
                    "ş": "s",
                    "ğ": "g",
                    "ç": "c",
                    "ś": "s",
                    "ć": "c",
                    "ë": "e",
                }
            )
            return "".join(c for c in s.lower().translate(table) if c.isalnum())

        return f"{ascii_(first)}.{ascii_(last)}@{self.pick(EMAIL_DOMAINS)}"


# ---------------------------------------------------------------------------
# Templates: each returns a list of (text, label) parts
# ---------------------------------------------------------------------------


def _greeting(g: _Gen, first: str, last: str, named: bool) -> list[Part]:
    if not named:
        return [(g.pick(GREETINGS_PLAIN) + "\n\n", None)]
    tpl = g.pick(GREETINGS_NAMED)
    if "{s}" in tpl:
        pre, post = tpl.split("{s}")
        return [(pre, None), (last, "SURNAME"), (post + "\n\n", None)]
    pre, post = tpl.split("{f}")
    return [(pre, None), (first, "FIRSTNAME"), (post + "\n\n", None)]


def _signoff(g: _Gen, first: str, last: str, with_name: bool) -> list[Part]:
    parts: list[Part] = [("\n\n" + g.pick(SIGNOFFS) + "\n", None)]
    if not with_name:
        parts.append(
            (
                g.pick(["Ihr Werkstatt-Team", "Ihr Serviceteam", "Die Teileabteilung"]),
                None,
            )
        )
        return parts
    if g.r.random() < 0.25:
        parts.append((first, "FIRSTNAME"))
    else:
        parts += [(first, "FIRSTNAME"), (" ", None), (last, "SURNAME")]
    role = g.pick(ROLES)
    if role:
        parts.append(("\n" + role, None))
    return parts


def t_workshop_note(g: _Gen, pii: bool) -> list[Part]:
    parts: list[Part] = [
        (f"{g.pick(CAR_MODELS)}, Motor {g.pick(ENGINE_CODES)}, {g.mileage()}. ", None),
        (f"Fehlerspeicher: {g.obd_code()} und {g.obd_code()}. ", None),
        (f"{g.pick(WORKSHOP_JOBS)}. {g.pick(PART_PREFIXES)} {g.part_number()}.", None),
    ]
    if pii:
        parts += [(" Kennzeichen ", None), (g.plate(), "LICENSEPLATENUM"), (".", None)]
    return parts


def t_parts_order(g: _Gen, pii: bool) -> list[Part]:
    first, last = g.pick(FIRST_NAMES), g.pick(SURNAMES)
    kind, ref = g.order_ref()
    body: list[Part] = [
        (
            f"für den {g.pick(CAR_MODELS)} ({g.pick(ENGINE_CODES)}) bitte folgende Teile bestellen:\n",
            None,
        ),
        (f"- {g.pick(PART_PREFIXES)} {g.part_number()}, 2 Stück\n", None),
        (f"- {g.pick(PART_PREFIXES)} {g.part_number()}\n", None),
        (f"- 5 l {g.pick(OIL_SPECS)}\n", None),
        (f"{kind}: {ref}, Liefertermin {g.date()}.", None),
    ]
    return _greeting(g, first, last, pii) + body + _signoff(g, first, last, pii)


def t_tyres(g: _Gen, pii: bool) -> list[Part]:
    parts: list[Part] = [
        (f"Reifenwechsel {g.pick(CAR_MODELS)}: 4x {g.tyre()} Winterreifen, ", None),
        (f"DOT {g.digits(4)}, Profiltiefe {g.r.randint(3, 8)} mm. ", None),
        (f"Einlagerung Regal {g.letters(1)}-{g.r.randint(1, 40)}, {g.price()}.", None),
    ]
    if pii:
        first, last = g.pick(FIRST_NAMES), g.pick(SURNAMES)
        parts += [
            (" Abholung durch ", None),
            (first, "FIRSTNAME"),
            (" ", None),
            (last, "SURNAME"),
            (", Tel. ", None),
            (g.phone(), "PHONENUMBER"),
            (".", None),
        ]
    return parts


def t_invoice(g: _Gen, pii: bool) -> list[Part]:
    kind, ref = g.order_ref()
    parts: list[Part] = [
        (f"{kind} {ref} vom {g.date()}\n", None),
        (f"Position 1: {g.pick(WORKSHOP_JOBS)}, {g.price()}\n", None),
        (f"Position 2: {g.pick(PART_PREFIXES)} {g.part_number()}, {g.price()}\n", None),
        (f"Zahlbar bis {g.date()} ohne Abzug.", None),
    ]
    if pii:
        first, last = g.pick(FIRST_NAMES), g.pick(SURNAMES)
        zip_, city = g.pick(CITIES)
        parts = [
            ("Rechnungsempfänger: ", None),
            (first, "FIRSTNAME"),
            (" ", None),
            (last, "SURNAME"),
            (", ", None),
            (g.pick(STREETS), "STREET"),
            (" ", None),
            (str(g.r.randint(1, 180)) + g.pick(["", "", "a", "b"]), "BUILDINGNUM"),
            (", ", None),
            (zip_, "ZIP"),
            (" ", None),
            (city, "CITY"),
            ("\n", None),
        ] + parts
    return parts


def t_appointment(g: _Gen, pii: bool) -> list[Part]:
    first, last = g.pick(FIRST_NAMES), g.pick(SURNAMES)
    body: list[Part] = [
        (
            f"Ihr Termin für die Inspektion ist am {g.pick(WEEKDAYS)}, {g.date()}, um {g.r.randint(7, 16)}:{g.pick(['00', '15', '30', '45'])} Uhr bestätigt. ",
            None,
        ),
        (
            f"Die HU ist fällig im {g.r.randint(1, 12):02d}/{g.pick(['2026', '2027'])}. ",
            None,
        ),
        (
            f"Bitte bringen Sie das Serviceheft mit (letzter Service bei {g.mileage()}).",
            None,
        ),
    ]
    if pii and g.r.random() < 0.5:
        body += [
            (" Ihr Fahrzeug mit dem Kennzeichen ", None),
            (g.plate(), "LICENSEPLATENUM"),
            (" ist bei uns hinterlegt.", None),
        ]
    return _greeting(g, first, last, pii) + body + _signoff(g, first, last, pii)


def t_it_ticket(g: _Gen, pii: bool) -> list[Part]:
    kind, ref = g.order_ref()
    parts: list[Part] = [
        (
            f"Ticket #INC-{g.digits(6)}: Update auf Version {g.version()} fehlgeschlagen, ",
            None,
        ),
        (
            f"Fehlercode 0x{g.digits(4)}{g.letters(2)}. Gerät {g.serial()}, Build {g.digits(8)}. ",
            None,
        ),
        (f"Bezug: {kind} {ref}. Nächster Versuch am {g.date()}.", None),
    ]
    if pii:
        first, last = g.pick(FIRST_NAMES), g.pick(SURNAMES)
        parts += [
            (" Gemeldet von ", None),
            (first, "FIRSTNAME"),
            (" ", None),
            (last, "SURNAME"),
            (" (", None),
            (g.email(first, last), "EMAIL"),
            (").", None),
        ]
    return parts


def t_customer_contact(g: _Gen, pii: bool) -> list[Part]:
    """Always has PII: birth date and plate in context next to non-PII codes."""
    first, last = g.pick(FIRST_NAMES), g.pick(SURNAMES)
    kind, ref = g.order_ref()
    return [
        ("Halter: ", None),
        (first, "FIRSTNAME"),
        (" ", None),
        (last, "SURNAME"),
        (", geboren am ", None),
        (g.dob(), "DATEOFBIRTH"),
        ("\n", None),
        ("Kennzeichen: ", None),
        (g.plate(), "LICENSEPLATENUM"),
        ("\n", None),
        (f"Fahrzeug: {g.pick(CAR_MODELS)}, Motorcode {g.pick(ENGINE_CODES)}\n", None),
        (f"{kind}: {ref}, Annahme am {g.date()}", None),
    ]


TEMPLATES: list[tuple[Callable[[_Gen, bool], list[Part]], bool]] = [
    # (template, can be PII-free)
    (t_workshop_note, True),
    (t_parts_order, True),
    (t_tyres, True),
    (t_invoice, True),
    (t_appointment, True),
    (t_it_ticket, True),
    (t_customer_contact, False),
]


def _assemble(parts: list[Part]) -> tuple[str, list[dict]]:
    text, mask, pos = [], [], 0
    for chunk, label in parts:
        if label is not None:
            mask.append(
                {"value": chunk, "label": label, "start": pos, "end": pos + len(chunk)}
            )
        text.append(chunk)
        pos += len(chunk)
    return "".join(text), mask


def generate(num_rows: int, seed: int = 13, pii_free_share: float = 0.5) -> list[dict]:
    """Return ``num_rows`` samples in the internal training format.

    PII-free rows carry ``"pii_free": True`` so the training loader keeps them.
    """
    g = _Gen(random.Random(seed))
    rows, seen = [], set()
    attempts = 0
    while len(rows) < num_rows and attempts < num_rows * 20:
        attempts += 1
        template, can_be_free = g.pick(TEMPLATES)
        pii = not (can_be_free and g.r.random() < pii_free_share)
        text, mask = _assemble(template(g, pii))
        if text in seen:
            continue
        seen.add(text)
        rows.append(
            {
                "text": text,
                "privacy_mask": mask,
                "coreferences": [],
                "language": "German",
                "country": "Germany",
                "source": "synthetic_german_negatives",
                "pii_free": not mask,
            }
        )
    return rows
