"""Template generator for negative and contrast training rows in six languages.

The LLM-generated Kiji data has almost no text where codes, numbers or dates
are *not* PII: in every language 97-99% of 6+-digit strings, 89-95% of
letter+digit codes and 88-98% of dates carry a PII label, and no row is
PII-free. So a model trained on it learns that every number, code and date is
PII. This module produces rows whose labels are correct by construction:

- PII-free rows: workshop notes, part numbers, fault codes, tyre sizes, order /
  invoice / ticket numbers, appointment and invoice dates, prices, versions.
- Contrast rows: the same kind of text plus real PII next to it (a plate after
  "licence plate" / "Kennzeichen" / "matrícula", a birth date after "born on" /
  "né le", a sign-off name, an address), so the model has to use context
  instead of shape.
- Person records: tables, CSV exports, lists, forms and JSON/XML with birth
  dates next to names and *no* birth word, sometimes beside an unlabelled
  event date. Without them the cue-only birth dates above (and in openpii)
  teach "no birth word -> no birth date".

Languages: German, English (UK), French (France), Spanish (Spain), Dutch
(Netherlands) and Danish (Denmark), the six languages of the Kiji data. Each
locale has its own sentences, names, addresses, phone, date and plate formats;
car models, engine and fault codes, tyre sizes and part numbers are shared.

Values used by the 22 hand-written German probes (tests/benchmark German eval)
are kept out of the pools, and rows that still contain one of their codes are
dropped, so the probes stay a held-out check.

Usage:
    from model.dataset.synthetic_negatives import generate
    rows = generate(1500, seed=13, language="French")
"""

from __future__ import annotations

import random
import re
import string
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

Part = tuple[str, str | None]  # (text, Kiji label or None)


def _l(s: str) -> list[str]:
    """Split a ``|``-separated pool (keeps spaces inside values)."""
    return [x.strip() for x in s.split("|")]


def _pairs(s: str) -> list[tuple[str, str]]:
    """Split ``zip=city|zip=city`` into (zip, city) pairs."""
    return [tuple(x.split("=", 1)) for x in _l(s)]  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Probe values (must never be generated)
# ---------------------------------------------------------------------------

PROBE_OBD_CODES = {"P0420", "P0171", "P0016"}  # used by the German probes
# Codes and PII strings from the German probes. A row containing one of them
# as a whole word is dropped.
PROBE_STRINGS = [
    "GA14DE",
    "P0420",
    "P0171",
    "P0016",
    "CJZA",
    "W204",
    "E46",
    "MX-5",
    "04E 115 561 H",
    "NC10-41-400",
    "13502050",
    "A 271 050 47 47",
    "7 551 444",
    "WA-2026-0815",
    "205/55 R16 91V",
    "M-AB 1234",
    "HH-KL 482E",
    "B-MX 55",
    "0241 9876543",
    "0171 2345678",
    "57 549 285 017",
    "DE89 3704 0044 0532 0130 00",
    "jens.kuehn@example.de",
]
# Names, streets and cities from the German probes; no pool may contain them.
PROBE_NAMES = set(
    _l(
        "Weiß|Jürgen|Großmann|Özdemir|Björn|Schäfer-Groß|Sören|Maik|Änne|Strauß|"
        "Jörg|Klaus-Dieter|Müller|Françoise|Lütke-Börding|Hauptstraße|München|"
        "Dominik|Kärcher|Am Gänsemarkt|Hamburg|Lessingstraße|Dresden|Böhm|"
        "Schloßallee|Aachen|Tobias|Rößler|Mareike|Dünnwald|Jens|Kühn|Ünal|Yıldız"
    )
)
_PROBE_RE = re.compile("|".join(rf"(?<!\w){re.escape(s)}(?!\w)" for s in PROBE_STRINGS))

# ---------------------------------------------------------------------------
# Shared non-PII pools
# ---------------------------------------------------------------------------

CAR_MODELS = _l(
    "VW Passat B8 (3G)|VW Polo 6R|VW Touran 1T|Skoda Octavia III (5E)|Seat Leon 5F|"
    "Audi A3 8V|Audi A6 C7|BMW E90 320d|BMW F10 520d|BMW G30|BMW E87 118i|"
    "BMW E39 525i|Mercedes W211 E 220 CDI|Mercedes W212|Mercedes W176 A 180|"
    "Mercedes Sprinter 906|Opel Astra J|Opel Astra K|Opel Insignia A|"
    "Ford Focus Mk3|Ford Fiesta Mk7|Ford Transit Custom|Toyota Corolla E12|"
    "Toyota Yaris XP13|Honda Civic FK2|Mazda 6 GJ|Mazda 3 BM|Renault Clio IV|"
    "Renault Kangoo II|Peugeot 308 T9|Citroën C4 Picasso|Hyundai i30 PD|Kia Ceed JD|"
    "Fiat Ducato 250|Volvo V70 III|Vauxhall Astra K|Peugeot 208 II|"
    "Renault Mégane IV|Dacia Sandero II|Seat Ibiza 6J|Skoda Fabia III|"
    "Nissan Qashqai J11|Citroën C3 III|Volvo XC60 II|Kia Picanto JA"
)
ENGINE_CODES = _l(
    "N52|N57|N20|M271|OM642|OM651 EVO|CAXA|CBAB|CHZB|DFGA|1KR-FE|2ZR-FXE|K9K|R18A|"
    "DV6TED4|B47D20|EA211|Z18XER|A20DTH|M9R"
)
OIL_SPECS = [  # (grade, approval)
    ("5W-30", "VW 504 00"),
    ("0W-30", "MB 229.52"),
    ("5W-40", "BMW LL-04"),
    ("5W-30", "Ford WSS-M2C913-D"),
    ("0W-20", "ACEA C5"),
    ("5W-30", "PSA B71 2290"),
]
TYRE_LOADS = _l("88H|91H|94V|95W|97Y|99V|103T")

PII_FIELDS = {
    "first": "FIRSTNAME",
    "last": "SURNAME",
    "plate": "LICENSEPLATENUM",
    "phone": "PHONENUMBER",
    "email": "EMAIL",
    "street": "STREET",
    "num": "BUILDINGNUM",
    "zip": "ZIP",
    "city": "CITY",
    "dob": "DATEOFBIRTH",
}

# ---------------------------------------------------------------------------
# Random helpers
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"\{(d|D|L|C)(\d+)\}")


class _Gen:
    def __init__(self, rng: random.Random, loc: Locale):
        self.r = rng
        self.loc = loc

    def pick(self, seq):
        return self.r.choice(seq)

    def digits(self, n: int) -> str:
        return "".join(self.r.choice(string.digits) for _ in range(n))

    def letters(self, n: int, alphabet: str = string.ascii_uppercase) -> str:
        return "".join(self.r.choice(alphabet) for _ in range(n))

    def expand(self, pattern: str, alphabet: str = string.ascii_uppercase) -> str:
        """Fill ``{dN}`` digits, ``{DN}`` digits without a leading zero,
        ``{LN}`` letters from ``alphabet`` and ``{CN}`` A-Z letters."""

        def sub(m: re.Match) -> str:
            kind, n = m.group(1), int(m.group(2))
            if kind == "d":
                return self.digits(n)
            if kind == "D":
                return self.pick("123456789") + self.digits(n - 1)
            if kind == "L":
                return self.letters(n, alphabet)
            return self.letters(n)

        return _TOKEN.sub(sub, pattern)

    def text(self, key: str) -> str:
        tpl = self.loc.text[key]
        return tpl if isinstance(tpl, str) else self.pick(tpl)

    def number(self, n: int) -> str:
        return f"{n:,}".replace(",", self.loc.thousands)

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
        kind, pattern = self.pick(self.loc.order_kinds)
        year = self.pick(["2024", "2025", "2026"])
        pattern = pattern.replace("{year}", year).replace("{yy}", year[2:])
        return kind, self.expand(pattern)

    def _date(self, formats: list[str], years: tuple[int, int]) -> str:
        d, m = self.r.randint(1, 28), self.r.randint(1, 12)
        y = self.r.randint(*years)
        month = self.loc.months[m - 1]
        return self.pick(formats).format(d=d, m=m, y=y, month=month)

    def date(self) -> str:
        return self._date(self.loc.date_formats, (2024, 2027))

    def due(self) -> str:
        return self._date(self.loc.due_formats, (2026, 2027))

    def time(self) -> str:
        h, mm = self.r.randint(7, 16), self.pick(["00", "15", "30", "45"])
        return self.loc.time_format.format(h=h, mm=mm)

    def price(self) -> str:
        return self.loc.price_format.format(
            amount=self.number(self.r.randint(19, 2400)), cents=self.digits(2)
        )

    def mileage(self) -> str:
        n = self.r.randint(12, 289) * 1000 + self.r.randint(0, 999)
        return self.loc.mileage_format.format(n=self.number(n))

    def version(self) -> str:
        return f"{self.r.randint(1, 9)}.{self.r.randint(0, 20)}.{self.r.randint(0, 99)}"

    def serial(self) -> str:
        return f"SN {self.letters(2)}{self.digits(2)}-{self.digits(4)}"

    # --- PII -----------------------------------------------------------
    def dob(self) -> str:
        return self._date(self.loc.dob_formats, (1940, 2004))

    def record_dob(self) -> str:
        """A birth date as it appears in exports and lists: local formats plus
        ISO dates and, for English, ordinals ("July 13th, 1960")."""
        d, m = self.r.randint(1, 28), self.r.randint(1, 12)
        y = self.r.randint(1940, 2006)
        style = self.r.random()
        if style < 0.2:
            return f"{y}-{m:02d}-{d:02d}"
        if style < 0.3:
            return f"{y}-{m:02d}-{d:02d}T00:00:00"
        if style < 0.5 and self.loc.code == "en":
            month = self.loc.months[m - 1]
            day = f"{d}{EN_ORDINALS.get(d, 'th')}"
            return self.pick([f"{month} {day}, {y}", f"{day} {month} {y}"])
        return self.pick(self.loc.dob_formats).format(
            d=d, m=m, y=y, month=self.loc.months[m - 1]
        )

    def plate(self) -> str:
        return self.loc.plate(self)

    def phone(self) -> str:
        return self.expand(self.pick(self.loc.phone_formats))

    def email(self, first: str, last: str) -> str:
        user = f"{_ascii(first)}{self.pick(['.', '.', '_', ''])}{_ascii(last)}"
        return f"{user}@{self.pick(self.loc.email_domains)}"

    def building_num(self) -> str:
        return str(self.r.randint(1, 180)) + self.pick(self.loc.num_suffixes)

    def address(self) -> dict[str, str]:
        zip_, city = self.pick(self.loc.cities)
        return {
            "street": self.pick(self.loc.streets),
            "num": self.building_num(),
            "zip": self.expand(zip_, "ABCEGHJKLMNPRTVWXYZ"),
            "city": city,
        }


_TRANSLIT = str.maketrans(
    {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "ø": "oe", "æ": "ae", "å": "aa"}
)


def _ascii(s: str) -> str:
    s = s.lower().replace("ı", "i").translate(_TRANSLIT)
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if c.isascii() and c.isalnum())


# ---------------------------------------------------------------------------
# Plates
# ---------------------------------------------------------------------------

DE_PLATE_DISTRICTS = _l(
    "B|K|F|S|D|DO|E|DU|BO|HB|H|L|DD|N|A|KA|MA|WI|MZ|KI|HL|OS|BI|MS|AC|BN|GE|RE|GT|"
    "PB|LU|FR|UL|R|PA|RO|OL|WOB|BS|MD"
)


def _plate_de(g: _Gen) -> str:
    district = g.pick(DE_PLATE_DISTRICTS)
    mid = g.letters(g.r.choice([1, 2]))
    num = str(g.r.randint(1, 9999 if len(district) + len(mid) <= 3 else 999))
    suffix = g.pick(["", "", "", "", "E", "H"])
    sep = g.pick(["-", "-", " "])
    return f"{district}{sep}{mid} {num}{suffix}"


def _plate_uk(g: _Gen) -> str:
    # Current format: area letters, age identifier, three random letters.
    letters = "ABCDEFGHJKLMNOPRSTUVWXY"
    age = g.pick([f"{y:02d}" for y in range(2, 26)] + [str(y) for y in range(51, 75)])
    plate = f"{g.letters(2, letters)}{age} {g.letters(3, 'ABCDEFGHJKLMNOPRSTUVWXYZ')}"
    return plate.replace(" ", "") if g.r.random() < 0.15 else plate


def _plate_fr(g: _Gen) -> str:
    letters = "ABCDEFGHJKLMNPRSTVWXYZ"  # SIV plates skip I, O, U
    if g.r.random() < 0.15:  # old FNI plates: 1234 AB 69
        return (
            f"{g.r.randint(1, 9999)} {g.letters(2, letters)} {g.r.randint(1, 95):02d}"
        )
    sep = "-" if g.r.random() < 0.85 else " "
    return f"{g.letters(2, letters)}{sep}{g.digits(3)}{sep}{g.letters(2, letters)}"


def _plate_es(g: _Gen) -> str:
    letters = "BCDFGHJKLMNPRSTVWXYZ"  # no vowels, no Ñ or Q
    sep = g.pick([" ", " ", " ", "-", ""])
    return f"{g.digits(4)}{sep}{g.letters(3, letters)}"


def _plate_nl(g: _Gen) -> str:
    sidecode = g.pick(
        [
            "{d2}-{L3}-{d1}",
            "{d1}-{L3}-{d2}",
            "{L2}-{d3}-{L1}",
            "{L1}-{d3}-{L2}",
            "{L3}-{d2}-{L1}",
            "{L1}-{d2}-{L3}",
            "{d2}-{L2}-{L2}",
            "{L2}-{L2}-{d2}",
        ]
    )
    return g.expand(sidecode, "BDFGHJKLNPRSTVXZ")


def _plate_dk(g: _Gen) -> str:
    plate = f"{g.letters(2, 'ABCDEFGHJKLMNOPRSTUVXYZ')} {g.digits(2)} {g.digits(3)}"
    return plate.replace(" ", "") if g.r.random() < 0.1 else plate


# ---------------------------------------------------------------------------
# Locales
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Locale:
    language: str  # as in the Kiji data
    country: str
    code: str
    first_names: list[str]
    surnames: list[str]
    streets: list[str]
    cities: list[tuple[str, str]]  # (ZIP pattern, city)
    num_suffixes: list[str]
    email_domains: list[str]
    phone_formats: list[str]
    plate: Callable[[_Gen], str]
    months: list[str]
    weekdays: list[str]
    date_formats: list[str]  # str.format with d, m, y, month
    dob_formats: list[str]
    due_formats: list[str]
    time_format: str
    thousands: str
    price_format: str  # amount, cents
    mileage_format: str  # n
    greetings_named: list[str]  # {first} or {last}
    greetings_plain: list[str]
    signoffs: list[str]
    team_signoffs: list[str]
    roles: list[str]
    part_prefixes: list[str]
    workshop_jobs: list[str]
    order_kinds: list[tuple[str, str]]  # (word, pattern for _Gen.expand)
    text: dict[str, str | list[str]]


GERMAN = Locale(
    language="German",
    country="Germany",
    code="de",
    first_names=_l(
        "Lukas|Hannah|Leon|Mia|Finn|Emilia|Paul|Lea|Jonas|Lina|Elias|Marie|Noah|"
        "Johanna|Felix|Clara|Moritz|Greta|Henrik|Ida|Ole|Frieda|Mehmet|Ayşe|Emre|Elif|"
        "Piotr|Agnieszka|Dragan|Jelena|Luca|Giulia|Ahmad|Fatma|Karl-Heinz|Hans-Peter|"
        "Anne-Marie|Eva-Maria|Sven|Ute|Dirk|Heike|Torsten|Silke|Matthias|Birgit|Ralf|"
        "Kerstin|Uwe|Sabine|Gülşen|Zoë|Malte|Wiebke|Detlef|Gisela|Bastian|Svenja|"
        "Jannik|Merle|Thi Lan|Nguyen|Oksana"
    ),
    surnames=_l(
        "Schmidt|Schneider|Fischer|Weber|Meyer|Wagner|Becker|Hoffmann|Schulz|Koch|"
        "Richter|Klein|Wolf|Neumann|Schwarz|Zimmermann|Braun|Hofmann|Hartmann|Krüger|"
        "Lange|Werner|Lehmann|Köhler|Jäger|Graf|Vogel|Fuchs|Kaiser|Peters|Scholz|"
        "Möller|Brandt|Haas|Schuster|Yılmaz|Kaya|Demir|Şahin|Kowalski|Nowak|Wiśniewski|"
        "Petrović|Rossi|Ferrari|Nguyen|Tran|Popescu|Ivanova|"
        "Schmidt-Rottluff|von Hagen|de Vries|Overbeck|Thießen|Bäumer|Großkopf|"
        "Kleinschmidt|Öztürk|Brückner|Weißenborn|Lührs|Janßen|Heß"
    ),
    streets=_l(
        "Bahnhofstraße|Schillerstraße|Gartenweg|Lindenallee|Am Mühlbach|Kirchplatz|"
        "Goethering|Friedrich-Ebert-Straße|Industriestraße|Ringstraße|Waldweg|"
        "Marktplatz|Hafenstraße|Uhlandstraße|Kastanienweg|An der Alster|Rosenstraße|"
        "Zum Sportplatz"
    ),
    cities=_pairs(
        "10115=Berlin|50667=Köln|60311=Frankfurt am Main|70173=Stuttgart|"
        "40213=Düsseldorf|04109=Leipzig|90402=Nürnberg|30159=Hannover|28195=Bremen|"
        "44135=Dortmund|45127=Essen|33602=Bielefeld|53111=Bonn|48143=Münster|"
        "76133=Karlsruhe|68159=Mannheim|86150=Augsburg|65183=Wiesbaden|24103=Kiel|"
        "18055=Rostock|99084=Erfurt|93047=Regensburg|79098=Freiburg im Breisgau|"
        "26122=Oldenburg"
    ),
    num_suffixes=["", "", "a", "b"],
    email_domains=_l("web.de|gmx.de|t-online.de|posteo.de|mailbox.org|example.de"),
    phone_formats=_l(
        "015{d1} {d7}|016{d1} {d7}|017{d1} {d7}|0{D2} {d5}|0{D2} {d6}|0{D3} {d6}|"
        "0{D3} {d7}|+49 1{d2} {d7}"
    ),
    plate=_plate_de,
    months=_l(
        "Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|"
        "Dezember"
    ),
    weekdays=_l("Montag|Dienstag|Mittwoch|Donnerstag|Freitag"),
    date_formats=["{d:02d}.{m:02d}.{y}", "{d}. {month} {y}", "{d:02d}.{m:02d}."],
    dob_formats=["{d:02d}.{m:02d}.{y}", "{d}. {month} {y}"],
    due_formats=["{m:02d}/{y}"],
    time_format="{h}:{mm} Uhr",
    thousands=".",
    price_format="{amount},{cents} EUR",
    mileage_format="{n} km",
    greetings_named=_l(
        "Hallo Herr {last},|Hallo Frau {last},|Guten Tag Herr {last},|"
        "Sehr geehrte Frau {last},|Sehr geehrter Herr {last},|Moin {first},|"
        "Hi {first},|Liebe {first},|Lieber {first},"
    ),
    greetings_plain=_l(
        "Hallo zusammen,|Sehr geehrte Damen und Herren,|Guten Morgen,|Hallo Team,|Moin,"
    ),
    signoffs=_l(
        "Viele Grüße|Mit freundlichen Grüßen|Beste Grüße|Gruß|LG|Freundliche Grüße|"
        "Herzliche Grüße|VG|Schöne Grüße aus der Werkstatt"
    ),
    team_signoffs=_l("Ihr Werkstatt-Team|Ihr Serviceteam|Die Teileabteilung"),
    roles=_l(
        "Serviceberater|Werkstattleitung|Teiledienst|Kundendienst|Disposition|"
        "Buchhaltung|"
    ),
    part_prefixes=_l("OE-Nr.|Teilenummer|Art.-Nr.|Ersatzteil|Best.-Nr."),
    workshop_jobs=_l(
        "Bremsbeläge HA erneuert|Zahnriemen inkl. Wasserpumpe getauscht|"
        "Ölwechsel mit Filter durchgeführt|Klimaanlage befüllt (R1234yf, 480 g)|"
        "Koppelstange VA rechts ersetzt|AGR-Ventil gereinigt|Glühkerzen 1-4 erneuert|"
        "Batterie geprüft, Ruhestrom 35 mA|Spur und Sturz vermessen|"
        "Domlager VA links getauscht|Thermostat ersetzt, Kühlsystem entlüftet|"
        "Radlager HA links erneuert"
    ),
    order_kinds=[
        ("Werkstattauftrag", "WA-{year}-{d5}"),
        ("Rechnungsnummer", "RE-{year}-{d5}"),
        ("Bestellnummer", "45{d8}"),
        ("Ticket", "#INC-{d6}"),
        ("Lieferschein", "LS {d7}"),
        ("Angebot", "AN-{d4}/{yy}"),
        ("Vorgangsnummer", "{C2}{d6}"),
    ],
    text={
        "workshop": "{car}, Motor {engine}, {mileage}. Fehlerspeicher: {obd1} und {obd2}. {job}. {pp} {part}.",
        "workshop_plate": " Kennzeichen {plate}.",
        "order_intro": "für den {car} ({engine}) bitte folgende Teile bestellen:\n",
        "order_line_qty": "- {pp} {part}, 2 Stück\n",
        "order_line": "- {pp} {part}\n",
        "order_oil": "- 5 l {grade} nach {spec}\n",
        "order_ref": "{kind}: {ref}, Liefertermin {date}.",
        "tyres": "Reifenwechsel {car}: 4x {tyre} Winterreifen, DOT {dot}, Profiltiefe {depth} mm. Einlagerung Regal {shelf}, {price}.",
        "tyres_pickup": " Abholung durch {first} {last}, Tel. {phone}.",
        "invoice": "{kind} {ref} vom {date}\nPosition 1: {job}, {price}\nPosition 2: {pp} {part}, {price2}\nZahlbar bis {date2} ohne Abzug.",
        "invoice_to": "Rechnungsempfänger: {first} {last}, {street} {num}, {zip} {city}\n",
        "appointment": "Ihr Termin für die Inspektion ist am {weekday}, {date}, um {time} bestätigt. Die HU ist fällig im {due}. Bitte bringen Sie das Serviceheft mit (letzter Service bei {mileage}).",
        "appointment_plate": " Ihr Fahrzeug mit dem Kennzeichen {plate} ist bei uns hinterlegt.",
        "ticket": "Ticket #INC-{inc}: Update auf Version {version} fehlgeschlagen, Fehlercode 0x{hex}. Gerät {serial}, Build {build}. Bezug: {kind} {ref}. Nächster Versuch am {date}.",
        "ticket_reporter": " Gemeldet von {first} {last} ({email}).",
        "contact": [
            "Halter: {first} {last}, geboren am {dob}\nKennzeichen: {plate}\nFahrzeug: {car}, Motorcode {engine}\n{kind}: {ref}, Annahme am {date}",
            "Halterin: {first} {last}, Geburtsdatum {dob}\nKennzeichen: {plate}\nFahrzeug: {car}, Motorcode {engine}\n{kind}: {ref}, Annahme am {date}",
        ],
    },
)

ENGLISH = Locale(
    language="English",
    country="United Kingdom",
    code="en",
    first_names=_l(
        "Oliver|Amelia|George|Isla|Harry|Ava|Jack|Mia|Charlie|Grace|Thomas|Freya|Oscar|"
        "Lily|James|Sophie|William|Evie|Alfie|Poppy|Rhys|Siân|Callum|Eilidh|Declan|"
        "Niamh|Priya|Arjun|Aisha|Mohammed|Chloe|Ellie|Gareth|Bethan|Kwame|Chidi|Wei|"
        "Zainab|Hamish|Fiona|Dylan|Megan|Tomasz|Rosie|Mary-Kate|Jean-Luc"
    ),
    surnames=_l(
        "Smith|Jones|Williams|Taylor|Brown|Davies|Evans|Wilson|Roberts|Johnson|Lewis|"
        "Walker|Robinson|Wood|Thompson|White|Watson|Jackson|Wright|Green|Harris|Cooper|"
        "King|Hughes|Edwards|Clarke|Patel|Khan|Ahmed|Singh|Okafor|Mensah|O'Brien|"
        "MacLeod|Fitzgerald|Llewellyn|Ashworth|Pemberton|Chen|Kowalczyk|Nwosu|"
        "Hargreaves|Featherstone|Bennett-Jones"
    ),
    streets=_l(
        "High Street|Station Road|Church Lane|Victoria Road|Mill Lane|Park Avenue|"
        "Queens Road|The Green|Manor Way|Kings Crescent|Elm Grove|Chapel Street|"
        "Harbour View|Orchard Close|Bridge Street|Ashfield Terrace"
    ),
    cities=_pairs(
        "M1 {d1}{L2}=Manchester|LS1 {d1}{L2}=Leeds|B3 {d1}{L2}=Birmingham|"
        "BS1 {d1}{L2}=Bristol|NE1 {d1}{L2}=Newcastle upon Tyne|S1 {d1}{L2}=Sheffield|"
        "CF10 {d1}{L2}=Cardiff|EH1 {d1}{L2}=Edinburgh|G2 {d1}{L2}=Glasgow|"
        "NG1 {d1}{L2}=Nottingham|L1 {d1}{L2}=Liverpool|LE1 {d1}{L2}=Leicester|"
        "CV1 {d1}{L2}=Coventry|PL1 {d1}{L2}=Plymouth|YO1 {d1}{L2}=York|"
        "EX4 {d1}{L2}=Exeter|NR2 {d1}{L2}=Norwich|AB10 {d1}{L2}=Aberdeen|"
        "BT1 {d1}{L2}=Belfast|SA1 {d1}{L2}=Swansea|OX1 {d1}{L2}=Oxford|"
        "DD1 {d1}{L2}=Dundee"
    ),
    num_suffixes=["", "", "", "A", "B"],
    email_domains=_l(
        "gmail.com|outlook.com|btinternet.com|yahoo.co.uk|sky.com|example.co.uk"
    ),
    phone_formats=_l(
        "07{d3} {d6}|07{d3} {d6}|+44 7{d3} {d6}|020 {d4} {d4}|0161 {d3} {d4}|"
        "0113 {d3} {d4}|01{d3} {d6}|+44 20 {d4} {d4}"
    ),
    plate=_plate_uk,
    months=_l(
        "January|February|March|April|May|June|July|August|September|October|"
        "November|December"
    ),
    weekdays=_l("Monday|Tuesday|Wednesday|Thursday|Friday"),
    date_formats=["{d:02d}/{m:02d}/{y}", "{d} {month} {y}", "{d:02d}/{m:02d}"],
    dob_formats=["{d:02d}/{m:02d}/{y}", "{d} {month} {y}"],
    due_formats=["{month} {y}", "{m:02d}/{y}"],
    time_format="{h}:{mm}",
    thousands=",",
    price_format="£{amount}.{cents}",
    mileage_format="{n} miles",
    greetings_named=_l(
        "Hi {first},|Hello {first},|Dear Mr {last},|Dear Mrs {last},|Dear Ms {last},|"
        "Morning {first},|Dear {first},"
    ),
    greetings_plain=_l("Hi all,|Dear Sir or Madam,|Good morning,|Hi team,|Hello,"),
    signoffs=_l(
        "Kind regards|Best regards|Many thanks|Regards|Thanks|Best wishes|Cheers|"
        "Yours sincerely"
    ),
    team_signoffs=_l("The Workshop Team|Customer Service|Parts Department"),
    roles=_l(
        "Service Advisor|Workshop Manager|Parts Desk|Customer Service|Bookings|"
        "Accounts|"
    ),
    part_prefixes=_l("OE no.|Part no.|Part number|Order code|Spare part"),
    workshop_jobs=_l(
        "Rear brake pads replaced|Timing belt and water pump replaced|"
        "Oil and filter change carried out|Air con regassed (R1234yf, 480 g)|"
        "Front right drop link replaced|EGR valve cleaned|Glow plugs 1-4 replaced|"
        "Battery tested, parasitic drain 35 mA|Four-wheel alignment checked and adjusted|"
        "Front left top mount replaced|Thermostat replaced, cooling system bled|"
        "Rear left wheel bearing replaced"
    ),
    order_kinds=[
        ("Job card", "JC-{year}-{d5}"),
        ("Invoice no.", "INV-{year}-{d5}"),
        ("PO number", "45{d8}"),
        ("Ticket", "#INC-{d6}"),
        ("Delivery note", "DN {d7}"),
        ("Quote", "QT-{d4}/{yy}"),
        ("Case ref", "{C2}{d6}"),
    ],
    text={
        "workshop": "{car}, engine {engine}, {mileage}. Stored fault codes: {obd1} and {obd2}. {job}. {pp} {part}.",
        "workshop_plate": [
            " Reg {plate}.",
            " Registration {plate}.",
            " Number plate {plate}.",
            " Licence plate {plate}.",
        ],
        "order_intro": "please order the following parts for the {car} ({engine}):\n",
        "order_line_qty": "- {pp} {part}, qty 2\n",
        "order_line": "- {pp} {part}\n",
        "order_oil": "- 5 l {grade} to {spec}\n",
        "order_ref": "{kind}: {ref}, delivery due {date}.",
        "tyres": "Tyre change {car}: 4x {tyre} winter tyres, DOT {dot}, tread depth {depth} mm. Stored on rack {shelf}, {price}.",
        "tyres_pickup": " Collection by {first} {last}, tel. {phone}.",
        "invoice": "{kind} {ref} dated {date}\nLine 1: {job}, {price}\nLine 2: {pp} {part}, {price2}\nPayment due by {date2}.",
        "invoice_to": "Bill to: {first} {last}, {num} {street}, {city} {zip}\n",
        "appointment": "Your service appointment is confirmed for {weekday} {date} at {time}. The MOT is due in {due}. Please bring the service book (last service at {mileage}).",
        "appointment_plate": " Your vehicle, registration {plate}, is on our system.",
        "ticket": "Ticket #INC-{inc}: update to version {version} failed, error code 0x{hex}. Device {serial}, build {build}. Ref: {kind} {ref}. Next attempt on {date}.",
        "ticket_reporter": " Reported by {first} {last} ({email}).",
        "contact": [
            "Keeper: {first} {last}, born on {dob}\nRegistration: {plate}\nVehicle: {car}, engine code {engine}\n{kind}: {ref}, booked in on {date}",
            "Customer: {first} {last}, date of birth {dob}\nLicence plate: {plate}\nVehicle: {car}, engine code {engine}\n{kind}: {ref}, booked in on {date}",
        ],
    },
)

FRENCH = Locale(
    language="French",
    country="France",
    code="fr",
    first_names=_l(
        "Lucas|Emma|Louis|Jade|Hugo|Louise|Gabriel|Alice|Arthur|Chloé|Jules|Léa|"
        "Raphaël|Manon|Théo|Camille|Nathan|Inès|Mathis|Zoé|Antoine|Margaux|"
        "Jean-Baptiste|Marie-Claire|Anne-Sophie|Pierre-Louis|Karim|Yasmine|Mehdi|Leïla|"
        "Moussa|Aminata|Thanh|Sébastien|Nathalie|Philippe|Sylvie|Benoît|Gaëlle|Hélène|"
        "Loïc|Maëlys|Noémie"
    ),
    surnames=_l(
        "Martin|Bernard|Dubois|Thomas|Robert|Richard|Petit|Durand|Leroy|Moreau|Simon|"
        "Laurent|Lefèvre|Michel|Garcia|David|Bertrand|Roux|Vincent|Fournier|Morel|"
        "Girard|André|Mercier|Dupont|Lambert|Bonnet|Benali|Haddad|Diallo|Traoré|Nguyen|"
        "Le Goff|Le Gall|Da Silva|Dos Santos|Fontaine|Rousseau|Chevalier|Faure|"
        "Gauthier|Lemaire|Barbier"
    ),
    streets=_l(
        "rue de la République|rue Victor Hugo|avenue Jean Jaurès|boulevard Gambetta|"
        "rue du Moulin|place de l'Église|chemin des Vignes|rue des Lilas|"
        "allée des Tilleuls|rue Pasteur|avenue de la Gare|impasse des Acacias|"
        "rue Nationale|boulevard Voltaire|route de Paris"
    ),
    cities=_pairs(
        "69002=Lyon|13001=Marseille|31000=Toulouse|06000=Nice|44000=Nantes|"
        "67000=Strasbourg|34000=Montpellier|33000=Bordeaux|59000=Lille|35000=Rennes|"
        "51100=Reims|76600=Le Havre|42000=Saint-Étienne|83000=Toulon|38000=Grenoble|"
        "21000=Dijon|49000=Angers|37000=Tours|63000=Clermont-Ferrand|87000=Limoges|"
        "29200=Brest|64100=Bayonne"
    ),
    num_suffixes=["", "", "", " bis", " ter"],
    email_domains=_l("orange.fr|free.fr|laposte.net|sfr.fr|gmail.com|example.fr"),
    phone_formats=_l(
        "06 {d2} {d2} {d2} {d2}|07 {d2} {d2} {d2} {d2}|0{D1} {d2} {d2} {d2} {d2}|"
        "+33 6 {d2} {d2} {d2} {d2}|06.{d2}.{d2}.{d2}.{d2}|+33 1 {d2} {d2} {d2} {d2}"
    ),
    plate=_plate_fr,
    months=_l(
        "janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|"
        "décembre"
    ),
    weekdays=_l("lundi|mardi|mercredi|jeudi|vendredi"),
    date_formats=["{d:02d}/{m:02d}/{y}", "{d} {month} {y}", "{d:02d}/{m:02d}"],
    dob_formats=["{d:02d}/{m:02d}/{y}", "{d} {month} {y}"],
    due_formats=["{month} {y}", "{m:02d}/{y}"],
    time_format="{h}h{mm}",
    thousands=" ",
    price_format="{amount},{cents} €",
    mileage_format="{n} km",
    greetings_named=_l(
        "Bonjour Monsieur {last},|Bonjour Madame {last},|Madame {last},|"
        "Monsieur {last},|Bonjour {first},|Salut {first},|Cher {first},|Chère {first},"
    ),
    greetings_plain=_l(
        "Bonjour,|Madame, Monsieur,|Bonjour à tous,|Bonjour l'équipe,|Salut,"
    ),
    signoffs=_l(
        "Cordialement|Bien cordialement|Bien à vous|Salutations distinguées|"
        "Merci et bonne journée|À bientôt|Bonne journée|Sincères salutations"
    ),
    team_signoffs=_l("L'équipe atelier|Le service client|Le magasin pièces"),
    roles=_l(
        "Conseiller service|Chef d'atelier|Magasin pièces|Service après-vente|"
        "Planning|Comptabilité|"
    ),
    part_prefixes=_l("Réf. OE|Référence|Réf. pièce|Code article|N° de pièce"),
    workshop_jobs=_l(
        "Plaquettes de frein AR remplacées|"
        "Courroie de distribution et pompe à eau remplacées|"
        "Vidange avec filtre à huile effectuée|Recharge climatisation (R1234yf, 480 g)|"
        "Biellette de barre stabilisatrice AV droite remplacée|Vanne EGR nettoyée|"
        "Bougies de préchauffage 1 à 4 remplacées|"
        "Batterie contrôlée, consommation au repos 35 mA|"
        "Géométrie contrôlée et réglée|Coupelle d'amortisseur AV gauche remplacée|"
        "Thermostat remplacé, circuit de refroidissement purgé|"
        "Roulement de roue AR gauche remplacé"
    ),
    order_kinds=[
        ("Ordre de réparation", "OR-{year}-{d5}"),
        ("N° de facture", "FA-{year}-{d5}"),
        ("Bon de commande", "45{d8}"),
        ("Ticket", "#INC-{d6}"),
        ("Bon de livraison", "BL {d7}"),
        ("Devis", "DV-{d4}/{yy}"),
        ("N° de dossier", "{C2}{d6}"),
    ],
    text={
        "workshop": "{car}, moteur {engine}, {mileage}. Mémoire défauts : {obd1} et {obd2}. {job}. {pp} {part}.",
        "workshop_plate": [
            " Immatriculation {plate}.",
            " Plaque d'immatriculation {plate}.",
            " Immat. {plate}.",
        ],
        "order_intro": "merci de commander les pièces suivantes pour la {car} ({engine}) :\n",
        "order_line_qty": "- {pp} {part}, 2 pièces\n",
        "order_line": "- {pp} {part}\n",
        "order_oil": "- 5 l {grade} selon {spec}\n",
        "order_ref": "{kind} : {ref}, livraison prévue le {date}.",
        "tyres": "Changement de pneus {car} : 4x {tyre} pneus hiver, DOT {dot}, profondeur de sculpture {depth} mm. Stockage étagère {shelf}, {price}.",
        "tyres_pickup": " Retrait par {first} {last}, tél. {phone}.",
        "invoice": "{kind} {ref} du {date}\nLigne 1 : {job}, {price}\nLigne 2 : {pp} {part}, {price2}\nPayable avant le {date2} sans escompte.",
        "invoice_to": "Facturé à : {first} {last}, {num} {street}, {zip} {city}\n",
        "appointment": "Votre rendez-vous pour la révision est confirmé le {weekday} {date} à {time}. Le contrôle technique est à faire avant {due}. Merci d'apporter le carnet d'entretien (dernière révision à {mileage}).",
        "appointment_plate": " Votre véhicule immatriculé {plate} est enregistré chez nous.",
        "ticket": "Ticket #INC-{inc} : échec de la mise à jour vers la version {version}, code d'erreur 0x{hex}. Appareil {serial}, build {build}. Réf. : {kind} {ref}. Nouvel essai le {date}.",
        "ticket_reporter": " Signalé par {first} {last} ({email}).",
        "contact": [
            "Titulaire : {first} {last}, né le {dob}\nImmatriculation : {plate}\nVéhicule : {car}, code moteur {engine}\n{kind} : {ref}, réception le {date}",
            "Cliente : {first} {last}, née le {dob}\nPlaque d'immatriculation : {plate}\nVéhicule : {car}, code moteur {engine}\n{kind} : {ref}, réception le {date}",
            "Client : {first} {last}, date de naissance : {dob}\nImmatriculation : {plate}\nVéhicule : {car}, code moteur {engine}\n{kind} : {ref}, réception le {date}",
        ],
    },
)

SPANISH = Locale(
    language="Spanish",
    country="Spain",
    code="es",
    first_names=_l(
        "Hugo|Lucía|Martín|Sofía|Pablo|Martina|Daniel|María|Alejandro|Paula|Álvaro|"
        "Julia|Adrián|Carmen|Diego|Valeria|Javier|Elena|Sergio|Alba|José Luis|"
        "María José|Juan Carlos|Ana Belén|Íñigo|Ainhoa|Jordi|Montserrat|Xabier|Nerea|"
        "Youssef|Fátima|Andrés|Rocío|Gonzalo|Inmaculada|Raúl|Lorena|Iker|Begoña|"
        "Ionela"
    ),
    surnames=_l(
        "García|Rodríguez|González|Fernández|López|Martínez|Sánchez|Pérez|Gómez|"
        "Jiménez|Ruiz|Hernández|Díaz|Moreno|Muñoz|Álvarez|Romero|Alonso|Gutiérrez|"
        "Navarro|Torres|Domínguez|Vázquez|Ramos|Gil|Serrano|Blanco|Molina|Castro|"
        "Ortega|Rubio|Iglesias|Etxeberria|Puig|Ferrer|Castells|El Amrani|Popescu|"
        "de la Fuente|del Río|Sáez"
    ),
    streets=_l(
        "Calle Mayor|Calle Real|Avenida de la Constitución|Calle del Carmen|"
        "Calle San Juan|Plaza de España|Calle de Alcalá|Avenida de Andalucía|"
        "Calle Nueva|Camino Viejo|Calle de la Paz|Ronda de Toledo|Calle del Sol|"
        "Travesía del Pilar|Paseo de la Estación"
    ),
    cities=_pairs(
        "28013=Madrid|08001=Barcelona|46002=Valencia|41004=Sevilla|50001=Zaragoza|"
        "29005=Málaga|30001=Murcia|07001=Palma|35002=Las Palmas de Gran Canaria|"
        "48001=Bilbao|03002=Alicante|14003=Córdoba|47001=Valladolid|36201=Vigo|"
        "33201=Gijón|15003=A Coruña|01001=Vitoria-Gasteiz|18009=Granada|33003=Oviedo|"
        "31001=Pamplona|39001=Santander|37001=Salamanca"
    ),
    num_suffixes=["", "", "", "", " bis"],
    email_domains=_l(
        "gmail.com|hotmail.es|telefonica.net|yahoo.es|outlook.es|example.es"
    ),
    phone_formats=_l(
        "6{d2} {d3} {d3}|6{d2} {d2} {d2} {d2}|+34 6{d2} {d3} {d3}|7{d2} {d3} {d3}|"
        "91 {d3} {d2} {d2}|93 {d3} {d2} {d2}|9{D2} {d3} {d3}"
    ),
    plate=_plate_es,
    months=_l(
        "enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|"
        "noviembre|diciembre"
    ),
    weekdays=_l("lunes|martes|miércoles|jueves|viernes"),
    date_formats=["{d:02d}/{m:02d}/{y}", "{d} de {month} de {y}", "{d:02d}/{m:02d}"],
    dob_formats=["{d:02d}/{m:02d}/{y}", "{d} de {month} de {y}", "{d:02d}-{m:02d}-{y}"],
    due_formats=["{month} de {y}", "{m:02d}/{y}"],
    time_format="{h}:{mm} h",
    thousands=".",
    price_format="{amount},{cents} €",
    mileage_format="{n} km",
    greetings_named=_l(
        "Hola, {first}:|Estimado Sr. {last}:|Estimada Sra. {last}:|"
        "Buenos días, Sr. {last}:|Buenos días, Sra. {last}:|Hola {first},|"
        "Querida {first}:|Querido {first}:"
    ),
    greetings_plain=_l(
        "Hola a todos:|Estimados señores:|Buenos días:|Hola, equipo:|Hola:"
    ),
    signoffs=_l(
        "Saludos|Un saludo|Atentamente|Saludos cordiales|Un cordial saludo|"
        "Gracias y un saludo|Muchas gracias"
    ),
    team_signoffs=_l(
        "El equipo del taller|Atención al cliente|Departamento de recambios"
    ),
    roles=_l(
        "Asesor de servicio|Jefe de taller|Recambios|Atención al cliente|"
        "Planificación|Contabilidad|"
    ),
    part_prefixes=_l(
        "Ref. OE|Referencia|N.º de pieza|Código de artículo|Ref. recambio"
    ),
    workshop_jobs=_l(
        "Pastillas de freno traseras sustituidas|"
        "Correa de distribución y bomba de agua cambiadas|"
        "Cambio de aceite y filtro realizado|"
        "Carga de aire acondicionado (R1234yf, 480 g)|"
        "Bieleta de la barra estabilizadora delantera derecha sustituida|"
        "Válvula EGR limpiada|Calentadores 1-4 sustituidos|"
        "Batería comprobada, consumo en reposo 35 mA|Alineación de dirección realizada|"
        "Copela del amortiguador delantero izquierdo cambiada|"
        "Termostato sustituido, circuito de refrigeración purgado|"
        "Rodamiento de rueda trasero izquierdo sustituido"
    ),
    order_kinds=[
        ("Orden de reparación", "OR-{year}-{d5}"),
        ("N.º de factura", "FAC-{year}-{d5}"),
        ("Pedido", "45{d8}"),
        ("Ticket", "#INC-{d6}"),
        ("Albarán", "ALB {d7}"),
        ("Presupuesto", "PRE-{d4}/{yy}"),
        ("N.º de expediente", "{C2}{d6}"),
    ],
    text={
        "workshop": "{car}, motor {engine}, {mileage}. Memoria de averías: {obd1} y {obd2}. {job}. {pp} {part}.",
        "workshop_plate": [" Matrícula {plate}.", " Matrícula del vehículo: {plate}."],
        "order_intro": "por favor, pedid las siguientes piezas para el {car} ({engine}):\n",
        "order_line_qty": "- {pp} {part}, 2 unidades\n",
        "order_line": "- {pp} {part}\n",
        "order_oil": "- 5 l {grade} según {spec}\n",
        "order_ref": "{kind}: {ref}, fecha de entrega {date}.",
        "tyres": "Cambio de neumáticos {car}: 4x {tyre} de invierno, DOT {dot}, profundidad del dibujo {depth} mm. Guardados en la estantería {shelf}, {price}.",
        "tyres_pickup": " Los recoge {first} {last}, tel. {phone}.",
        "invoice": "{kind} {ref} del {date}\nLínea 1: {job}, {price}\nLínea 2: {pp} {part}, {price2}\nPagadero antes del {date2}.",
        "invoice_to": [
            "Facturar a: {first} {last}, {street} {num}, {zip} {city}\n",
            "Facturar a: {first} {last}, {street}, {num}, {zip} {city}\n",
        ],
        "appointment": "Su cita para la revisión queda confirmada el {weekday}, {date}, a las {time}. La ITV vence en {due}. Por favor, traiga el libro de mantenimiento (última revisión a los {mileage}).",
        "appointment_plate": " Su vehículo con matrícula {plate} está registrado en nuestro taller.",
        "ticket": "Ticket #INC-{inc}: falló la actualización a la versión {version}, código de error 0x{hex}. Equipo {serial}, build {build}. Referencia: {kind} {ref}. Próximo intento el {date}.",
        "ticket_reporter": " Notificado por {first} {last} ({email}).",
        "contact": [
            "Titular: {first} {last}, nacido el {dob}\nMatrícula: {plate}\nVehículo: {car}, código de motor {engine}\n{kind}: {ref}, recepción el {date}",
            "Titular: {first} {last}, nacida el {dob}\nMatrícula: {plate}\nVehículo: {car}, código de motor {engine}\n{kind}: {ref}, recepción el {date}",
            "Cliente: {first} {last}, fecha de nacimiento: {dob}\nMatrícula: {plate}\nVehículo: {car}, código de motor {engine}\n{kind}: {ref}, recepción el {date}",
        ],
    },
)

DUTCH = Locale(
    language="Dutch",
    country="Netherlands",
    code="nl",
    first_names=_l(
        "Daan|Emma|Sem|Julia|Lucas|Mila|Levi|Tess|Finn|Sophie|Milan|Bram|Fenna|Jesse|"
        "Sanne|Thijs|Lotte|Ruben|Anouk|Jeroen|Marieke|Pieter|Femke|Willem|Ilse|Koen|"
        "Annelies|Mohamed|Fatima|Yusuf|Hatice|Rajesh|Shanti|Jan-Willem|Anne-Fleur|"
        "Gijs|Joost|Maartje|Sietske"
    ),
    surnames=_l(
        "de Jong|Jansen|de Vries|van den Berg|van Dijk|Bakker|Janssen|Visser|Smit|"
        "Meijer|de Boer|Mulder|de Groot|Bos|Vos|Peters|Hendriks|van Leeuwen|Dekker|"
        "Brouwer|de Wit|Dijkstra|Smits|de Graaf|van der Meer|van der Linden|Kok|"
        "Jacobs|de Haan|Vermeulen|Postma|Hoekstra|El Idrissi|Sedney|Kartosen|"
        "van 't Hof|Huisman|Wouters"
    ),
    streets=_l(
        "Dorpsstraat|Kerkstraat|Stationsweg|Molenweg|Schoolstraat|Julianastraat|"
        "Beatrixlaan|Wilhelminastraat|Prinsengracht|Industrieweg|Nieuwstraat|Markt|"
        "Lindelaan|Esdoornlaan|Havenkade|Burgemeester de Withstraat"
    ),
    cities=_pairs(
        "1012 {L2}=Amsterdam|3011 {L2}=Rotterdam|2511 {L2}=Den Haag|3511 {L2}=Utrecht|"
        "5611 {L2}=Eindhoven|9711 {L2}=Groningen|5038 {L2}=Tilburg|1311 {L2}=Almere|"
        "4811 {L2}=Breda|6511 {L2}=Nijmegen|7511 {L2}=Enschede|2011 {L2}=Haarlem|"
        "6811 {L2}=Arnhem|8011 {L2}=Zwolle|6211 {L2}=Maastricht|2311 {L2}=Leiden|"
        "3811 {L2}=Amersfoort|7311 {L2}=Apeldoorn|5211 {L2}='s-Hertogenbosch|"
        "8911 {L2}=Leeuwarden|1811 {L2}=Alkmaar|3311 {L2}=Dordrecht"
    ),
    num_suffixes=["", "", "", "a", "b", "-2"],
    email_domains=_l("ziggo.nl|kpnmail.nl|outlook.com|gmail.com|hotmail.nl|example.nl"),
    phone_formats=_l(
        "06-{d8}|06 {d8}|06 {d2} {d2} {d2} {d2}|+31 6 {d8}|020-{D7}|010 {D3} {d4}|"
        "030-{D7}|0{D2}-{D7}"
    ),
    plate=_plate_nl,
    months=_l(
        "januari|februari|maart|april|mei|juni|juli|augustus|september|oktober|"
        "november|december"
    ),
    weekdays=_l("maandag|dinsdag|woensdag|donderdag|vrijdag"),
    date_formats=["{d:02d}-{m:02d}-{y}", "{d} {month} {y}", "{d:02d}-{m:02d}"],
    dob_formats=["{d:02d}-{m:02d}-{y}", "{d} {month} {y}"],
    due_formats=["{month} {y}", "{m:02d}-{y}"],
    time_format="{h}.{mm} uur",
    thousands=".",
    price_format="€ {amount},{cents}",
    mileage_format="{n} km",
    greetings_named=_l(
        "Beste {first},|Hoi {first},|Hallo {first},|Geachte heer {last},|"
        "Geachte mevrouw {last},|Beste meneer {last},|Beste mevrouw {last},"
    ),
    greetings_plain=_l(
        "Beste allemaal,|Geachte heer/mevrouw,|Goedemorgen,|Hoi team,|Hallo,"
    ),
    signoffs=_l(
        "Met vriendelijke groet|Groeten|Groetjes|Mvg|Hartelijke groet|"
        "Vriendelijke groeten|Met vriendelijke groeten"
    ),
    team_signoffs=_l("Het werkplaatsteam|Klantenservice|Afdeling onderdelen"),
    roles=_l(
        "Serviceadviseur|Werkplaatschef|Magazijn|Klantenservice|Planning|Administratie|"
    ),
    part_prefixes=_l("OE-nr.|Onderdeelnummer|Art.nr.|Bestelnr.|Onderdeel"),
    workshop_jobs=_l(
        "Remblokken achter vervangen|Distributieriem en waterpomp vervangen|"
        "Olie en oliefilter ververst|Airco bijgevuld (R1234yf, 480 g)|"
        "Stabilisatorstang rechtsvoor vervangen|EGR-klep gereinigd|"
        "Gloeibougies 1-4 vervangen|Accu gecontroleerd, ruststroom 35 mA|"
        "Uitlijning gecontroleerd en afgesteld|Veerpootlager linksvoor vervangen|"
        "Thermostaat vervangen, koelsysteem ontlucht|Wiellager linksachter vervangen"
    ),
    order_kinds=[
        ("Werkorder", "WO-{year}-{d5}"),
        ("Factuurnummer", "F-{year}-{d5}"),
        ("Inkoopordernummer", "45{d8}"),
        ("Ticket", "#INC-{d6}"),
        ("Pakbon", "PB {d7}"),
        ("Offerte", "OF-{d4}/{yy}"),
        ("Dossiernummer", "{C2}{d6}"),
    ],
    text={
        "workshop": "{car}, motor {engine}, {mileage}. Foutgeheugen: {obd1} en {obd2}. {job}. {pp} {part}.",
        "workshop_plate": [" Kenteken {plate}.", " Kenteken: {plate}."],
        "order_intro": "graag de volgende onderdelen bestellen voor de {car} ({engine}):\n",
        "order_line_qty": "- {pp} {part}, 2 stuks\n",
        "order_line": "- {pp} {part}\n",
        "order_oil": "- 5 l {grade} volgens {spec}\n",
        "order_ref": "{kind}: {ref}, leverdatum {date}.",
        "tyres": "Bandenwissel {car}: 4x {tyre} winterbanden, DOT {dot}, profieldiepte {depth} mm. Opslag rek {shelf}, {price}.",
        "tyres_pickup": " Wordt opgehaald door {first} {last}, tel. {phone}.",
        "invoice": "{kind} {ref} van {date}\nRegel 1: {job}, {price}\nRegel 2: {pp} {part}, {price2}\nTe betalen vóór {date2}.",
        "invoice_to": "Factuuradres: {first} {last}, {street} {num}, {zip} {city}\n",
        "appointment": "Uw afspraak voor de onderhoudsbeurt is bevestigd op {weekday} {date} om {time}. De APK verloopt in {due}. Neem a.u.b. het onderhoudsboekje mee (laatste beurt bij {mileage}).",
        "appointment_plate": " Uw auto met kenteken {plate} staat bij ons geregistreerd.",
        "ticket": "Ticket #INC-{inc}: update naar versie {version} mislukt, foutcode 0x{hex}. Apparaat {serial}, build {build}. Betreft: {kind} {ref}. Nieuwe poging op {date}.",
        "ticket_reporter": " Gemeld door {first} {last} ({email}).",
        "contact": [
            "Eigenaar: {first} {last}, geboren op {dob}\nKenteken: {plate}\nVoertuig: {car}, motorcode {engine}\n{kind}: {ref}, binnengebracht op {date}",
            "Klant: {first} {last}, geboortedatum {dob}\nKenteken: {plate}\nVoertuig: {car}, motorcode {engine}\n{kind}: {ref}, binnengebracht op {date}",
        ],
    },
)

DANISH = Locale(
    language="Danish",
    country="Denmark",
    code="da",
    first_names=_l(
        "William|Ida|Noah|Emma|Oscar|Alma|Lucas|Freja|Carl|Clara|Victor|Agnes|Magnus|"
        "Ella|Frederik|Karla|Mikkel|Mette|Rasmus|Signe|Anders|Lone|Henrik|Birgitte|"
        "Morten|Hanne|Kasper|Pernille|Mads|Line|Camilla|Sigrid|Asger|Maja|Hassan|"
        "Amina|Jakob|Nanna|Jørgen|Kirsten|Poul|Inger|Ann-Sofie"
    ),
    surnames=_l(
        "Nielsen|Jensen|Hansen|Pedersen|Andersen|Christensen|Larsen|Sørensen|"
        "Rasmussen|Jørgensen|Petersen|Madsen|Kristensen|Olsen|Thomsen|Christiansen|"
        "Poulsen|Johansen|Møller|Mortensen|Knudsen|Jakobsen|Mikkelsen|Lauridsen|"
        "Frederiksen|Holm|Østergaard|Bach|Dahl|Skov|Lund|Kjær|Bjerregaard|"
        "Vestergaard|Fogh|Abdi|Hussein|Demir"
    ),
    streets=_l(
        "Vestergade|Østergade|Nørregade|Søndergade|Kirkevej|Skolevej|Stationsvej|"
        "Møllevej|Bygaden|Strandvejen|Industrivej|Engvej|Birkevej|Havnegade|Ringvej|"
        "Jernbanegade"
    ),
    cities=_pairs(
        "1050=København K|8000=Aarhus C|5000=Odense C|9000=Aalborg|6700=Esbjerg|"
        "8900=Randers|6000=Kolding|8700=Horsens|7100=Vejle|2610=Rødovre|4000=Roskilde|"
        "7400=Herning|3400=Hillerød|8600=Silkeborg|4700=Næstved|3000=Helsingør|"
        "4600=Køge|6400=Sønderborg|7500=Holstebro|8800=Viborg|2800=Kongens Lyngby|"
        "5700=Svendborg"
    ),
    num_suffixes=["", "", "", "A", "B"],
    email_domains=_l("gmail.com|mail.dk|outlook.dk|hotmail.com|jubii.dk|example.dk"),
    phone_formats=_l(
        "2{d1} {d2} {d2} {d2}|4{d1} {d2} {d2} {d2}|6{d1} {d2} {d2} {d2}|"
        "+45 {D2} {d2} {d2} {d2}|{D4} {d4}|3{d1} {d2} {d2} {d2}"
    ),
    plate=_plate_dk,
    months=_l(
        "januar|februar|marts|april|maj|juni|juli|august|september|oktober|november|"
        "december"
    ),
    weekdays=_l("mandag|tirsdag|onsdag|torsdag|fredag"),
    date_formats=[
        "{d:02d}.{m:02d}.{y}",
        "{d}. {month} {y}",
        "{d:02d}-{m:02d}-{y}",
        "{d}/{m}",
    ],
    dob_formats=["{d:02d}.{m:02d}.{y}", "{d}. {month} {y}", "{d:02d}-{m:02d}-{y}"],
    due_formats=["{month} {y}", "{m:02d}/{y}"],
    time_format="kl. {h}.{mm}",
    thousands=".",
    price_format="{amount},{cents} kr.",
    mileage_format="{n} km",
    greetings_named=_l(
        "Hej {first},|Kære {first},|Hej {first}!|Kære hr. {last},|Kære fru {last},"
    ),
    greetings_plain=_l("Hej alle,|Til rette vedkommende,|Godmorgen,|Hej team,|Hej,"),
    signoffs=_l(
        "Med venlig hilsen|Venlig hilsen|Mvh|Vh|Hilsen|Bedste hilsner|De bedste hilsner"
    ),
    team_signoffs=_l("Værkstedet|Kundeservice|Reservedelsafdelingen"),
    roles=_l(
        "Servicekonsulent|Værkstedschef|Reservedele|Kundeservice|Planlægning|"
        "Bogholderi|"
    ),
    part_prefixes=_l("OE-nr.|Reservedelsnr.|Varenr.|Bestillingsnr.|Reservedel"),
    workshop_jobs=_l(
        "Bremseklodser bag udskiftet|Tandrem og vandpumpe skiftet|"
        "Olie og oliefilter skiftet|Aircondition fyldt op (R1234yf, 480 g)|"
        "Stabilisatorstag højre for udskiftet|EGR-ventil renset|"
        "Gløderør 1-4 udskiftet|Batteri kontrolleret, hvilestrøm 35 mA|"
        "Hjulafstilling kontrolleret og justeret|Toplejer venstre for udskiftet|"
        "Termostat udskiftet, kølesystem udluftet|Hjulleje venstre bag udskiftet"
    ),
    order_kinds=[
        ("Arbejdsordre", "AO-{year}-{d5}"),
        ("Fakturanr.", "FAK-{year}-{d5}"),
        ("Indkøbsordre", "45{d8}"),
        ("Sag", "#INC-{d6}"),
        ("Følgeseddel", "FS {d7}"),
        ("Tilbud", "TB-{d4}/{yy}"),
        ("Sagsnummer", "{C2}{d6}"),
    ],
    text={
        "workshop": "{car}, motor {engine}, {mileage}. Fejlkoder: {obd1} og {obd2}. {job}. {pp} {part}.",
        "workshop_plate": [" Nummerplade {plate}.", " Reg.nr. {plate}."],
        "order_intro": "bestil venligst følgende dele til {car} ({engine}):\n",
        "order_line_qty": "- {pp} {part}, 2 stk.\n",
        "order_line": "- {pp} {part}\n",
        "order_oil": "- 5 l {grade} iht. {spec}\n",
        "order_ref": "{kind}: {ref}, levering {date}.",
        "tyres": "Dækskifte {car}: 4x {tyre} vinterdæk, DOT {dot}, mønsterdybde {depth} mm. Opbevaring reol {shelf}, {price}.",
        "tyres_pickup": " Afhentes af {first} {last}, tlf. {phone}.",
        "invoice": "{kind} {ref} af {date}\nLinje 1: {job}, {price}\nLinje 2: {pp} {part}, {price2}\nBetales senest {date2}.",
        "invoice_to": "Faktureres til: {first} {last}, {street} {num}, {zip} {city}\n",
        "appointment": "Din tid til service er bekræftet {weekday} den {date} {time}. Næste syn skal ske senest {due}. Tag venligst servicebogen med (sidste service ved {mileage}).",
        "appointment_plate": " Din bil med nummerpladen {plate} er registreret hos os.",
        "ticket": "Sag #INC-{inc}: opdatering til version {version} mislykkedes, fejlkode 0x{hex}. Enhed {serial}, build {build}. Vedr.: {kind} {ref}. Nyt forsøg den {date}.",
        "ticket_reporter": " Indmeldt af {first} {last} ({email}).",
        "contact": [
            "Ejer: {first} {last}, født den {dob}\nNummerplade: {plate}\nKøretøj: {car}, motorkode {engine}\n{kind}: {ref}, indleveret den {date}",
            "Kunde: {first} {last}, fødselsdato {dob}\nReg.nr.: {plate}\nKøretøj: {car}, motorkode {engine}\n{kind}: {ref}, indleveret den {date}",
        ],
    },
)

LOCALES: dict[str, Locale] = {
    loc.language: loc for loc in (GERMAN, ENGLISH, FRENCH, SPANISH, DUTCH, DANISH)
}

# Words for person records (tables, lists, exports) per locale code. Records put
# a birth date next to a name *without* a birth word, as real exports and lists
# do; mix v2 had no such rows, and the model learned "no birth word -> no birth
# date" in every language. "event" keys introduce dates that are not PII.
RECORD_WORDS: dict[str, dict[str, list[str]]] = {
    "de": {
        "name": _l("Name|Vor- und Nachname|Person"),
        "city": _l("Ort|Wohnort|Stadt"),
        "phone": _l("Telefon|Tel.|Mobil"),
        "person": _l("Teilnehmer|Teilnehmerin|Mitglied|Mitarbeiter|Kunde|Person"),
        "list": _l(
            "Teilnehmerliste|Mitgliederliste|Personalliste|Kundenliste|Anmeldungen"
        ),
        "event": _l("Angelegt am|Letzte Änderung|Eintritt|Anmeldung am|Stand"),
        "dob_key": _l("geb.|Geb.|GebDat|geb_datum|dob"),
    },
    "en": {
        "name": _l("Name|Full name|Person"),
        "city": _l("City|Town|Location"),
        "phone": _l("Phone|Tel.|Mobile"),
        "person": _l("Participant|Member|Applicant|Employee|Customer|Student|Party"),
        "list": _l("Participant list|Members|Staff list|Customer list|Registrations"),
        "event": _l("Registered|Last updated|Joined|Created|Signed up"),
        "dob_key": _l("DOB|D.O.B.|dob|bday|student_dob"),
    },
    "fr": {
        "name": _l("Nom|Nom complet|Personne"),
        "city": _l("Ville|Commune|Localité"),
        "phone": _l("Téléphone|Tél.|Portable"),
        "person": _l("Participant|Participante|Membre|Salarié|Client|Élève"),
        "list": _l("Liste des participants|Membres|Liste du personnel|Inscriptions"),
        "event": _l("Créé le|Mis à jour le|Inscrit le|Entrée|Date d'inscription"),
        "dob_key": _l("DDN|ddn|date_naiss|dob"),
    },
    "es": {
        "name": _l("Nombre|Nombre completo|Persona"),
        "city": _l("Ciudad|Localidad|Municipio"),
        "phone": _l("Teléfono|Tel.|Móvil"),
        "person": _l("Participante|Socio|Socia|Empleado|Cliente|Alumno"),
        "list": _l("Lista de participantes|Socios|Plantilla|Inscripciones"),
        "event": _l("Alta|Última modificación|Fecha de alta|Inscrito el|Creado"),
        "dob_key": _l("F. nac.|fnac|fecha_nac|dob"),
    },
    "nl": {
        "name": _l("Naam|Volledige naam|Persoon"),
        "city": _l("Plaats|Woonplaats|Stad"),
        "phone": _l("Telefoon|Tel.|Mobiel"),
        "person": _l("Deelnemer|Lid|Medewerker|Klant|Leerling"),
        "list": _l("Deelnemerslijst|Ledenlijst|Personeelslijst|Inschrijvingen"),
        "event": _l("Aangemaakt|Laatst gewijzigd|Lid sinds|Ingeschreven op|Stand"),
        "dob_key": _l("geb.|Geb.datum|geb_datum|dob"),
    },
    "da": {
        "name": _l("Navn|Fulde navn|Person"),
        "city": _l("By|Bopæl|Postdistrikt"),
        "phone": _l("Telefon|Tlf.|Mobil"),
        "person": _l("Deltager|Medlem|Medarbejder|Kunde|Elev"),
        "list": _l("Deltagerliste|Medlemsliste|Personaleliste|Tilmeldinger"),
        "event": _l("Oprettet|Sidst ændret|Medlem siden|Tilmeldt|Status pr."),
        "dob_key": _l("f.|Fødselsdag|fdato|dob"),
    },
}
EN_ORDINALS = {1: "st", 2: "nd", 3: "rd", 21: "st", 22: "nd", 23: "rd"}

# ---------------------------------------------------------------------------
# Templates: each returns a list of (text, label) parts
# ---------------------------------------------------------------------------

_FORMATTER = string.Formatter()


def _fill(tpl: str, values: dict[str, str]) -> list[Part]:
    """Split ``tpl`` on its ``{fields}``; fields in PII_FIELDS get their label."""
    parts: list[Part] = []
    for literal, field, _, _ in _FORMATTER.parse(tpl):
        if literal.startswith(".") and parts and parts[-1][0].endswith("."):
            literal = literal[1:]  # "am 06.08." + "." -> one full stop
        if literal:
            parts.append((literal, None))
        if field is not None:
            parts.append((values[field], PII_FIELDS.get(field)))
    return parts


def _values(g: _Gen) -> dict[str, str]:
    """Fresh values for every template field (PII and non-PII)."""
    loc = g.loc
    first, last = g.pick(loc.first_names), g.pick(loc.surnames)
    kind, ref = g.order_ref()
    grade, spec = g.pick(OIL_SPECS)
    return {
        "car": g.pick(CAR_MODELS),
        "engine": g.pick(ENGINE_CODES),
        "mileage": g.mileage(),
        "obd1": g.obd_code(),
        "obd2": g.obd_code(),
        "job": g.pick(loc.workshop_jobs),
        "pp": g.pick(loc.part_prefixes),
        "part": g.part_number(),
        "kind": kind,
        "ref": ref,
        "date": g.date(),
        "date2": g.date(),
        "due": g.due(),
        "weekday": g.pick(loc.weekdays),
        "time": g.time(),
        "tyre": g.tyre(),
        "dot": g.digits(4),
        "depth": str(g.r.randint(3, 8)),
        "shelf": f"{g.letters(1)}-{g.r.randint(1, 40)}",
        "price": g.price(),
        "price2": g.price(),
        "grade": grade,
        "spec": spec,
        "inc": g.digits(6),
        "version": g.version(),
        "hex": g.digits(4) + g.letters(2),
        "serial": g.serial(),
        "build": g.digits(8),
        "first": first,
        "last": last,
        "plate": g.plate(),
        "phone": g.phone(),
        "email": g.email(first, last),
        "dob": g.dob(),
        **g.address(),
    }


def _part_line(g: _Gen, key: str) -> list[Part]:
    return _fill(
        g.text(key), {"pp": g.pick(g.loc.part_prefixes), "part": g.part_number()}
    )


def _greeting(g: _Gen, v: dict[str, str], named: bool) -> list[Part]:
    if not named:
        return [(g.pick(g.loc.greetings_plain) + "\n\n", None)]
    return _fill(g.pick(g.loc.greetings_named), v) + [("\n\n", None)]


def _signoff(g: _Gen, v: dict[str, str], with_name: bool) -> list[Part]:
    parts: list[Part] = [("\n\n" + g.pick(g.loc.signoffs) + "\n", None)]
    if not with_name:
        parts.append((g.pick(g.loc.team_signoffs), None))
        return parts
    if g.r.random() < 0.25:
        parts.append((v["first"], "FIRSTNAME"))
    else:
        parts += [(v["first"], "FIRSTNAME"), (" ", None), (v["last"], "SURNAME")]
    role = g.pick(g.loc.roles)
    if role:
        parts.append(("\n" + role, None))
    return parts


def t_workshop_note(g: _Gen, pii: bool) -> list[Part]:
    v = _values(g)
    parts = _fill(g.text("workshop"), v)
    if pii:
        parts += _fill(g.text("workshop_plate"), v)
    return parts


def t_parts_order(g: _Gen, pii: bool) -> list[Part]:
    v = _values(g)
    body = (
        _fill(g.text("order_intro"), v)
        + _part_line(g, "order_line_qty")
        + _part_line(g, "order_line")
        + _fill(g.text("order_oil"), v)
        + _fill(g.text("order_ref"), v)
    )
    return _greeting(g, v, pii) + body + _signoff(g, _values(g), pii)


def t_tyres(g: _Gen, pii: bool) -> list[Part]:
    v = _values(g)
    parts = _fill(g.text("tyres"), v)
    if pii:
        parts += _fill(g.text("tyres_pickup"), v)
    return parts


def t_invoice(g: _Gen, pii: bool) -> list[Part]:
    v = _values(g)
    parts = _fill(g.text("invoice"), v)
    if pii:
        parts = _fill(g.text("invoice_to"), v) + parts
    return parts


def t_appointment(g: _Gen, pii: bool) -> list[Part]:
    v = _values(g)
    body = _fill(g.text("appointment"), v)
    if pii and g.r.random() < 0.5:
        body += _fill(g.text("appointment_plate"), v)
    # The greeting addresses the customer; the sign-off is a staff member.
    return _greeting(g, v, pii) + body + _signoff(g, _values(g), pii)


def t_it_ticket(g: _Gen, pii: bool) -> list[Part]:
    v = _values(g)
    parts = _fill(g.text("ticket"), v)
    if pii:
        parts += _fill(g.text("ticket_reporter"), v)
    return parts


def t_customer_contact(g: _Gen, pii: bool) -> list[Part]:
    """Always has PII: birth date and plate in context next to non-PII codes."""
    return _fill(g.text("contact"), _values(g))


def _person(g: _Gen) -> dict[str, str]:
    _, city = g.pick(g.loc.cities)
    first = g.pick(g.loc.first_names)
    last = g.pick(g.loc.surnames)
    return {
        "first": first,
        "last": last,
        "dob": g.record_dob(),
        "city": city,
        "phone": g.phone(),
        "email": g.email(first, last),
    }


def _name_parts(p: dict[str, str]) -> list[Part]:
    return [(p["first"], "FIRSTNAME"), (" ", None), (p["last"], "SURNAME")]


def t_person_record(g: _Gen, pii: bool) -> list[Part]:
    """Always has PII: birth dates in a table, CSV, list, form or JSON/XML record,
    with no birth word in front of them. Some records also carry an unlabelled
    event date (created, joined, last updated) to contrast with."""
    words = RECORD_WORDS[g.loc.code]
    people = [_person(g) for _ in range(g.r.randint(1, 4))]
    style = g.r.randrange(6)
    parts: list[Part] = []

    if style == 0:  # Markdown table
        if g.r.random() < 0.6:
            dob_col = g.pick(words["dob_key"] + ["", "", ""])
            parts.append(
                (
                    f"| # | {g.pick(words['name'])} | {dob_col} | {g.pick(words['city'])} |\n"
                    "|---|---|---|---|\n",
                    None,
                )
            )
        for i, p in enumerate(people, 1):
            label = f"{g.pick(words['person'])} {string.ascii_uppercase[i - 1]}"
            parts.append((f"| {label if g.r.random() < 0.5 else i} | ", None))
            parts += _name_parts(p)
            parts += [(" | ", None), (p["dob"], "DATEOFBIRTH"), (" | ", None)]
            parts += [(p["city"], "CITY"), (" |\n", None)]
    elif style == 1:  # CSV export
        sep = g.pick([",", ";", "\t"])
        if g.r.random() < 0.6:
            parts.append(
                (
                    sep.join(["id", "name", g.pick(["date", "d", "dob", ""]), "city"])
                    + "\n",
                    None,
                )
            )
        for i, p in enumerate(people, 1):
            parts.append((f"{i}{sep}", None))
            parts += _name_parts(p)
            parts += [(sep, None), (p["dob"], "DATEOFBIRTH"), (sep, None)]
            parts += [(p["city"], "CITY"), ("\n", None)]
    elif style == 2:  # Bulleted list
        parts.append((g.pick(words["list"]) + ":\n", None))
        form = g.r.randrange(3)
        for i, p in enumerate(people, 1):
            if form == 0:  # - Name, date, city
                parts.append(("- ", None))
                parts += _name_parts(p)
                parts += [(", ", None), (p["dob"], "DATEOFBIRTH"), (", ", None)]
                parts += [(p["city"], "CITY"), ("\n", None)]
            elif form == 1:  # - Name (date)
                parts.append(("- ", None))
                parts += _name_parts(p)
                parts += [(" (", None), (p["dob"], "DATEOFBIRTH"), (")\n", None)]
            else:  # Participant B: Name - date
                label = f"{g.pick(words['person'])} {string.ascii_uppercase[i - 1]}"
                parts.append((f"{i}. {label}: ", None))
                parts += _name_parts(p)
                parts += [(" - ", None), (p["dob"], "DATEOFBIRTH"), ("\n", None)]
    elif style == 3:  # Key-value form, date on its own line under the name
        for p in people[:2]:
            parts.append((f"{g.pick(words['name'])}: ", None))
            parts += _name_parts(p)
            parts += [("\n", None), (p["dob"], "DATEOFBIRTH"), ("\n", None)]
            parts += [
                (f"{g.pick(words['phone'])}: ", None),
                (p["phone"], "PHONENUMBER"),
            ]
            if g.r.random() < 0.5:
                parts.append((f"\n{g.pick(words['event'])}: {g.date()}", None))
            parts.append(("\n\n", None))
    elif style == 4:  # JSON
        parts.append(("[\n", None))
        for i, p in enumerate(people):
            key = g.pick(words["dob_key"] + ["dob", "d", "date"])
            parts.append(('  {"name": "', None))
            parts += _name_parts(p)
            parts += [(f'", "{key}": "', None), (p["dob"], "DATEOFBIRTH")]
            parts += [('", "email": "', None), (p["email"], "EMAIL"), ('"', None)]
            if g.r.random() < 0.5:
                parts.append(
                    (
                        f', "{g.pick(["created", "updated", "joined"])}": "{g.date()}"',
                        None,
                    )
                )
            parts.append(("}" + (",\n" if i < len(people) - 1 else "\n"), None))
        parts.append(("]", None))
    else:  # XML
        tag = g.pick(["dob", "Dob", "d", "date"])
        parts.append(("<people>\n", None))
        for p in people:
            parts.append(("  <person><name>", None))
            parts += _name_parts(p)
            parts += [(f"</name><{tag}>", None), (p["dob"], "DATEOFBIRTH")]
            parts += [(f"</{tag}><city>", None), (p["city"], "CITY")]
            parts.append(("</city>", None))
            if g.r.random() < 0.5:
                parts.append((f"<created>{g.date()}</created>", None))
            parts.append(("</person>\n", None))
        parts.append(("</people>", None))

    # Strip trailing whitespace so the row does not end on blank lines.
    while parts and parts[-1][1] is None and not parts[-1][0].strip():
        parts.pop()
    if parts and parts[-1][1] is None:
        parts[-1] = (parts[-1][0].rstrip(), None)
    return parts


TEMPLATES: list[tuple[Callable[[_Gen, bool], list[Part]], bool]] = [
    # (template, can be PII-free)
    (t_workshop_note, True),
    (t_parts_order, True),
    (t_tyres, True),
    (t_invoice, True),
    (t_appointment, True),
    (t_it_ticket, True),
    (t_customer_contact, False),
    (t_person_record, False),
]
# Six of eight templates can be PII-free; 0.66 of those makes about half of
# all rows PII-free.
DEFAULT_PII_FREE_SHARE = 0.66


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


def generate(
    num_rows: int,
    seed: int = 13,
    pii_free_share: float = DEFAULT_PII_FREE_SHARE,
    language: str = "German",
) -> list[dict]:
    """Return ``num_rows`` samples in ``language`` in the internal training format.

    ``language`` is one of ``LOCALES`` (the Kiji language names). PII-free rows
    carry ``"pii_free": True`` so the training loader keeps them.
    """
    loc = LOCALES[language]
    g = _Gen(random.Random(f"{seed}:{loc.code}"), loc)
    rows, seen = [], set()
    attempts = 0
    while len(rows) < num_rows and attempts < num_rows * 20:
        attempts += 1
        template, can_be_free = g.pick(TEMPLATES)
        pii = not (can_be_free and g.r.random() < pii_free_share)
        text, mask = _assemble(template(g, pii))
        if text in seen or _PROBE_RE.search(text):
            continue
        seen.add(text)
        rows.append(
            {
                "text": text,
                "privacy_mask": mask,
                "coreferences": [],
                "language": loc.language,
                "country": loc.country,
                "source": f"synthetic_negatives_{loc.code}",
                "pii_free": not mask,
            }
        )
    return rows
