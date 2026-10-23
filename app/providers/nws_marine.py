"""NWS marine text products: coastal waters forecasts, surf zone forecasts, advisories.

Sources (all free, public domain NWS products)
----------------------------------------------
* Latest products: NWS API ``/products/types/{CWF|SRF}/locations/OKX``
  (https://www.weather.gov/documentation/services-web-api). CWF falls back to
  the per-zone text files on tgftp.nws.noaa.gov if the API is down.
* Archived products for past dates: Iowa Environmental Mesonet AFOS archive
  (https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py), a public
  mirror of the NWS text feed.
* Past advisories (Small Craft, Gale, High Surf, Rip Current …): IEM VTEC
  archive by zone (https://mesonet.agron.iastate.edu/json/vtec_events_byugc.php).

What is used
------------
CWF "Seas/Waves X to Y ft" is the forecast significant wave height (combined
seas) for a zone; "Wave Detail: E 6 ft at 9 seconds" lists the main wave
components. SRF "Surf Height" is the NWS forecaster's breaking-surf forecast for
ocean beaches (issued in the beach season), used only as independent validation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from app.core.http import FetchError, HttpClient

NWS = "https://api.weather.gov"
TGFTP_CWF = "https://tgftp.nws.noaa.gov/data/forecasts/marine/coastal/an/{zone}.txt"
IEM_AFOS = "https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py"
IEM_VTEC = "https://mesonet.agron.iastate.edu/json/vtec_events_byugc.php"
OFFICE = "OKX"
LOCAL_TZ = ZoneInfo("America/New_York")

COMPASS16 = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
DAYS = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
DAY_FULL = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]
TTL_LATEST = 1800
TTL_ARCHIVE = 30 * 86400


# ---------------------------------------------------------------- dataclasses
@dataclass
class WaveComponent:
    dir_deg: float | None
    height_ft: float
    period_s: float


@dataclass
class MarinePeriod:
    name: str
    start: pd.Timestamp            # UTC
    end: pd.Timestamp              # UTC
    seas_lo_ft: float | None = None
    seas_hi_ft: float | None = None
    seas_occ_ft: float | None = None   # "occasionally to N ft"
    kind: str = "seas"                 # seas (ocean) | waves (bays / Sound)
    components: list[WaveComponent] = field(default_factory=list)
    text: str = ""


@dataclass
class ZoneForecast:
    zone: str
    name: str
    issued: pd.Timestamp           # UTC
    expires: pd.Timestamp | None
    headlines: list[str]
    periods: list[MarinePeriod]
    product: str = "CWF"

    def stale(self, now: pd.Timestamp | None = None) -> bool:
        now = now or pd.Timestamp.now(tz="UTC")
        if self.expires is not None:
            return now > self.expires
        return now - self.issued > pd.Timedelta(hours=14)


# ------------------------------------------------------------------- parsing
_ISSUED = re.compile(r"^(\d{3,4}) (AM|PM) (EDT|EST) \w{3} (\w{3}) (\d{1,2}) (\d{4})\s*$", re.M)
_UGC = re.compile(r"^((?:[A-Z]{2}Z\d{3}[->])(?:[\d>\-]|[A-Z]{2}Z)*?)(\d{6})-\s*$", re.M)
_PERIOD = re.compile(r"^\.([A-Z][A-Z ]*?)\.\.\.(.*?)(?=^\.[A-Z][A-Z ]*?\.\.\.|\Z)", re.M | re.S)
_SEAS = re.compile(r"\b(Seas|Waves)\s+(?:(\d+)\s+to\s+(\d+)\s+ft|around\s+(\d+)\s+ft|(\d+)\s+ft\s+or\s+less|less\s+than\s+(\d+)\s+ft)"
                   r"(?:,?\s+occasionally\s+to\s+(\d+)\s+ft)?", re.I)
_DETAIL = re.compile(r"Wave Detail:(.*?)(?:\.\s|\.$)", re.S)
_COMP = re.compile(r"\b(N|NNE|NE|ENE|E|ESE|SE|SSE|S|SSW|SW|WSW|W|WNW|NW|NNW)\s+(\d+)\s+ft\s+at\s+(\d+)\s+seconds")
_HEADLINE = re.compile(r"^\.\.\.(.+?)\.\.\.\s*$", re.M | re.S)


def parse_issued(text: str) -> pd.Timestamp | None:
    """'926 AM EDT Fri Oct 23 2026' → UTC timestamp."""
    m = _ISSUED.search(text)
    if not m:
        return None
    hhmm, ampm, _tz, mon, day, year = m.groups()
    hhmm = hhmm.zfill(4)
    hour, minute = int(hhmm[:2]) % 12 + (12 if ampm == "PM" else 0), int(hhmm[2:])
    local = datetime.strptime(f"{year} {mon} {day} {hour:02d}:{minute:02d}", "%Y %b %d %H:%M")
    return pd.Timestamp(local, tz=LOCAL_TZ).tz_convert("UTC")


def expand_ugc(line: str) -> list[str]:
    """'ANZ330-335>338-340-' → ['ANZ330', 'ANZ335', 'ANZ336', 'ANZ337', 'ANZ338', 'ANZ340']."""
    out, prefix = [], None
    for tok in [t for t in line.replace("\n", "").split("-") if t]:
        if len(tok) == 6 and tok.isdigit():
            continue                      # DDHHMM expiration
        m = re.match(r"^([A-Z]{2}Z)?(\d{3})(?:>(\d{3}))?$", tok)
        if not m:
            continue
        prefix = m.group(1) or prefix
        a = int(m.group(2))
        b = int(m.group(3)) if m.group(3) else a
        out += [f"{prefix}{n:03d}" for n in range(a, b + 1)]
    return out


def _ugc_expiry(line: str, issued: pd.Timestamp) -> pd.Timestamp | None:
    """UGC lines end with the product expiration as DDHHMM (UTC)."""
    m = re.search(r"(\d{2})(\d{2})(\d{2})-?\s*$", line.strip())
    if not m:
        return None
    d, h, mi = (int(x) for x in m.groups())
    base = issued.tz_convert("UTC")
    for month_shift in (0, 1):
        y, mo = base.year, base.month + month_shift
        if mo > 12:
            y, mo = y + 1, 1
        try:
            t = pd.Timestamp(year=y, month=mo, day=d, hour=h, minute=mi, tz="UTC")
        except ValueError:
            continue
        if t >= base - pd.Timedelta(hours=1):
            return t
    return None


def _is_night(name: str) -> bool:
    return any(w in name for w in ("NIGHT", "TONIGHT", "OVERNIGHT", "EVENING"))


def _next_boundary(t_local: pd.Timestamp, night: bool) -> pd.Timestamp:
    """Next 6 AM (night periods) or 6 PM (day periods) local wall-clock time after t_local."""
    hour = 6 if night else 18
    d = t_local.date()
    for add in range(3):
        dd = d + timedelta(days=add)
        cand = pd.Timestamp(datetime(dd.year, dd.month, dd.day, hour), tz=LOCAL_TZ)
        if cand > t_local:
            return cand
    return t_local + pd.Timedelta(hours=12)


def period_bounds(names: list[str], issued_utc: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Map consecutive NWS period names to UTC intervals (day ends 6 PM local, night ends 6 AM local)."""
    out = []
    cur = issued_utc.tz_convert(LOCAL_TZ)
    for name in names:
        end = _next_boundary(cur, _is_night(name))
        out.append((cur.tz_convert("UTC"), end.tz_convert("UTC")))
        cur = end
    return out


def _seas(text: str) -> tuple[float | None, float | None, float | None, str]:
    m = _SEAS.search(text)
    if not m:
        return None, None, None, "seas"
    kind = m.group(1).lower()
    lo = hi = None
    if m.group(2):
        lo, hi = float(m.group(2)), float(m.group(3))
    elif m.group(4):
        lo = hi = float(m.group(4))
    elif m.group(5):
        lo, hi = 0.0, float(m.group(5))
    elif m.group(6):
        lo, hi = 0.0, float(m.group(6))
    occ = float(m.group(7)) if m.group(7) else None
    return lo, hi, occ, kind


def _components(text: str) -> list[WaveComponent]:
    m = _DETAIL.search(text)
    if not m:
        return []
    return [WaveComponent(COMPASS16.index(d) * 22.5, float(h), float(p)) for d, h, p in _COMP.findall(m.group(1))]


def parse_cwf(text: str) -> dict[str, ZoneForecast]:
    """Parse a Coastal Waters Forecast product into zone forecasts keyed by zone id."""
    issued = parse_issued(text)
    if issued is None:
        return {}
    out: dict[str, ZoneForecast] = {}
    for block in text.split("$$"):
        um = _UGC.search(block)
        if not um:
            continue
        zones = expand_ugc(um.group(0))
        body = block[um.end():]
        lines = [ln for ln in body.strip().splitlines()]
        name = lines[0].rstrip("-").strip() if lines else ""
        b_issued = parse_issued(body) or issued
        heads = [re.sub(r"\s+", " ", h).strip() for h in _HEADLINE.findall(body)]
        pm = list(_PERIOD.finditer(body))
        names = [p.group(1).strip() for p in pm]
        bounds = period_bounds(names, b_issued) if names else []
        periods = []
        for p, (s, e) in zip(pm, bounds):
            txt = re.sub(r"\s+", " ", p.group(2)).strip()
            lo, hi, occ, kind = _seas(txt)
            periods.append(MarinePeriod(p.group(1).strip(), s, e, lo, hi, occ, kind, _components(txt), txt))
        exp = _ugc_expiry(um.group(0), b_issued)
        for z in zones:
            out[z] = ZoneForecast(z, name, b_issued, exp, heads, periods)
    return out


def zone_hourly(zf: ZoneForecast, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Step-function hourly series of the zone forecast (NaN outside its periods)."""
    cols = {k: np.full(len(index), np.nan) for k in
            ("nws_seas_lo_ft", "nws_seas_hi_ft", "nws_seas_occ_ft", "nws_c1_ft", "nws_c1_period_s", "nws_c1_dir",
             "nws_c2_ft", "nws_c2_period_s", "nws_c2_dir")}
    for p in zf.periods:
        m = (index >= p.start) & (index < p.end)
        for k, v in (("nws_seas_lo_ft", p.seas_lo_ft), ("nws_seas_hi_ft", p.seas_hi_ft), ("nws_seas_occ_ft", p.seas_occ_ft)):
            if v is not None:
                cols[k][m] = v
        for j, c in enumerate(p.components[:2], 1):
            cols[f"nws_c{j}_ft"][m] = c.height_ft
            cols[f"nws_c{j}_period_s"][m] = c.period_s
            cols[f"nws_c{j}_dir"][m] = c.dir_deg if c.dir_deg is not None else np.nan
    return pd.DataFrame(cols, index=index)


# --------------------------------------------------------------- SRF parsing
@dataclass
class SurfDay:
    zone: str
    day: date
    surf_lo_ft: float | None
    surf_hi_ft: float | None
    rip_risk: str | None
    issued: pd.Timestamp


_SRF_DAY = re.compile(r"^\.([A-Z ]+?)\.\.\.\s*$(.*?)(?=^\.[A-Z ]+?\.\.\.|\Z)", re.M | re.S)
_SURF_H = re.compile(r"Surf Height\.*\s*(?:(\d+)\s+to\s+(\d+)\s+feet|around\s+(\d+)\s+feet?|(\d+)\s+feet?\s+or\s+less|less\s+than\s+(\d+)\s+feet?)", re.I)
_RIP = re.compile(r"Rip Current Risk\.*\s*([A-Za-z]+)")


def _day_for(name: str, issued_local: datetime) -> date | None:
    n = name.strip().upper()
    if n in ("TODAY", "REST OF TODAY", "THIS AFTERNOON", "THIS MORNING"):
        return issued_local.date()
    for i, d in enumerate(DAY_FULL):
        if n.startswith(d):
            delta = (i - issued_local.weekday()) % 7
            return issued_local.date() + timedelta(days=delta)
    return None


def parse_srf(text: str) -> list[SurfDay]:
    issued = parse_issued(text)
    if issued is None:
        return []
    loc = issued.tz_convert(LOCAL_TZ).to_pydatetime()
    out = []
    for block in text.split("$$"):
        um = _UGC.search(block)
        if not um:
            continue
        zones = expand_ugc(um.group(0))
        body = block[um.end():]
        for m in _SRF_DAY.finditer(body):
            day = _day_for(m.group(1), loc)
            sh = _SURF_H.search(m.group(2))
            if day is None or not sh:
                continue
            g = sh.groups()
            if g[0]:
                lo, hi = float(g[0]), float(g[1])
            elif g[2]:
                lo = hi = float(g[2])
            else:
                lo, hi = 0.0, float(g[3] or g[4])
            rip = _RIP.search(m.group(2))
            for z in zones:
                out.append(SurfDay(z, day, lo, hi, rip.group(1).title() if rip else None, issued))
    return out


# ----------------------------------------------------------------- fetching
def split_products(blob: str) -> list[str]:
    """IEM concatenates products as \\x01 … \\x03 (newest first)."""
    parts = [p.strip("\x01\x03\n ") for p in blob.split("\x03")]
    return [p for p in parts if "\n" in p]


async def latest_product_text(http: HttpClient, product: str, office: str = OFFICE) -> tuple[str, str]:
    lst = await http.request(f"{NWS}/products/types/{product}/locations/{office}", provider="NOAA NWS", ttl=TTL_LATEST)
    graph = lst.data.get("@graph", [])
    if not graph:
        raise FetchError(f"NWS {product}{office}: no current product")
    pid = graph[0]["id"]
    res = await http.request(f"{NWS}/products/{pid}", provider="NOAA NWS", ttl=TTL_ARCHIVE)
    return res.data.get("productText", ""), res.url


async def latest_cwf(http: HttpClient, zones: list[str]) -> tuple[dict[str, ZoneForecast], str]:
    """Latest Coastal Waters Forecast; falls back to per-zone tgftp text files."""
    try:
        text, url = await latest_product_text(http, "CWF")
        parsed = parse_cwf(text)
        if parsed:
            return parsed, url
    except FetchError:
        pass
    out: dict[str, ZoneForecast] = {}
    url = ""
    for z in zones:
        try:
            res = await http.request(TGFTP_CWF.format(zone=z.lower()), provider="NOAA NWS (tgftp)", ttl=TTL_LATEST, as_json=False)
        except FetchError:
            continue
        url = res.url
        out.update({k: v for k, v in parse_cwf(res.data).items() if k == z})
    if not out:
        raise FetchError("NWS CWF unavailable from API and tgftp")
    return out, url


async def archived_products(http: HttpClient, pil: str, start: datetime, end: datetime) -> list[str]:
    res = await http.request(IEM_AFOS, provider="IEM AFOS archive (NWS text)", as_json=False, ttl=TTL_ARCHIVE,
                             params={"pil": pil, "sdate": start.strftime("%Y-%m-%dT%H:%MZ"),
                                     "edate": end.strftime("%Y-%m-%dT%H:%MZ"), "fmt": "text", "limit": "9999"})
    return split_products(res.data)


async def archived_cwf_hourly(http: HttpClient, zone: str, index: pd.DatetimeIndex) -> tuple[pd.DataFrame, list[str], str]:
    """Hourly NWS zone forecast for past hours: each hour takes the latest product issued before it.

    Returns (frame, headlines seen, product issuance note).
    """
    t0 = (index.min() - pd.Timedelta(hours=18)).to_pydatetime()
    t1 = index.max().to_pydatetime()
    products = await archived_products(http, f"CWF{OFFICE}", t0, t1)
    return cwf_hourly_from_products(products, zone, index)


def cwf_hourly_from_products(products: list[str], zone: str, index: pd.DatetimeIndex) -> tuple[pd.DataFrame, list[str], str]:
    """Build an hourly zone series from several archived CWF issuances (latest issued before each hour wins)."""
    fcs = sorted((z for z in (parse_cwf(p).get(zone) for p in products) if z), key=lambda z: z.issued)
    frame = pd.DataFrame(index=index)
    heads: list[str] = []
    if not fcs:
        raise FetchError(f"no archived CWF for {zone}")
    for zf in fcs:
        h = zone_hourly(zf, index)
        valid = index >= zf.issued
        for c in h:
            if c not in frame:
                frame[c] = np.nan
            frame.loc[valid & h[c].notna(), c] = h.loc[valid & h[c].notna(), c]
        heads += [x for x in zf.headlines if x not in heads]
    note = f"{len(fcs)} archived CWF issuances {fcs[0].issued:%m-%d %H:%MZ}…{fcs[-1].issued:%m-%d %H:%MZ}"
    return frame, heads, note


async def surf_zone_forecasts(http: HttpClient, start: datetime | None = None, end: datetime | None = None) -> list[SurfDay]:
    """NWS Surf Zone Forecast surf heights: latest product, or the archive for a date range."""
    if start is None:
        lst = await http.request(f"{NWS}/products/types/SRF/locations/{OFFICE}", provider="NOAA NWS", ttl=TTL_LATEST)
        if not lst.data.get("@graph"):
            return []                    # normal outside the beach season: no surf zone forecast issued
        text, _ = await latest_product_text(http, "SRF")
        return parse_srf(text)
    out: list[SurfDay] = []
    for p in await archived_products(http, f"SRF{OFFICE}", start, end or start + timedelta(days=1)):
        out += parse_srf(p)
    return out


async def advisories(http: HttpClient, zone: str, start: date, end: date) -> list[dict]:
    """Archived NWS watches/warnings/advisories (VTEC) for a zone between two dates."""
    res = await http.request(IEM_VTEC, provider="IEM VTEC archive (NWS)", ttl=TTL_ARCHIVE,
                             params={"ugc": zone, "sdate": start.isoformat(), "edate": end.isoformat()})
    out = []
    for e in res.data.get("events", []):
        try:
            out.append({"name": e.get("name", ""), "issue": pd.Timestamp(e["issue"]), "expire": pd.Timestamp(e["expire"]),
                        "zone": zone})
        except (KeyError, ValueError):
            continue
    return out


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
