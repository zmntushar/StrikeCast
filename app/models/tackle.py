"""Condition-based lure / jig / rig suggestions.

Rules of thumb that shore anglers in the NY Bight commonly use, driven by the
forecast at a given hour:

* jig / bucktail / sinker WEIGHT from water depth in casting range (NOAA
  bathymetry) and current speed — you need enough weight to hold bottom
* COLOR from water clarity (surf, storms, rain) and light (day / dawn-dusk /
  night, moon illumination)
* PROFILE from the bait most likely present (sand eels, bunker, spearing…)

These are starting points, not guarantees. Product names are common examples of
a lure *type*; any equivalent works.
"""
from __future__ import annotations

import math
from typing import Any

DEFAULT_DEPTH_M = {"beach": 2.5, "jetty": 4.0, "inlet": 6.0, "pier": 4.0, "bay": 3.0, "river_mouth": 3.0}

BAIT_MATCH = {
    "sand_eel": "slim sand-eel profile (diamond jig / Ava, Deadly Dick, Hogy, Slug-Go)",
    "adult_bunker": "big bunker profile (7–9\" paddletail, large swimming plug, snag-and-drop)",
    "peanut_bunker": "3–4\" peanut-bunker profile (Kastmaster, epoxy jig, small swimmer)",
    "bay_anchovy": "small 2–3\" rainfish profile (epoxy jig, small metal, Albie Snax)",
    "silverside": "slim 3–4\" spearing profile (small tin, Gulp! swimming mullet, teaser)",
}


def _num(v: Any, default: float) -> float:
    try:
        f = float(v)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


def _quarter(oz: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, round(oz * 4) / 4))


def _oz(oz: float) -> str:
    whole, frac = int(oz), oz - int(oz)
    f = {0.0: "", 0.25: "¼", 0.375: "⅜", 0.5: "½", 0.75: "¾"}.get(round(frac, 3), f"{frac:.2f}"[1:])
    return f"{whole if whole else ''}{f} oz" if (whole or f) else "¼ oz"


def water_clarity(row: Any) -> str:
    ci = _num(row.get("clarity_index"), float("nan"))
    if not math.isnan(ci):
        return "clear" if ci < 0.25 else "moderate" if ci < 0.55 else "stained"
    surf = _num(row.get("nearshore_wave_ft"), 1.0)
    phase = str(row.get("storm_phase") or "none")
    since = _num(row.get("hours_since_storm"), 99)
    rain = _num(row.get("precip_in"), 0)
    if phase == "during" or (phase == "post" and since < 24) or surf > 3.5 or rain > 0.1:
        return "stained"
    if surf < 1.5 and phase == "none":
        return "clear"
    return "moderate"


def recommend(species: str, row: Any, environment: str, depth_m: float | None = None) -> dict[str, Any]:
    """Return a tackle suggestion for one species at one forecast hour."""
    depth = depth_m if depth_m and depth_m > 0 else DEFAULT_DEPTH_M.get(environment, 3.0)
    depth_ft = depth * 3.28
    cur = abs(_num(row.get("current_kt"), _num(row.get("current_norm"), 0.4) * 2.0))
    light = str(row.get("light_phase") or "day")
    moon = _num(row.get("moon_illum"), 50)
    clarity = water_clarity(row)
    bait = str(row.get("bait_top") or "")

    if light == "night":
        color = "white / pearl (bright moon)" if moon >= 50 else "black or dark purple (dark night)"
    elif clarity == "stained":
        color = "chartreuse, yellow or pink (dirty water)"
    elif light in ("dawn", "dusk"):
        color = "white, bone or chartreuse"
    else:
        color = "natural: white, olive/silver, bone" if clarity == "clear" else "white or chartreuse"

    match = BAIT_MATCH.get(bait, "")
    notes: list[str] = []
    if cur > 2.0:
        notes.append(f"Strong current (~{cur:.1f} kt): cast up-current and let it swing; go heavier if you lose bottom.")
    elif cur < 0.4:
        notes.append("Slack water: go lighter and work it slowly.")

    if species == "striped_bass":
        oz = _quarter(0.25 + depth_ft / 20 + cur * 0.4, 0.5, 3.0)
        if light == "night":
            primary = "Darter or needlefish plug, slow-retrieved; live eel if allowed"
            jig = f"{_oz(oz)} bucktail with a 5–6\" pork rind / soft trailer, swung slowly"
        elif bait == "adult_bunker":
            primary = "7–9\" paddletail or large swimming plug; snag-and-drop if bunker are pushed in"
            jig = f"{_oz(oz)} bucktail or jig-head paddletail"
        elif bait == "sand_eel":
            primary = "Diamond jig / Ava (1–2 oz) or Hogy/Slug-Go on a jig head"
            jig = f"{_oz(oz)} bucktail with a slim teaser 18\" above"
        else:
            primary = "SP minnow / swimming plug at low light; pencil popper if fish are breaking"
            jig = f"{_oz(oz)} bucktail with soft-plastic trailer"
        rig = "Circle hooks required with natural bait; 30–40 lb fluorocarbon leader."
    elif species == "bluefish":
        oz = _quarter(0.75 + depth_ft / 30 + cur * 0.25, 1.0, 3.0)
        primary = "Pencil popper or walking plug when birds are working" if light != "night" else "Cut bunker on a fish-finder rig"
        jig = f"{_oz(oz)} metal (Hopkins, Kastmaster, Ava) — fast retrieve"
        rig = "Short wire or 60–80 lb leader — blues bite through mono."
    elif species == "fluke":
        oz = _quarter(0.75 + depth_ft / 15 + cur * 0.5, 1.0, 4.0)
        primary = "Bucktail tipped with a 4–6\" Gulp! swimming mullet or squid strip, bounced on bottom"
        jig = f"{_oz(oz)} bucktail with a teaser 18–24\" above"
        rig = "Work channel edges and drop-offs on the moving tide; 20–30 lb fluoro leader."
    elif species == "blackfish":
        oz = _quarter(0.375 + depth_ft / 40 + cur * 0.25, 0.375, 1.5)
        primary = "Tog jig tipped with half a green or Asian crab, tight to the rocks"
        jig = f"{_oz(oz)} tog jig (or 2–4 oz sinker on a single-hook tog rig)"
        rig = "Short 30–50 lb leader; fish the slower-current windows; expect snags."
        color = "chartreuse, orange or white jig head (crab is the attractor)"
    elif species == "porgy":
        oz = _quarter(1.0 + depth_ft / 15 + cur * 0.5, 1.0, 4.0)
        primary = "Hi-lo rig with #1–#4 hooks, clam strips or squid; small diamond jig + teaser also works"
        jig = f"{_oz(oz)} bank sinker on the hi-lo (or ½–1 oz small jig)"
        rig = "Small hooks win — porgies nibble. Chum with crushed clam near structure."
        color = "pink or red teasers / beads"
    else:
        oz = 1.0
        primary, jig, rig = "", "", ""
    return {"primary": primary, "jig": jig, "weight_oz": oz, "color": color, "rig": rig,
            "bait_match": match, "clarity": clarity, "depth_ft": round(depth_ft), "current_kt": round(cur, 1),
            "notes": notes}
