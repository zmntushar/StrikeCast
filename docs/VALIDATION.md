# Surf model validation

Reproduce with `python scripts/validate_surf.py` (writes `docs/validation_results.json`).
Nothing was tuned to these numbers. They're reported as they came out, and the flags in
[SURF_MODEL.md](SURF_MODEL.md#validation-flags-never-change-the-value) exist to surface errors, not hide them.

## 1. Offshore wave model vs NDBC buoys (Sep 13 – Oct 23, 2026)

Open-Meteo Marine (best match) at each buoy's location vs the buoy's measured values, hourly.

| Buoy | Hours | Hs bias | Hs MAE | Hs r | Direction (buoy − model) | Direction MAE | Buoy Tp / model period |
|---|---|---|---|---|---|---|---|
| 44065 NY Harbor Entrance | 545 | +0.02 ft | 0.36 ft | 0.97 | +9.6° median | 22.9° | 1.08 |
| 44025 Long Island 30 nm S | 499 | −0.01 ft | 0.44 ft | 0.98 | +5.0° median | 21.5° | 1.09 |

**Findings.** Model wave *height* offshore is reliable, so the old method's low surf was not caused by a low-biased model.
*Direction* is the weak input. Over the 7 days before Oct 23 the model ran +17–22° counter-clockwise of the
buoys (+25–32° in the last 48 h), which turns easterly swell into "along the beach" on south-facing shores.
StrikeCast now corrects it from the buoys. The model's mean periods sit below the buoys' peak periods by the ratio
JONSWAP theory predicts, so they're converted rather than fitted.

A 7-day comparison of other Open-Meteo wave models (ECMWF WAM, NOAA GFS-Wave 0.16°/0.25°, DWD GWAM) found better direction
for GFS-Wave (15–22° MAE) but worse heights (bias up to −0.8 ft at 44025). Best match (MeteoFrance MFWAM) was kept for height
skill; its direction error is corrected from the buoys.

## 2. Ocean-beach surf vs NWS Surf Zone Forecasts (Jun 1 – Sep 15, 2026)

The NWS Surf Zone Forecast (SRF) is an independent forecaster product giving a breaking-surf range for each beach zone. It's a
forecast, not an observation, and is issued only in the beach season, so it can't check fall/winter swell events.
Each day compares StrikeCast's 8 AM–6 PM mean hindcast (archived model + buoys) with that day's SRF issued
before 8 AM. 105 days per spot.

| Spot (SRF zone) | Method | Within NWS range ±0.5 ft | Below range | Mean bias vs NWS midpoint | MAE |
|---|---|---|---|---|---|
| Jones Beach Field 2 (NYZ179) | **new, buoy-corrected** | 46.7% | 8.6% | +0.55 ft | 0.83 ft |
| | new, model only | 43.8% | 6.7% | +0.59 ft | 0.82 ft |
| | old heuristic | 66.7% | 29.5% | −0.39 ft | 0.56 ft |
| Point Lookout (NYZ179) | **new, buoy-corrected** | 57.1% | 19.0% | +0.12 ft | 0.71 ft |
| | new, model only | 58.1% | 17.1% | +0.14 ft | 0.69 ft |
| | old heuristic | 52.4% | 47.6% | −0.80 ft | 0.81 ft |
| Democrat Point (NYZ080) | **new, buoy-corrected** | 56.2% | 17.1% | +0.22 ft | 0.68 ft |
| | new, model only | 51.4% | 12.4% | +0.35 ft | 0.70 ft |
| | old heuristic | 38.1% | 61.9% | −1.04 ft | 1.04 ft |

By NWS category (buoy-corrected new model, daily means):

| NWS said | n (Field 2) | Field 2 | Point Lookout | Democrat Pt | old method (Democrat Pt) |
|---|---|---|---|---|---|
| 1–2 ft | 15–16 | 2.4 | 2.0 | 2.2 | 1.2 |
| 2 ft | 20–24 | 2.9 | 2.5 | 2.6 | 1.5 |
| 2–3 ft | 16 | 3.2 | 2.8 | 2.9 | 1.7 |
| 3 ft | 26–28 | 3.4 | 2.9 | 3.1 | 1.8 |
| 3–4 ft | 7 | 3.4 | 3.1 | 3.2 | 1.8 |
| 4 ft | 7 | 3.1 | 2.7 | 2.9 | 2.0 |

**Findings.**

* The old method was **systematically low**: below the NWS range on 30–62% of days, averaging −0.4 to −1.0 ft,
  worst at Democrat Point and Point Lookout where the hand-set exposure factors (0.80, 0.85) applied.
* The new model removes that low bias and is closer on average at Point Lookout and Democrat Point. At Field 2 (a fully
  exposed beach where the old method had no exposure penalty) it runs ~0.5 ft above the NWS midpoint.
* Over-estimation is concentrated on small summer days (NWS "1–2 ft"). On 3–4 ft days the new model is within ~0.4 ft.
  Part of this is regression toward the mean from grouping by NWS's 1-ft categories (correlation with NWS: 0.69–0.74 new,
  0.75–0.77 old). The likely physical part is no energy loss across the shelf for short, steep summer wind-waves
  (a documented limitation).
* Summer has few long-period, oblique-swell days, which is where the old method failed worst. The fall case below shows it.

## 3. Long Island Sound wind chop vs NWS zone forecast (Oct 1 – 22, 2026)

Sunken Meadow Field 3 wind chop (SPM fetch-limited growth from hindcast wind) vs the NWS coastal-waters forecast for
ANZ335 (72 archived issuances):

| Hours | Within NWS range ±0.5 ft | Bias vs NWS midpoint |
|---|---|---|
| 521 | 73.3% (71.7% in onshore wind) | −0.51 ft |

The NWS zone covers the open western Sound; a beach in Smithtown Bay sees shorter fetches from the west and east, so
slightly lower values are expected.

## 4. Live spot check: Oct 23, 2026, ~1:45 PM EDT

NE wind 13–20 kt easing, Small Craft Advisories for ANZ353/ANZ355. NWS: "Seas 4 to 6 ft, occasionally to 8 ft. Wave
Detail: E 6 ft at 9 seconds" (Fire Island Inlet–Moriches) and "Seas 3 to 5 ft … E 5 ft at 9 seconds" (Sandy Hook–Fire
Island Inlet).

| Spot | Old app | New surf (range) | Kind / confidence | Offshore (corrected model) | Measured buoy | NWS |
|---|---|---|---|---|---|---|
| Robert Moses – Democrat Pt | — | **3.5 ft** (1.9–5.6) | breaking · low (oblique ESE swell, inlet) | 4.1 ft · 7 s · ESE | 44025: 6.6 ft (30 nm out) | seas 4–6, occ. 8 |
| Point Lookout | 1.3 ft | **3.0 ft** (1.6–5.0) | breaking · low (22% of energy via diffraction around Jones Beach West End) | 4.3 ft · 6 s · ESE | 44065: 4.8 ft | seas 3–5, occ. 6 |
| Jones Beach Field 2 | — | **5.0 ft** (3.4–6.9) | breaking · medium | 4.4 ft · 6 s · ESE (+ 2.5 ft 10 s secondary swell) | 44065: 4.8 ft | seas 3–5, occ. 6 |
| Jones Beach Field 10 (bay pier) | 0.1 ft "surf" | **0.3 ft wind chop** (0.2–0.5) | sheltered bay · medium | — | — | bays: waves 1–2 ft → ≤ 1 ft |
| Sunken Meadow Field 3 | — | **1.3 ft wind chop** (0.7–2.0) | Sound · low | — | — | Sound: seas 1–2 ft, NE 2 ft at 4 s |

Every spot now gets its own number driven by its exposure. Field 10 is correctly treated as sheltered water, and the
ocean beaches are consistent with the official NWS wave forecast.

## 5. Historical comparison check: Democrat Point, Oct 10 vs Oct 22, 2026

From archived data only (`compare_dates`):

| | Oct 10 | Oct 22 |
|---|---|---|
| Buoy 44025 Hs mean / max | 1.8 ft (only 3 of 24 h reported: outage, flagged) | 8.4 / 13.0 ft (24/24 h) |
| Buoy dominant period | 4.3 s | 10.3 s |
| Model offshore Hs mean | 1.6 ft | 6.2 ft |
| Swell (used, primary) | 1.3 ft · 9 s · SE | 5.2 ft · 11 s · ESE |
| Surf estimate mean / max | 2.2 / 2.4 ft (medium confidence) | 6.1 / 8.8 ft (low confidence) |
| NWS zone forecast (archived) | seas 2–3 ft | seas 5–12 ft |
| Advisories (archived VTEC) | none | Small Craft Advisory, Coastal Flood Statement |
| Wind | 6 mph E | 15 mph N (offshore), gusts 37 |
| Water temperature | 64.5°F | 61.7°F |
| Tide at sunrise | late incoming | mid outgoing |

The Oct 10 buoy outage is reported as such. Neither day's gaps are filled with current or other-day data.
