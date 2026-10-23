# StrikeCast surf model

How StrikeCast turns free offshore wave data into the surf an angler meets at each spot,
and what it does when data are missing. Validation results are in [VALIDATION.md](VALIDATION.md).

## Why the old numbers were too low

The previous estimate (removed `nearshore_waves()` in `app/features/weather_features.py`) was

```
surf = model wave height ~1 km off the spot × exposure (0.05–1.0, hand-set) × cos²(angle/2)
```

Four problems combined, all pushing the result down:

| Problem | Effect | Example (Oct 23 2026, E–ESE seas 4–6 ft, NWS "E 6 ft at 9 s") |
|---|---|---|
| **Coastal model cell.** The point 1 km offshore snapped to a coarse (1/12°, ~9 km) model cell that partly covers land, bays or the inlet. | Offshore height 15–30 % low. | Point Lookout cell: 3.5 ft vs 4.1–4.5 ft 15 km out. Field 10's "ocean" cell sat over the bay. |
| **Two attenuation factors stacked.** A hand-set `exposure` multiplier *and* a cos² direction factor. Neither came from wave physics; together they cut height in half for easterly swell. | Systematically low on any oblique swell. | Point Lookout: 3.48 ft × 0.85 × 0.46 = **1.3 ft**. |
| **Mean direction only.** One averaged direction (here 82°, along the beach) ignored the separate swell trains. Open-Meteo's wave directions also run ~10–30° counter-clockwise of the NDBC buoys (measured). | Long-period ESE swell treated as if running along the beach. | Buoy 44065 measured the swell from ESE (112°); the model said 82°. |
| **No shoaling or breaking.** Offshore height was scaled *down*, never transformed. Real waves grow as they shoal before breaking. | Breaker heights under-predicted even for head-on swell. | — |

"0.1 ft at Jones Beach" was **Field 10**, a pier on the bay side of the barrier island. It was labelled
"surf", which it is not. The old code used the bay's wind chop formula there, scaled by a 0.05 exposure factor.

## Terms (kept separate everywhere)

| Term | Meaning | Where it shows |
|---|---|---|
| **Surf** (breaking wave height, Hb) | Significant height of waves where they break near the beach. | Spot card "Surf", charts, Bite Index |
| **Wind chop** | Fetch-limited waves made by local wind in bays and the Sound. Ocean swell can't get there. | Spot card "Wind chop" at sheltered spots |
| **Offshore waves** (Hs) | Average of the highest third of waves, ~12 km out in 20–30 m of water. | "Offshore" row, buoy row |
| **Swell / wind waves** | Partitions of the same sea state. Their **energies** add (Hs² = Σ Hᵢ²); their heights never do. | "Swell" (primary + secondary), "Wind waves" |
| **Period** | Peak period Tp (s) unless stated. | All wave rows |

## Data sources and fallback

| Priority | Source | Type | Used for |
|---|---|---|---|
| 1 | [Open-Meteo Marine](https://open-meteo.com/en/docs/marine-weather-api) best-match (MeteoFrance MFWAM, 1/12°), at a point **12 km offshore along the spot's facing direction** (20–37 m deep; `configs/exposure.yaml`) | model | Hourly offshore boundary: total Hs, primary swell, secondary swell, wind sea (height, period, direction) |
| ↳ correction | [NOAA NDBC](https://www.ndbc.noaa.gov/) 44065 (Nassau) / 44025 (Suffolk) stdmet + spectral summary | measured | Model scaled by measured/model height ratio and rotated by measured–model direction offset at the buoy |
| 2 | [NWS Coastal Waters Forecast](https://www.weather.gov/okx/) (CWF, zones ANZ3xx), via api.weather.gov; per-zone text from tgftp.nws.noaa.gov if the API is down | predicted | Fallback boundary ("Wave Detail" components). Always a cross-check, plus marine advisory headlines |
| 3 | NDBC buoy partitions | measured | Fallback for **past/current hours only**. Never projected forward |
| — | NWS Surf Zone Forecast (SRF, zones NYZ…) | predicted | Independent breaking-surf check, beach season only |
| — | [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/) AFOS & VTEC archives | archived NWS products | CWF, SRF and advisories for past dates (date comparison, validation) |
| — | NOAA NCEI DEM mosaic (CUDEM ~3 m nearshore, coarser offshore) | survey | Exposure rays, boundary depth, beach slope, sandbars |
| — | NWS / Open-Meteo wind | predicted / model | Wind chop, surf quality (clean/choppy), local wind effects |
| — | NOAA CO-OPS tides & currents | predicted | Tide level over sandbars, ebb current at inlets |

Every source carries provider, station, distance, observation time and age in the **Data sources** table.
Stale data are flagged, not silently used:

* Buoy older than 3 h is marked stale. Older than 6 h, StrikeCast tries the next buoy; if none is fresh, there is no buoy correction.
* An NWS product past its UGC expiration time is marked **EXPIRED**.
* Cached responses served during an outage are marked `stale`, and confidence drops.
* QC (`marine.qc_waves`) removes impossible values (Hs outside 0–60 ft, periods outside 1–30 s, directions outside 0–360°), isolated spikes and stuck sensors. It never fills gaps.

If no source is usable, the card says **"Surf estimate unavailable"**. No number is invented.

## Ocean beaches: physics

Code: `app/features/surf.py` (pure functions), `app/services/surf.py` (pipeline).

1. **Partitions.** Primary swell, secondary swell and wind sea at the offshore point. Model energy not in any partition becomes its
   own component. Open-Meteo gives mean periods; peak period comes from standard JONSWAP spectral ratios,
   **Tp ≈ Te/0.90** (total) and **Tp ≈ Tm01/0.78** (partitions). Measured buoy/model ratios were 1.12 and 1.28, matching theory.
2. **Buoy correction** (`anchor_to_buoy`). At the buoy's location: height ratio and circular direction offset, measured vs
   model. The correction combines the running bias over the last ~4 days with the last 12 h departure, which fades back to
   the running bias over ~12 h of forecast. Ratios outside 0.5–2, or offsets beyond 45°, are **not applied**.
   They're flagged as a provider inconsistency and confidence drops.
3. **Directional spreading.** Each partition is spread over a wrapped-normal spectrum (σ = 15° swell, 30° wind sea;
   typical values, Goda 2000 / CEM II-1) in 5° bins.
4. **Sheltering.** 36 rays from the spot over the NOAA DEM give the over-water distance to land in every direction
   (`scripts/compute_exposure.py` → `configs/exposure.yaml`). Directions with land closer than 10 km can't bring offshore waves.
   Waves are first refracted (Snell's law) to the water depth at the edge of the sheltering land, then tested.
   Blocked directions keep energy by Sommerfeld far-field diffraction:
   Kd ≈ 1 / (2 √(2π k r) · sin(β/2)), capped at 0.5 on the shadow line.
5. **Shoaling + refraction to breaking.** Energy flux E·Cg·cos α is conserved from the offshore depth to breaking over
   straight, parallel contours (linear wave theory; full dispersion relation; Snell's law at the breaker depth).
6. **Breaking height.** Komar & Gaughan (1972), generalised for obliquity:
   **Hb = 0.39 g^⅕ (Tp · H0² · cos α0 / cos αb)^⅖**. H0² cos α0 comes from flux conservation from the offshore depth;
   breaker depth hb = Hb / γb with γb from Weggel (1972) using the DEM beach slope.
7. **Combine.** Directional bins and partitions add as breaking energy flux, Hb_total = (Σ Hb_i^{5/2})^{2/5}.
8. **Extras.**
   * Tide over the bar: the DEM profile finds a candidate outer bar. If the breaker depth exceeds the water over the bar at
     the current tide (NAVD88 ≈ MLLW + 0.7 m), waves break on the **outer bar**, otherwise **near the beach**.
   * Breaker type from the Iribarren number (spilling / plunging / surging).
   * **Inlets** (Democrat Point, Jones Inlet, Moriches, Shinnecock): wave–current interaction, Longuet-Higgins & Stewart (1961),
     H/H0 = c0 / √(c(c − 2U)). This uses the NOAA ebb current running against the waves and is reported as "Inlet mouth"
     height, with current blocking flagged.
   * Wave power P = ρ g² Hs² Te / (64π).

## Bays and Long Island Sound: wind chop

Ocean swell can't reach these shorelines (the ray-cast finds no open-ocean path). Surf there is
fetch-limited wind sea from the **SPM (1984) shallow-water growth equations**:

```
U_A = 0.71 U10^1.23
g H / U_A² = 0.283 tanh(0.530 (g d/U_A²)^¾) · tanh( 0.00565 (g F/U_A²)^½ / tanh(0.530 (g d/U_A²)^¾) )
```

F is the over-water fetch toward the direction the wind comes from (mean of rays within ±15°, CEM II-2), and d is the mean
depth along it. Wind is a 2-hour vector mean, so waves lag sudden wind shifts. Offshore wind at a spot means zero fetch and no chop.

## Uncertainty and confidence

Range = min/max of the estimate with all wave directions rotated ±10°, times (1 ± u)^0.8 for the offshore-height
uncertainty u, times (1 ± 0.15) for the scatter of the breaker relation. Wind chop uses wind speed ±20%, direction ±15°
and ±20% formula scatter.

| u (offshore height) | when |
|---|---|
| 0.18 | buoy-corrected, within 6 h of the last buoy reading |
| 0.25 | model only |
| 0.30 / 0.35 | buoy-only / NWS-only fallback, or buoy and model disagree |
| + up to 0.10 | forecast lead time (5 days) |
| + 0.08 each | most energy arriving > 60° off the beach normal; > 30% arriving only by diffraction |
| + 0.03 / 0.05 | no beach-slope data / stale cached model |

The **confidence** label comes from the half-width of the range relative to the estimate: high ≤ 30%, medium ≤ 45%,
low above. Validation flags widen it further.

## Validation flags (never change the value)

* Model offshore Hs outside the NWS zone's forecast seas ±1.5 ft.
* Wind chop outside the NWS bay/Sound wave range ±1.5 ft.
* *Implausibly low*: ≥ 3 ft offshore, ≥ 7 s, aimed within 45° of the beach, not sheltered, yet surf < 40% of offshore.
* Buoy/model provider inconsistency.

## Bite Index integration

* **Surf factor** uses the physical surf (ocean) or chop (sheltered). Its weight scales with exposure: open ocean 1.0,
  Sound 0.6, bay 0.3. Bay chop isn't a feeding trigger the way ocean whitewater is.
* Species bands (`waves_ft` in `configs/species.yaml`) are unchanged. Striped bass and bluefish score best in moderate surf
  (1.5–4.5 ft and 1–5 ft); fluke, blackfish and porgy prefer < 2–2.5 ft. Calm water is not assumed to be best.
* **Water-clarity proxy** (0 clear … 1 dirty; heuristic) from 24 h of surf, 48 h of rain and storm phase. Feeds a new
  species-specific clarity factor (bass like some color; fluke, porgy, tog prefer clear) and the tackle color advice.
* **Fishability** at inlet spots uses the larger of beach surf and inlet-mouth waves.
* **Surf safety** (OK / Caution / Dangerous) is shown separately and does **not** feed the Bite Index. It considers breaker
  height, long-period surges on rocks (≥ 8–9 s) and inlet ebb against the waves.

## Assumptions and limitations

* Straight, parallel depth contours between the offshore point and the beach. Inlet ebb shoals, jetties and groin fields
  focus or shelter waves in ways this cannot capture. Point Lookout and Democrat Point are the most affected; their
  confidence is usually "low".
* Diffraction uses a semi-infinite-barrier result; refraction into shadow zones over shoals isn't modelled, so sheltered
  sides may be under-estimated (the card says so when > 30% of the energy is diffracted).
* No energy loss (bottom friction, whitecapping) between the offshore point and the beach, so small, steep summer
  wind-waves can come out a little high (see validation).
* Komar & Gaughan was calibrated on regular waves. Using it with Hs and Tp is standard practice, but scatter is ~±15%.
* DEM surveys are dated; sandbars move after storms. Bar/tide "break zone" is labeled "candidate".
* Wind chop assumes steady (fetch-limited) growth; duration limits on brief wind events are only roughly handled by the
  2-h smoothing. Over-water wind can exceed the land-influenced model wind.
* The NAVD88–MLLW offset (0.7 m) is a south-shore approximation.
* Individual set waves run larger than the significant height (H1/10 ≈ 1.27 Hs).

## Re-running the geometry

```bash
python scripts/compute_exposure.py              # all spots → configs/exposure.yaml
python scripts/compute_exposure.py captree      # one spot
```

Custom locations (ZIP / coordinates) are ray-cast automatically at run time.
