# Blitz model and the November 2026 Bite Index fixes

Code: [`app/models/blitz.py`](../app/models/blitz.py) · settings: `blitz:` in
[`configs/species.yaml`](../configs/species.yaml) · bay/ocean gauges: `bait_flush_reference` in
[`configs/stations.yaml`](../configs/stations.yaml) · re-run the check: `scripts/validate_blitz.py`.

## What a blitz needs

A shore blitz is striped bass or bluefish driving dense bait to the surface **within casting range**, usually under
diving birds. Anglers and the weekly reports agree on the ingredients, and they have to happen **together**:

| Component | Importance | How it is measured | 1.0 means |
|---|---|---|---|
| `bait` — bait near shore | very high (0.22) | `bait_flush` (below) × spot type, blended with the bait calendar | bait pouring out of the bays at an inlet / jetty / ocean beach |
| `migration` — fish moving through | very high (0.16) | species migration calendar for the spot (`pf_<species>_season`), +8 % while the water is cooling | peak of the run |
| `temperature` | high (0.12) | measured water temperature at the spot vs `blitz.temp_f` (bass 45/50–60/66 °F, blues 56/60–72/78 °F) | inside the band |
| `access` — bait held in range, surface feeding visible | high (0.18) | wind and gusts, offshore component and its persistence, nearshore surf, clarity proxy, storm phase, rain, thunder, fog | ≤ 12 mph, no strong offshore wind, surf ≤ 3 ft, clean water |
| `tide` — moving water | high (0.12) | share of peak current (NOAA current predictions); at inlets, jetties and river mouths the **ebb** scores 1.0 and the flood 0.82 | strong ebb at an inlet |
| `light` | moderate (0.10) | sun elevation and whether it is rising: first light (−9° → +12°, rising) 1.0; last light (+10° → −6°, setting) 0.95; midday 0.45 (+0.2 overcast); night 0.25 | first light |
| `front` — barometer | moderate (0.10) | 24-hour pressure change and cloud: steady/slowly falling 1.0; the clear, rising high behind a cold front down to ~0.4; a deepening low (−6 to −12 hPa/24 h) down to ~0.7 | slow steady fall |

**Bait flush.** In the fall, juvenile menhaden (peanut bunker), silversides and bay anchovies leave the south-shore
bays through the inlets as the bays cool. StrikeCast measures it with the Great South Bay buoy (NDBC 44069) against the
ocean buoys (44065, then 44025):

```
band    = trapezoid(bay_temp, 42, 47, 60, 66)        # exodus happens while the bay is ~47-60 F
trigger = max(ramp(-bay_3day_change, 0.5, 4), ramp(ocean - bay, 1, 7))
flush   = band * (0.3 + 0.7 * trigger)               # Sept-Dec only; NaN otherwise
```

Forecast hours carry the latest gauge readings forward for up to 4 days. Sound spots have no bay/ocean pair and use
their own cooling trend as a labeled estimate (which can never produce an alert). The same flush also scales the
peanut-bunker / anchovy / spearing bait index at ocean spots, so the Bite Index sees the bait push too.

**Score** = 100 × weighted geometric mean of the seven components (so a single weak factor drags it down).

**Levels** (all thresholds in `configs/species.yaml`):

* **🔥 Alert**: score ≥ 90 **and** every component above its floor (migration 0.8, temperature 0.7, bait 0.7, access
  0.85, tide 0.85, light 0.9, front 0.8) **and** measured water temperature, measured bay/ocean gauges, tide
  predictions, and at most 72 h of lead time. Otherwise it is held at watch and the card says why.
* **👀 Watch**: score ≥ 82 and looser floors (one ingredient marginal or unverified).
* A *Birds diving* / *Surface blitz* report in the Fishing Log within 25 km lifts watches to alerts for 36 h
  (birds are the best real-time sign and have no free data feed). A report never creates a signal by itself.

## Validation (hindcast replay, Oct 20 – Nov 22, 2026)

Every date was rebuilt with the historical pipeline (measured buoys, NOAA tide/current predictions, archived
Open-Meteo weather and waves) for the saved spots. Ground truth is thin — weekly On The Water / The Fisherman reports
and the user's Friday Nov 20 blitz — so this checks that alerts land in reported blitz stretches and stay quiet in
conditions the reports call bad. It is **not** a hit-rate measurement.

| Period | Conditions | Reports | West End 2 · Point Lookout · Democrat Pt |
|---|---|---|---|
| Oct 20–23 | NE blow, 6–8 ft surf, then cold front | storms pushed shore fish south | nothing |
| Oct 24–30 | settling, bay 56 °F vs ocean 62 °F | "blitzes popping up at a moment's notice", huge push of bunker | watches at dawn/dusk |
| Oct 31 | post-front (+10.6 hPa) | — | nothing |
| Nov 1–4 | light S–SW wind, flush 0.5–0.75 | "off-the-charts" bunker-pod blitzes (boat), dawn the window | **alerts Nov 2–3** at first light (Democrat Pt also Nov 1 at dusk), watches Nov 1 and 4 |
| Nov 5–10 | warm spell (bay ≈ ocean), 4–5 ft surf, 13–17 mph W/NW | surf sporadic | nothing |
| Nov 11 | light W, flush 0.9 | — | watch (Democrat Pt: alert at dusk) |
| Nov 12–16 | nor'easter, then NW 10–16 mph post-front | north winds pushed bait out of casting range | nothing |
| Nov 17–21 | light W–E, bay 49 °F vs ocean 58 °F, flush 1.0 | "still blitzing", peanut bunker tight to the beach, outgoing tide best, sunrise/sunset | **alerts every day at the Jones Inlet spots** (first light on the ebb; Nov 17 at dusk), Democrat Pt Nov 18–20 |
| **Fri Nov 20** | 4–6 mph W, 1.5 ft surf, slowly falling barometer, ebb at first light | **user's blitz** | **highest score of the 34 days: 97.2–97.5**, alert 7–9 AM |
| Nov 22 | light NE, overcast, ahead of the next storm | — | alert 8 AM at the two Jones Inlet spots |

Alerts: West End 2 on 8 of 34 days, Point Lookout 8, Democrat Point 7; watch-or-better on 16–18. Bay piers (Field 10,
Green Island) and the Sound spot (Sunken Meadow, no bay/ocean gauge pair) produced no alerts.

The live forecast on Nov 23 (post-front Nov 24, a 56 mph-gust storm Nov 26–28) correctly shows no setup all week.

## Why the old predictions did not pay off — and what changed

The Bite Index peaked at **80–92 ("Hot") on every one of the 34 days** at the ocean spots, storms included, so it could
not tell a good day from a bad one. The logged trip of Oct 23 (Field 10, 11 AM, bluebird day after a cold front,
pressure +11 hPa/24 h, no fish) was rated **74–77 "Very good"**. Causes and fixes:

| Weakness | Fix |
|---|---|
| Pressure used only the 3-hour change, so the big rise behind a cold front looked "neutral" | 24-hour change added: post-front rise penalized, slow steady fall rewarded (`feeding.pressure_factor`) |
| Factors were averaged, so one bad factor (20 mph offshore wind, 6 ft dirty surf, post-front high) barely moved the score | wind / waves / pressure / clarity (and light for bass/blues) are **limiting**: below 0.55 they multiply the score down (`feeding_limiting`) — the same rule now feeds the 90-min optimizer's conditions component, and bass/blues windows are capped by it (`limiting: {conditions: 0.4}`) |
| A strong offshore wind bottomed out at 0.55 | falls to 0.35 at 20 mph offshore (blows bait out of range) |
| Pre-storm feeding boost (+10 % bass) applied even once a gale was building | only while wind < 18 mph and surf below the species' rough limit |
| A current borrowed from another location (labeled proxy) counted fully | weighted 60 % |
| Bait index was pure climatology | scaled by the measured bay bait flush at ocean spots |
| The 90-min optimizer averaged in season and temperature at 10 % each, so porgy scored 87 "Hot" in late November | season × temperature is now a ceiling (score × √(season × temperature)); late-November porgy → 28 |
| A bay buoy could stand in for a Sound beach when the Sound gauge was down (49 °F vs 55 °F, presence swung 88 → 64) | a bay or Sound gauge is never substituted for a different water body |
| NOAA CO-OPS answers request bursts with 403, which silently dropped tides and currents | 403 from CO-OPS is retried with backoff and requests are spaced |

Effect on the Oct 23 trip hour: **74 → 50 ("Fair")**. Post-front days are 4–8 points lower on average at their best
hour; calm settled days are unchanged.

### Honest limits

* The Bite Index still rates the best hour of most fall days 80+: the best hour of almost any fall day has good light and
  current, and no free data says whether fish are actually in front of you. Use it to pick the **hour**, and the blitz
  alert — which uses measured bait movement and is strict — to pick the **day**.
* Nothing here sees bait or birds directly. Log your sightings (birds diving / surface blitz) — they confirm alerts —
  and log skunks too: with ~20+ logged sessions the Fishing Log can show which components actually track your results.
* The calibration rests on one month and qualitative reports. Re-run `scripts/validate_blitz.py` over next spring's and
  next fall's runs and adjust the thresholds in `configs/species.yaml` if alerts are too rare or too frequent.
