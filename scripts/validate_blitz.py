"""Replay the blitz model over past dates (hindcasts) and list every alert / watch per spot.

    python scripts/validate_blitz.py 2026-10-20 2026-11-22            # your saved spots
    python scripts/validate_blitz.py 2026-11-15 2026-11-22 --spots jones_beach_west_end_2 point_lookout_jetties

Each date is rebuilt with the historical pipeline (measured buoys, NOAA tide/current predictions, archived
model weather and waves) exactly as the Compare Dates tab does, so the result shows what the model WOULD
have said. Compare it with fishing reports or your own log: alerts should sit inside reported blitz
stretches and be absent on storm / post-front / strong-offshore-wind days. Days are fetched one at a
time with a short pause (NOAA CO-OPS throttles bursts). Results are cached, so re-runs are fast.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services import my_spots  # noqa: E402
from app.services.forecast import collect, historical_frame, load_spots, run_sync  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("start", type=date.fromisoformat)
    ap.add_argument("end", type=date.fromisoformat)
    ap.add_argument("--spots", nargs="*", help="spot ids (default: your saved spots)")
    ap.add_argument("--json", type=Path, help="also write the results to this JSON file")
    ap.add_argument("--pause", type=float, default=2.0, help="seconds between dates (NOAA throttling)")
    a = ap.parse_args()

    ids = set(a.spots or my_spots.saved_ids())
    spots = [s for s in load_spots() if s.id in ids]
    if not spots:
        sys.exit("no matching spots")
    tz = spots[0].timezone
    rows = []
    d = a.start
    while d <= a.end:
        res = run_sync(collect(spots, historical_frame(d, tz)))
        for sp in spots:
            fc = res[sp.id]
            day = fc.display
            lv = day["blitz_level"].fillna("")
            hrs = lambda lvl: [int(h) for h in day.index[lv == lvl].tz_convert(tz).hour]  # noqa: E731,B023
            rows.append({"date": str(d), "spot": sp.id, "alert_hours": hrs("alert"), "watch_hours": hrs("watch"),
                         "max_score": round(float(day["blitz_score"].max()), 1),
                         "missing": fc.quality.get("missing", []),
                         "windows": [{"level": w.level, "start": w.start.tz_convert(tz).strftime("%H:%M"),
                                      "end": w.end.tz_convert(tz).strftime("%H:%M"), "score": w.score,
                                      "reasons": w.reasons} for w in fc.blitz]})
            flag = "ALERT" if rows[-1]["alert_hours"] else "watch" if rows[-1]["watch_hours"] else "-"
            miss = f"  (missing: {', '.join(rows[-1]['missing'])})" if rows[-1]["missing"] else ""
            print(f"{d}  {sp.short:<14} {flag:<5}  alert {rows[-1]['alert_hours']}  watch {rows[-1]['watch_hours']}"
                  f"  max {rows[-1]['max_score']:.0f}{miss}", flush=True)
        d += timedelta(days=1)
        if d <= a.end:
            time.sleep(a.pause)

    n_days = (a.end - a.start).days + 1
    print("\nDays with an alert / with a watch or better, per spot (of", n_days, "):")
    for sp in spots:
        r = [x for x in rows if x["spot"] == sp.id]
        print(f"  {sp.name:<45} {sum(bool(x['alert_hours']) for x in r):>3} / "
              f"{sum(bool(x['alert_hours'] or x['watch_hours']) for x in r):>3}")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print("wrote", a.json)


if __name__ == "__main__":
    main()
