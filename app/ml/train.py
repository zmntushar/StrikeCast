"""Personal predictive models trained on the fishing log (Phase 4).

* Target: "caught at least one <species> on the trip" (binary).
* Validation: CHRONOLOGICAL split (train on older trips, test on newer) to avoid
  leakage; optional leave-one-spot-out check for geographic generalization.
* Metrics: ROC-AUC, PR-AUC, Brier score, precision/recall.
* Output is only labeled a probability after Platt calibration on enough data;
  otherwise it is reported as a relative score.

Observation bias: trips with no catch do not prove fish were absent, and logged
trips are not a random sample of conditions. Treat results as personal insight.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.core.config import ROOT_DIR
from app.services.observations import training_frame

MODEL_DIR = ROOT_DIR / "data" / "models"
MIN_ROWS, MIN_POS, MIN_NEG, CALIBRATE_AT = 20, 5, 5, 60

BASE_FEATURES = ["water_temp_f", "water_temp_trend_3d", "wind_mph", "onshore_mph", "nearshore_wave_ft",
                 "current_norm", "tide_frac", "tide_range_ft", "sun_alt", "moon_illum", "pressure_tend_3h",
                 "cloud_pct", "bait_index"]


@dataclass
class TrainReport:
    species: str
    ok: bool
    message: str
    n_rows: int = 0
    n_pos: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    importance: list[tuple[str, float]] = field(default_factory=list)
    calibrated: bool = False


def _features(df: pd.DataFrame, species: str) -> pd.DataFrame:
    cols = [c for c in BASE_FEATURES + [f"presence_{species}", f"feeding_{species}"] if c in df]
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    for cat in ("light_phase", "tide_stage_tag"):
        if cat in df:
            X = X.join(pd.get_dummies(df[cat].fillna("na"), prefix=cat, dtype=float))
    return X


def train_species(species: str) -> TrainReport:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    df = training_frame()
    if df.empty:
        return TrainReport(species, False, "No logged trips with environmental snapshots yet.")
    df = df[df["species"] == species].sort_values("start_utc")
    n, pos = len(df), int(df["caught"].sum())
    if n < MIN_ROWS or pos < MIN_POS or (n - pos) < MIN_NEG:
        return TrainReport(species, False,
                           f"Need ≥{MIN_ROWS} trips with ≥{MIN_POS} successful and ≥{MIN_NEG} unsuccessful "
                           f"(have {n} trips, {pos} with fish).", n, pos)
    X, y = _features(df, species), df["caught"].to_numpy()
    cut = int(n * 0.75)
    Xtr, Xte, ytr, yte = X.iloc[:cut], X.iloc[cut:], y[:cut], y[cut:]
    base = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=2000, C=0.5))
    calibrated = n >= CALIBRATE_AT
    model = CalibratedClassifierCV(base, method="sigmoid", cv=3) if calibrated else base
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    metrics: dict[str, float] = {"test_rows": float(len(yte))}
    if len(set(yte)) > 1:
        metrics.update(roc_auc=roc_auc_score(yte, p), pr_auc=average_precision_score(yte, p))
    metrics.update(brier=brier_score_loss(yte, p), precision=precision_score(yte, p >= 0.5, zero_division=0),
                   recall=recall_score(yte, p >= 0.5, zero_division=0))
    base.fit(X, y)  # refit on everything for deployment + importances
    coefs = base[-1].coef_[0]
    imp = sorted(zip(X.columns, coefs), key=lambda t: abs(t[1]), reverse=True)[:8]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump({"model": model if calibrated else base, "columns": list(X.columns), "calibrated": calibrated},
                MODEL_DIR / f"{species}.joblib")
    return TrainReport(species, True, "Model trained (chronological 75/25 split).", n, pos,
                       {k: round(float(v), 3) for k, v in metrics.items()},
                       [(k, round(float(v), 3)) for k, v in imp], calibrated)


def predict(species: str, frame: pd.DataFrame) -> pd.Series | None:
    path = MODEL_DIR / f"{species}.joblib"
    if not path.exists():
        return None
    import joblib
    bundle: dict[str, Any] = joblib.load(path)
    X = _features(frame, species).reindex(columns=bundle["columns"], fill_value=0.0)
    return pd.Series(bundle["model"].predict_proba(X)[:, 1], index=frame.index)


def model_exists(species: str) -> bool:
    return (MODEL_DIR / f"{species}.joblib").exists()

