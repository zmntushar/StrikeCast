"""Spots scored in worker processes must match in-process scoring exactly."""
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from app.core.http import FetchError
from app.services import forecast as F


class _AllFail(dict):
    def get(self, key, default=None):
        return FetchError("offline")


def test_process_pool_matches_sequential():
    spots = [F.get_spot(i) for i in ("point_lookout_jetties", "jones_beach_west_end_2", "robert_moses_democrat_point")]
    frame = F.forecast_frame(2)
    raw = _AllFail()
    seq = F._assemble_all(spots, frame, raw, None, F.Progress(None, 1), None)
    steps = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        par = F._assemble_all(spots, frame, raw, None, F.Progress(lambda p, l: steps.append(l), 1), pool)
    assert list(par) == [s.id for s in spots] and len(steps) == len(spots)
    for sid in seq:
        a, b = seq[sid].hourly["bite_striped_bass"], par[sid].hourly["bite_striped_bass"]
        assert np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True)
        assert seq[sid].quality == par[sid].quality


def test_workers_do_not_rerun_the_main_script(tmp_path, monkeypatch):
    """Under Streamlit, __main__ is the dashboard script; spawned workers must not execute it."""
    import sys
    import types
    from app.core.config import get_settings

    script = tmp_path / "dashboard_like.py"
    script.write_text("raise RuntimeError('main script must not run in workers')\n")
    fake_main = types.ModuleType("__main__")
    fake_main.__file__ = str(script)
    monkeypatch.setitem(sys.modules, "__main__", fake_main)
    monkeypatch.setattr(get_settings(), "assemble_workers", 2)
    F._drop_pool()
    try:
        pool = F._pool()
        assert pool is not None
        assert pool.submit(F._warm).result(timeout=60) > 0
        assert sys.modules["__main__"] is fake_main          # restored after workers start
    finally:
        F._drop_pool()
