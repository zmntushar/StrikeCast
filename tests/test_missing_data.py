"""If every provider fails, the pipeline must still run, mark data missing, and
never invent observations."""
import numpy as np

from app.core.http import FetchError
from app.services.forecast import assemble, forecast_frame, get_spot


class _AllFail(dict):
    def get(self, key, default=None):  # every provider call failed
        return FetchError(f"{key} offline")


def test_assemble_with_no_data():
    spot = get_spot("jones_beach_field_2")
    fc = assemble(spot, forecast_frame(2), _AllFail(), 0)
    d = fc.display
    assert not d.empty
    for col in ("water_temp_f", "wind_mph", "tide_ft", "wave_ft", "current_kt"):
        assert d[col].isna().all(), col  # explicit missing, nothing fabricated
    assert fc.quality["grade"] == "D"
    for m in ("weather", "tides", "water temperature", "bathymetry"):
        assert m in fc.quality["missing"]
    assert np.isfinite(d["bite_striped_bass"]).all()  # still scores (neutral fallbacks)
    assert any(not s.ok for s in fc.sources)


def test_partial_data_marks_kinds():
    spot = get_spot("robert_moses_democrat_point")
    fc = assemble(spot, forecast_frame(1), _AllFail(), 0)
    kinds = {s.feature: s.kind.value for s in fc.sources}
    assert kinds.get("sun & moon") == "computed"
    assert kinds.get("weather") == "missing"
