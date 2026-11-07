"""Saved spots ("Your spots"): defaults, add/remove, and saving a searched place."""
import pytest

from app.schemas.env import Spot
from app.services import my_spots
from app.services.forecast import get_spot, load_spots

DEFAULTS = ["jones_beach_field_10", "sunken_meadow_field_3", "point_lookout_jetties",
            "jones_beach_west_end_2", "robert_moses_democrat_point", "jones_beach_green_island"]


@pytest.fixture(autouse=True)
def fresh_store():
    my_spots.STORE.unlink(missing_ok=True)
    yield
    my_spots.STORE.unlink(missing_ok=True)


def test_defaults_come_from_config():
    assert my_spots.saved_ids() == DEFAULTS
    favs = {s.id for s in load_spots() if s.favorite}
    assert favs == set(DEFAULTS)


def test_add_and_remove_persist():
    my_spots.remove("sunken_meadow_field_3")
    my_spots.add("jones_beach_field_2")
    my_spots.add("jones_beach_field_2")  # no duplicates
    ids = my_spots.saved_ids()
    assert "sunken_meadow_field_3" not in ids and ids.count("jones_beach_field_2") == 1
    assert ids[-1] == "jones_beach_field_2"
    assert get_spot("jones_beach_field_2").favorite and not get_spot("sunken_meadow_field_3").favorite


def test_empty_list_is_kept():
    my_spots.set_ids([])
    assert my_spots.saved_ids() == []
    assert not any(s.favorite for s in load_spots())


def test_save_searched_place_then_remove():
    spot = Spot(id="custom_40.7000_-72.9000", name="Smith Point", short="Smith Point", lat=40.7, lon=-72.9,
                environment="beach", water_body="ocean", facing_deg=170, tide_station="8516385",
                temp_sources=["ndbc:44025"], wave_sources=["ndbc:44025"], custom=True)
    my_spots.save_custom(spot, "Smith Point Bridge")
    s = get_spot(spot.id)
    assert s.favorite and s.custom and s.name == "Smith Point Bridge"
    assert spot.id in my_spots.saved_ids()
    stations = my_spots._read()["stations"]
    assert "44025" in stations["ndbc"] and "8516385" in stations["coops"]
    my_spots.remove(spot.id)
    assert spot.id not in {x.id for x in load_spots()}


def test_reorder_saved_spots():
    my_spots.move("robert_moses_democrat_point", -1)
    assert my_spots.saved_ids()[3:5] == ["robert_moses_democrat_point", "jones_beach_west_end_2"]
    my_spots.move("robert_moses_democrat_point", -10)          # clamps at the top
    assert my_spots.saved_ids()[0] == "robert_moses_democrat_point"
    my_spots.move("robert_moses_democrat_point", 1)
    assert my_spots.saved_ids()[:2] == ["jones_beach_field_10", "robert_moses_democrat_point"]
    my_spots.move("not_saved", 1)                              # unknown id: no change
    assert len(my_spots.saved_ids()) == len(DEFAULTS)
    fav_order = [s.id for s in load_spots() if s.favorite]     # config order is unchanged
    assert set(fav_order) == set(DEFAULTS)
