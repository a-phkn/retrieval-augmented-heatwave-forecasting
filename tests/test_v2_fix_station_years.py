"""v2_fix/station_safe_years.py: the station files are cut line by line, and the runners' redirected settings exist."""
import importlib

from v2_fix import station_safe_years as fx

TIDY = "date,tmax_c,tmin_c,prcp_mm,tmax_qflag,tmax_sflag\r\n2010-12-31,21.4,,0.0,,S\r\n2011-01-01,20.9,,0.0,,S\r\n"
GHCN = ('"STATION","DATE","LATITUDE","LONGITUDE","ELEVATION","NAME","PRCP","PRCP_ATTRIBUTES","TMAX","TMAX_ATTRIBUTES"\n'
        '"IN005010600","2010-06-01","23.067","72.633","55.0","AHMADABAD, IN","    0",",,S","  412",",,S"\n'
        '"IN005010600","2012-06-01","23.067","72.633","55.0","AHMADABAD, IN","    0",",,S","  420",",,S"\n')


def test_rows_after_the_cut_year_are_dropped_and_kept_lines_are_unchanged(tmp_path):
    for text, col in ((TIDY, 0), (GHCN, 1)):
        src, dst = tmp_path / "src.csv", tmp_path / "out" / "dst.csv"
        src.write_bytes(text.encode("utf-8"))
        assert fx.cut_csv(src, dst, 2010, col) == 1
        lines = text.encode("utf-8").splitlines(keepends=True)
        assert dst.read_bytes() == b"".join(lines[:2])  # the header and the 2010 row, byte for byte
        assert fx.cut_csv(src, dst, 9999, col) == 0 and dst.read_bytes() == src.read_bytes()


def test_the_settings_the_driver_redirects_still_exist_in_the_study_code():
    for module, attribute in fx.PATCHED:
        assert hasattr(importlib.import_module(module), attribute), f"{module}.{attribute} is gone"


def test_values_match_to_floating_point_noise_but_not_beyond():
    a = {"x": {"ci_low": 0.02072144511459397, "n": 12, "ok": True, "nan": float("nan")}, "l": [1.0, "a"]}
    assert fx.same_values(a, {"x": {**a["x"], "ci_low": 0.020721445114593964}, "l": [1.0, "a"]})[0]
    assert not fx.same_values(a, {"x": {**a["x"], "ci_low": 0.0208}, "l": [1.0, "a"]})[0]
    assert not fx.same_values(a, {"x": {**a["x"], "ok": False}, "l": [1.0, "a"]})[0]
    assert fx.same_values(a, {"x": a["x"]})[2] == ["/l[0]", "/l[1]"]  # missing keys are reported


def test_the_comparison_reads_both_versions():
    h = {"moderate/acclimatised": {"alert_days": 10, "supported": False,
                                   "primary_diff": {"estimate": 0.1, "ci_low": 0.0, "ci_high": 0.2}}}
    empty = {"h_a_delhi.json": h, "graded_delhi.json": {}, "station_checks.json": {}, "alert_gap.json": {}}
    new = {**empty, "h_a_delhi.json": {"moderate/acclimatised": {**h["moderate/acclimatised"], "alert_days": 7}}}
    text = fx.comparison(empty, new, 2010, {"safdarjung": 5})
    assert "| moderate/acclimatised | 10 -> 7 | +0.100 [+0.000, +0.200] -> +0.100 [+0.000, +0.200] |" in text
