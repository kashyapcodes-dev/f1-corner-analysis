import pandas as pd

from f1corner.lapfilter import car_issue_flags, filter_laps, is_wet, pole_lap


def lap(driver="VER", team="RB", n=1, t=90.0, **kw):
    d = dict(
        driver=driver, team=team, lap_number=n, lap_time_s=t, s1_s=t / 3, s2_s=t / 3, s3_s=t / 3,
        pit_in=False, pit_out=False, green=True, deleted=False, compound="SOFT",
    )  # fmt: skip
    return d | kw


def run(rows, **kw):
    return filter_laps(pd.DataFrame(rows), **kw)


def reason(rows, i=0):
    return run(rows).laps["drop_reason"].iloc[i]


def test_clean_lap_kept():
    r = run([lap()])
    assert len(r.kept) == 1 and r.reasons.sum() == 0


def test_pit_in_and_out_dropped():
    assert reason([lap(pit_in=True)]) == "pit"
    assert reason([lap(pit_out=True)]) == "pit"


def test_not_green_and_unknown_dropped():
    assert reason([lap(green=False)]) == "not_green"
    assert reason([lap(green=pd.NA)]) == "not_green"


def test_deleted_dropped_na_means_not_deleted():
    assert reason([lap(deleted=True)]) == "deleted"
    assert reason([lap(deleted=pd.NA)]) is pd.NA or pd.isna(reason([lap(deleted=pd.NA)]))


def test_sectors_must_sum():
    assert reason([lap(s3_s=40.0)]) == "sectors"
    assert reason([lap(s3_s=pd.NA)]) == "sectors"
    assert pd.isna(reason([lap(s1_s=30.002, s2_s=30.0, s3_s=30.0, t=90.0)]))  # within tol


def test_no_time():
    assert reason([lap(t=pd.NA, s1_s=pd.NA)]) == "no_time"


def test_102_percent_rule_per_driver():
    rows = [lap(n=1, t=90.0), lap(n=2, t=91.7), lap(n=3, t=92.0), lap("HAM", "F", 1, t=100.0)]
    r = run(rows)
    assert list(r.laps["drop_reason"].fillna("ok")) == ["ok", "ok", "slow", "ok"]  # 91.8 limit
    assert r.reasons["slow"] == 1


def test_best_lap_ignores_deleted_for_102_rule():
    # fast-but-deleted lap must not make the remaining laps look slow
    rows = [lap(n=1, t=80.0, deleted=True), lap(n=2, t=90.0)]
    r = run(rows)
    assert list(r.laps["drop_reason"].fillna("ok")) == ["deleted", "ok"]


def test_first_failing_rule_recorded_and_counts():
    r = run([lap(pit_out=True, green=False, deleted=True), lap(n=2)])
    assert r.laps["drop_reason"].iloc[0] == "pit"
    assert r.reasons.to_dict() == dict(no_time=0, pit=1, not_green=0, deleted=0, sectors=0, slow=0)


def test_wet_session_excluded_whole():
    rows = [lap(), lap(n=2, compound="INTERMEDIATE")]
    assert is_wet(pd.DataFrame(rows))
    r = run(rows)
    assert r.wet_session and r.kept.empty and r.reasons["wet_session"] == 2


def test_qualifying_keeps_single_best_per_driver():
    rows = [lap(n=1, t=90.0), lap(n=2, t=89.5), lap("HAM", "F", 1, t=91.0)]
    r = run(rows, qualifying=True)
    assert sorted(zip(r.kept["driver"], r.kept["lap_number"])) == [("HAM", 1), ("VER", 2)]
    assert r.reasons["not_best"] == 1
    assert pole_lap(r.kept)["driver"] == "VER"


def test_teammate_gap_flag_over_1pct():
    kept = pd.DataFrame(
        [lap("VER", "RB", t=90.0), lap("TSU", "RB", t=91.0), lap("NOR", "M", t=90.0),
         lap("PIA", "M", t=90.8)]
    )  # fmt: skip
    f = car_issue_flags(kept)
    assert list(f["driver"]) == ["TSU"] and f["gap_pct"].iloc[0] > 1.0
