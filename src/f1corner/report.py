"""Markdown report from analyzed events (team pace, corner-class profile, Tests A/B, robustness)."""

from __future__ import annotations

import pandas as pd

from .robustness import _md


def pivot_pace(base: dict) -> pd.DataFrame:
    p = base["pace"].rename("pace_pct").reset_index()
    t = p.pivot(index="team", columns="event", values="pace_pct")
    t.insert(0, "mean", t.mean(axis=1))
    return t.sort_values("mean").reset_index()


def _pivot_profile(base: dict) -> pd.DataFrame:
    return base["season_profile"].unstack("cls").reset_index()


def build(base: dict, metas: list[dict], missing: list[str], flags: pd.DataFrame | None) -> str:
    L = ["# F1 Corner DNA report", "", "## Scope", ""]
    L.append(
        _md(
            pd.DataFrame(
                [
                    (m["event"], m["pole_driver"], m["lap_s"], m["n_teams"], m["n_corners"])
                    for m in metas
                ],
                columns=["event", "pole", "pole_lap_s", "teams", "corners"],
            )
        )
    )
    if missing:
        L += ["", f"Skipped (no derived data, wet or failed): {', '.join(missing)}"]
    L += [
        "",
        f"Corner classes: GMM, k = {base['k']} (BIC), fit on the earliest season's corners.",
        "",
        "## Team pace",
        "",
        "Median % deficit to the best team per corner (better teammate). Lower = faster.",
        "",
        _md(pivot_pace(base), "{:.2f}"),
        "",
        "## Corner-class profile",
        "",
        "Mean (corner delta - team pace) per class, averaged over events. Negative = stronger "
        "than the team's own average in that class.",
        "",
        _md(_pivot_profile(base), "{:+.2f}"),
    ]
    for key, title in (
        ("test_a", "Test A: season-to-season"),
        ("test_b", "Test B: first vs second half"),
    ):
        L += ["", f"## {title}", ""]
        t = base.get(key)
        if t is None:
            L.append("Not computable with the selected events (needs more events/seasons).")
        else:
            L += [_md(t, "{:.2f}"), "", "Only ~10 teams per correlation: intervals are wide."]
    L += ["", "## Prediction", "", "Not computable here: needs a per-round quali gap table."]
    if flags is not None:
        shown = flags[~flags["default"].eq("not computable")]
        L += [
            "",
            "## Robustness",
            "",
            f"{int(shown['changed'].sum())} of {len(shown)} comparisons flagged as changed.",
            "",
            _md(shown),
        ]
    return "\n".join(L) + "\n"
