"""Build the README result tables from the saved evaluation JSON.

Aggregates the three TD3 seeds into mean and range rather than listing them
separately, and keeps the per-seed numbers available in the JSON. Writes
Markdown to stdout so the README is generated from the data, not typed.

Run:  python scripts/make_tables.py > results/tables.md
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nav.stats import wilson  # noqa: E402

LABEL = {"rpp": "D* + RPP", "mpc": "D* + MPC",
         "td3_waypoint": "D* + TD3", "td3_goal": "TD3 end-to-end"}
ORDER = ["rpp", "mpc", "td3_waypoint", "td3_goal"]


def family(method: str) -> str:
    return method.rsplit("_s", 1)[0] if method.startswith("td3_") else method


def load(tags: list[str]) -> list[dict]:
    rows = []
    for t in tags:
        p = ROOT / "results" / f"{t}.json"
        if p.exists():
            rows += json.loads(p.read_text())["episodes"]
    return rows


def cell(eps: list[dict]) -> str:
    """success as n/N with a Wilson interval, averaged over seeds of a family."""
    by_seed = defaultdict(list)
    for e in eps:
        by_seed[e["method"]].append(e)
    ks, ns = [], []
    for m, es in by_seed.items():
        ks.append(sum(e["reached"] for e in es))
        ns.append(len(es))
    n = ns[0]
    if len(ks) == 1:
        lo, hi = wilson(ks[0], n)
        return f"{ks[0]}/{n} [{100*lo:.0f}, {100*hi:.0f}]"
    return f"{np.mean(ks):.1f}/{n} ({min(ks)}-{max(ks)})"


def fails(eps: list[dict]) -> str:
    r = sum(e["hit"] == "static" for e in eps)
    w = sum(e["hit"] == "worker" for e in eps)
    t = sum(e["timeout"] for e in eps)
    parts = [f"{v} {k}" for k, v in (("rack", r), ("worker", w), ("timeout", t)) if v]
    return ", ".join(parts) or "none"


def median_time(eps: list[dict]) -> str:
    ok = [e["time"] for e in eps if e["reached"]]
    return f"{np.median(ok):.1f}" if ok else "n/a"


def table(rows: list[dict], group_key: str, header: str) -> None:
    groups = sorted({r[group_key] for r in rows}, reverse=(group_key == "aisle"))
    fams = [f for f in ORDER if any(family(r["method"]) == f for r in rows)]
    print(f"| {header} | " + " | ".join(LABEL[f] for f in fams) + " |")
    print("|---|" + "---|" * len(fams))
    for g in groups:
        sub = [r for r in rows if r[group_key] == g]
        extra = ""
        if group_key == "aisle":
            nrows = sorted({r.get("rows", 4) for r in sub})[0]
            extra = f" ({nrows} rows)"
        print(f"| {g}{extra} | " + " | ".join(
            cell([r for r in sub if family(r["method"]) == f]) for f in fams) + " |")
    print()
    print(f"| {header} | " + " | ".join(f"{LABEL[f]} median s" for f in fams) + " |")
    print("|---|" + "---|" * len(fams))
    for g in groups:
        sub = [r for r in rows if r[group_key] == g]
        print(f"| {g} | " + " | ".join(
            median_time([r for r in sub if family(r["method"]) == f]) for f in fams) + " |")
    print()
    print(f"Failures by cause, pooled over seeds ({header}):")
    print()
    print(f"| {header} | " + " | ".join(LABEL[f] for f in fams) + " |")
    print("|---|" + "---|" * len(fams))
    for g in groups:
        sub = [r for r in rows if r[group_key] == g]
        print(f"| {g} | " + " | ".join(
            fails([r for r in sub if family(r["method"]) == f]) for f in fams) + " |")
    print()


def main() -> None:
    main_rows = load(["w04", "w8"])
    print("### Workers, at the original 3.0 m aisle\n")
    print("Success as n/50 with a Wilson 95% interval for a single policy, or as "
          "the mean over three training seeds with the range in brackets.\n")
    table(main_rows, "workers", "workers")

    sweep = load(["sweep"]) + [r for r in main_rows if r["workers"] == 4]
    print("### Density, at 4 workers\n")
    print("A narrower aisle adds rows rather than opening a bypass, so occupancy "
          "rises from 12.3% to 18.4% and free navigable cells fall from 68% to 52%. "
          "One policy per method here. **The TD3 policies were trained only at the "
          "3.0 m, four-row layout, so every narrower column is out of distribution "
          "for them and in distribution for the two classical controllers.**\n")
    table(sweep, "aisle", "aisle m")


if __name__ == "__main__":
    main()
