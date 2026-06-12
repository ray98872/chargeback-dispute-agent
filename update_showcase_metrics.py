"""
Refreshes the numbers on the GitHub Pages showcase (docs/index.html) from
your most recent local pipeline run, so the public page always reflects a
real run on your machine.

Reads:  fintech.db, model/conformal.json
Writes: docs/index.html (in place, only the <span/b data-m=\"...\"> values)

Usage:
    python update_showcase_metrics.py
"""

import json
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).parent
HTML = ROOT / "docs" / "index.html"


def fmt_pct(x: float, dp: int = 1) -> str:
    s = f"{x * 100:.{dp}f}"
    return (s.rstrip("0").rstrip(".") if "." in s else s) + "%"


def main() -> None:
    conn = sqlite3.connect(ROOT / "fintech.db")
    total = conn.execute("SELECT COUNT(*) FROM disputes").fetchone()[0]
    n_auto = conn.execute(
        "SELECT COUNT(*) FROM disputes WHERE status='Resolved - Auto Refunded'"
    ).fetchone()[0]
    n_human = conn.execute(
        "SELECT COUNT(*) FROM disputes WHERE status='Requires Human Review'"
    ).fetchone()[0]
    auto_acc = conn.execute(
        "SELECT AVG(ai_category = true_category) FROM disputes "
        "WHERE status='Resolved - Auto Refunded'"
    ).fetchone()[0] or 0.0
    conn.close()

    meta = json.loads((ROOT / "model" / "conformal.json").read_text())
    processed = (n_auto + n_human) or 1

    values = {
        "n": str(total),
        "auto_rate": fmt_pct(n_auto / processed),
        "auto_acc": fmt_pct(auto_acc, 1),
        "escalated": str(n_human),
        "human_rate": fmt_pct(n_human / processed),
        "coverage": fmt_pct(meta["conformal_coverage_at_95"]),
        "set_size": f"{meta['avg_prediction_set_size']:.2f}",
        "test_acc": fmt_pct(meta["test_accuracy"]),
    }

    html = HTML.read_text(encoding="utf-8")
    for key, val in values.items():
        html, n = re.subn(
            rf'(data-m="{key}"[^>]*>)[^<]*(<)', rf"\g<1>{val}\g<2>", html
        )
        print(f"  {key:<10} -> {val:<8} ({n} occurrence{'s' if n != 1 else ''})")
    HTML.write_text(html, encoding="utf-8")
    print(f"Updated {HTML}")


if __name__ == "__main__":
    main()
