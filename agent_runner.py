"""
Agentic Workflow: autonomous dispute resolution loop
====================================================
Connects the conformal classifier to the SQLite database:

  1. Pull every dispute with status 'Pending'.
  2. Classify its free-text with the conformal DistilBERT pipeline.
  3. Decide:
       - calibrated confidence > 0.95 AND singleton prediction set
            -> autonomous action: UPDATE status to 'Resolved - Auto Refunded'
       - otherwise
            -> escalate: UPDATE status to 'Requires Human Review'
  4. Every decision is written to `agent_audit_log` - an autonomous agent
     touching money movement must leave an audit trail.

The agent never reads `true_category`; it acts only on the model output.

Usage:
    python agent_runner.py [--threshold 0.95] [--limit N] [--dry-run]
"""

import argparse
import sqlite3
import time
from datetime import datetime, timezone

from ai_classifier import ConformalDisputeClassifier

DB_PATH = "fintech.db"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(sep=" ", timespec="seconds")


def run_agent(threshold: float, limit: int | None, dry_run: bool) -> None:
    clf = ConformalDisputeClassifier()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")

    query = "SELECT dispute_id, dispute_text FROM disputes WHERE status = 'Pending' ORDER BY dispute_id"
    if limit:
        query += f" LIMIT {int(limit)}"
    pending = conn.execute(query).fetchall()

    print(f"Agent online | device={clf.device} | threshold={threshold:.0%} "
          f"| pending disputes={len(pending)}{' | DRY RUN' if dry_run else ''}")
    print("-" * 78)

    n_auto, n_human = 0, 0
    t0 = time.perf_counter()

    for dispute_id, text in pending:
        result = clf.classify(text)
        decisive = result.is_decisive(threshold)

        if decisive:
            new_status = "Resolved - Auto Refunded"
            action = "AUTO_REFUND"
            n_auto += 1
        else:
            new_status = "Requires Human Review"
            action = "ESCALATE_TO_HUMAN"
            n_human += 1

        detail = (
            f"class={result.predicted_class} "
            f"confidence={result.confidence:.4f} "
            f"credibility={result.credibility:.4f} "
            f"prediction_set={result.prediction_set}"
        )

        if not dry_run:
            conn.execute(
                """
                UPDATE disputes
                SET status = ?, ai_category = ?, ai_confidence = ?,
                    ai_credibility = ?, resolved_at = ?
                WHERE dispute_id = ?
                """,
                (
                    new_status,
                    result.predicted_class,
                    result.confidence,
                    result.credibility,
                    now() if decisive else None,
                    dispute_id,
                ),
            )
            conn.execute(
                "INSERT INTO agent_audit_log (dispute_id, action, detail, logged_at) "
                "VALUES (?, ?, ?, ?)",
                (dispute_id, action, detail, now()),
            )
            conn.commit()

        flag = "AUTO " if decisive else "HUMAN"
        print(f"[{flag}] dispute #{dispute_id:>3} -> {result.predicted_class:<17} "
              f"conf={result.confidence:.4f} set={len(result.prediction_set)}")

    elapsed = time.perf_counter() - t0
    total = len(pending) or 1
    print("-" * 78)
    print(f"Processed {len(pending)} disputes in {elapsed:.1f}s "
          f"({elapsed / total:.2f}s each)")
    print(f"  Auto-resolved:         {n_auto} ({n_auto / total:.1%})")
    print(f"  Sent to human review:  {n_human} ({n_human / total:.1%})")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the dispute resolution agent")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Calibrated confidence required for autonomous action")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N pending disputes")
    parser.add_argument("--dry-run", action="store_true",
                        help="Classify but do not write to the database")
    args = parser.parse_args()
    run_agent(args.threshold, args.limit, args.dry_run)
