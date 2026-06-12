"""
Product Dashboard: internal dispute-operations tool
===================================================
Streamlit app for the customer support team. Reads `fintech.db` and shows:

  - Business metrics: total disputes, automation rate, estimated hours saved
  - Disputes the agent autonomously resolved (with calibrated confidence)
  - The human review queue, ordered by how urgently a human is needed
  - Model quality: backtest vs hidden ground truth + conformal coverage
  - Optional live playground to classify ad-hoc dispute text

Usage:
    streamlit run app.py
"""

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

DB_PATH = Path(__file__).parent / "fintech.db"
MODEL_DIR = Path(__file__).parent / "model"
MINUTES_SAVED_PER_AUTO_RESOLUTION = 25  # avg manual handling time per dispute

st.set_page_config(
    page_title="Dispute Resolution Agent",
    page_icon="⚖️",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #

@st.cache_data(ttl=30)
def load_disputes() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT d.dispute_id, d.status, d.ai_category, d.ai_confidence,
               d.ai_credibility, d.true_category, d.opened_at, d.resolved_at,
               d.dispute_text,
               t.transaction_id, t.customer_name, t.merchant, t.amount,
               t.currency, t.timestamp AS txn_time
        FROM disputes d
        JOIN transactions t ON t.transaction_id = d.transaction_id
        ORDER BY d.dispute_id
        """,
        conn,
    )
    conn.close()
    return df


@st.cache_data(ttl=30)
def load_audit_log() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM agent_audit_log ORDER BY log_id DESC LIMIT 200", conn
    )
    conn.close()
    return df


if not DB_PATH.exists():
    st.error("`fintech.db` not found. Run `python generate_data.py` first.")
    st.stop()

df = load_disputes()
auto = df[df.status == "Resolved - Auto Refunded"]
human = df[df.status == "Requires Human Review"]
pending = df[df.status == "Pending"]

# --------------------------------------------------------------------------- #
# Header + business metrics
# --------------------------------------------------------------------------- #

st.title("⚖️ Autonomous Chargeback & Dispute Resolution Agent")
st.caption(
    "DistilBERT fine-tuned on dispute text · split conformal prediction for "
    "calibrated uncertainty · autonomous SQL actions above 95% confidence"
)

total = len(df)
processed = len(auto) + len(human)
automation_rate = (len(auto) / processed) if processed else 0.0
hours_saved = len(auto) * MINUTES_SAVED_PER_AUTO_RESOLUTION / 60

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total disputes", f"{total:,}")
c2.metric("Auto-resolved", f"{len(auto):,}")
c3.metric("Automation rate", f"{automation_rate:.1%}")
c4.metric("Est. hours saved", f"{hours_saved:,.0f} h")
c5.metric("Awaiting human", f"{len(human):,}")

if len(pending):
    st.info(f"{len(pending)} disputes still pending — run `python agent_runner.py`.")

st.divider()

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #

tab_auto, tab_queue, tab_quality, tab_audit, tab_live = st.tabs(
    [
        "✅ Auto-resolved",
        "\U0001f9d1‍⚖️ Human review queue",
        "\U0001f4d0 Model quality",
        "\U0001f4dc Audit log",
        "\U0001f9ea Live playground",
    ]
)

TABLE_COLS = {
    "dispute_id": st.column_config.NumberColumn("Dispute", format="%d"),
    "merchant": "Merchant",
    "customer_name": "Customer",
    "amount": st.column_config.NumberColumn("Amount", format="%.2f"),
    "currency": "CCY",
    "ai_category": "AI classification",
    "ai_confidence": st.column_config.ProgressColumn(
        "Calibrated confidence", min_value=0.0, max_value=1.0, format="%.4f"
    ),
    "dispute_text": "Dispute text",
}

with tab_auto:
    st.subheader("Disputes autonomously resolved & refunded")
    st.caption(
        "Conformal confidence exceeded 95% and the 95% prediction set was a "
        "singleton — the agent executed the refund UPDATE itself."
    )
    if auto.empty:
        st.warning("Nothing auto-resolved yet. Run the agent.")
    else:
        st.dataframe(
            auto[list(TABLE_COLS)].sort_values("ai_confidence", ascending=False),
            column_config=TABLE_COLS,
            width="stretch",
            hide_index=True,
        )
        st.bar_chart(
            auto.ai_category.value_counts(),
            horizontal=True,
            color="#c97e4e",
        )

with tab_queue:
    st.subheader("Queue for the support team")
    st.caption(
        "The model could not reject competing explanations at the 95% level — "
        "exactly the cases where a human should decide. Sorted lowest "
        "confidence first."
    )
    if human.empty:
        st.success("Queue is empty \U0001f389")
    else:
        st.dataframe(
            human[list(TABLE_COLS)].sort_values("ai_confidence"),
            column_config=TABLE_COLS,
            width="stretch",
            hide_index=True,
        )

with tab_quality:
    st.subheader("Model quality & calibration")
    scored = df[df.ai_category.notna()]
    if scored.empty:
        st.warning("Run the agent first.")
    else:
        backtest_auto = (auto.ai_category == auto.true_category).mean() if len(auto) else 0
        backtest_all = (scored.ai_category == scored.true_category).mean()
        q1, q2, q3 = st.columns(3)
        q1.metric(
            "Accuracy on auto-resolved",
            f"{backtest_auto:.1%}",
            help="Agreement with hidden ground-truth labels on disputes the "
                 "agent actioned autonomously. This is the number that "
                 "justifies the 95% threshold.",
        )
        q2.metric("Accuracy on all classified", f"{backtest_all:.1%}")
        q3.metric(
            "Accuracy on escalated",
            f"{(human.ai_category == human.true_category).mean():.1%}" if len(human) else "—",
            help="Even escalated disputes carry a best-guess label for the "
                 "human reviewer to start from. Expected to be LOWER than the "
                 "auto-resolved accuracy — that gap is the point of the "
                 "conformal gate.",
        )
        st.caption("Calibrated confidence distribution (all classified disputes)")
        st.bar_chart(
            pd.cut(scored.ai_confidence, bins=[0, .5, .8, .9, .95, .99, 1.0])
            .value_counts()
            .sort_index()
            .rename(index=str),
            color="#c97e4e",
        )

        import json
        conf_path = MODEL_DIR / "conformal.json"
        if conf_path.exists():
            meta = json.loads(conf_path.read_text())
            st.caption("Training-time evaluation (held-out test set)")
            m1, m2, m3 = st.columns(3)
            m1.metric("Test accuracy", f"{meta.get('test_accuracy', 0):.1%}")
            m2.metric(
                "Conformal coverage @95%",
                f"{meta.get('conformal_coverage_at_95', 0):.1%}",
                help="Guaranteed ≥95% in expectation by construction; "
                     "empirical check on the held-out test set.",
            )
            m3.metric(
                "Avg prediction-set size",
                f"{meta.get('avg_prediction_set_size', 0):.2f}",
            )

with tab_audit:
    st.subheader("Agent audit trail")
    st.caption("Every autonomous decision is logged — last 200 entries.")
    audit = load_audit_log()
    if audit.empty:
        st.warning("No agent activity logged yet.")
    else:
        st.dataframe(audit, width="stretch", hide_index=True)

with tab_live:
    st.subheader("Classify a dispute live")
    if not (MODEL_DIR / "conformal.json").exists():
        st.warning("Trained model not found — run `python train_model.py`.")
    else:
        text = st.text_area(
            "Dispute text",
            "I cancelled this subscription weeks ago but they charged me "
            "again this month, which their terms say cannot happen.",
            height=110,
        )
        if st.button("Classify", type="primary"):
            from ai_classifier import ConformalDisputeClassifier

            @st.cache_resource
            def get_classifier():
                return ConformalDisputeClassifier()

            r = get_classifier().classify(text)
            l1, l2, l3 = st.columns(3)
            l1.metric("Predicted class", r.predicted_class)
            l2.metric("Calibrated confidence", f"{r.confidence:.4f}")
            l3.metric("Credibility", f"{r.credibility:.4f}")
            st.write("Conformal p-values per class:")
            st.json(r.p_values)
            if r.is_decisive():
                st.success("Agent would AUTO-RESOLVE this dispute.")
            else:
                st.warning("Agent would escalate to a human.")
