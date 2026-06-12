# Autonomous Chargeback & Dispute Resolution Agent

A production-grade local MVP of an autonomous fintech agent: a fine-tuned
DistilBERT classifies free-text customer disputes, **split conformal
prediction** converts raw softmax scores into calibrated uncertainty with a
finite-sample coverage guarantee, and an agent loop autonomously executes
refund actions in SQLite — but *only* when the statistics permit it.

Runs entirely locally. Zero paid APIs. CPU is enough.

**Write-up & showcase:** https://ray98872.github.io/chargeback-dispute-agent/

## Architecture

`generate_data.py` seeds **fintech.db** (SQLite: transactions, disputes, agent_audit_log).
`train_model.py` fine-tunes DistilBERT on labelled dispute text and computes conformal
calibration scores. `agent_runner.py` pulls Pending disputes, gets calibrated confidence
from `ai_classifier.py`, and either executes the refund UPDATE itself (confidence > 95%)
or escalates to the human queue. `app.py` is the Streamlit ops dashboard over the same DB.

## Why conformal prediction (and not just softmax)?

A softmax of 0.97 is **not** a probability you can act on — neural networks
are notoriously miscalibrated. Split conformal prediction fixes this with a
distribution-free guarantee: hold out a calibration set, compute
nonconformity scores `1 − p(true class)`, then for any new dispute compute a
p-value per class against that empirical distribution.

The agent thresholds on **confidence = 1 − (second largest p-value)**: it
acts autonomously only when every competing class is rejected at the 95%
level *and* the 95% prediction set is a singleton. The result: the agent's
error rate on autonomous actions is statistically controlled, and everything
ambiguous lands in front of a human — which is exactly how you'd gate an
agent that moves money.

## Quickstart

```bash
# 1. Environment (Windows PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# CPU-only torch (smaller/faster install):
#   pip install torch --index-url https://download.pytorch.org/whl/cpu

# (macOS/Linux: source .venv/bin/activate)

# 2. Generate the synthetic bank
python generate_data.py

# 3. Fine-tune DistilBERT + conformal calibration (~3-10 min on CPU)
python train_model.py

# 4. Unleash the agent on 500 pending disputes
python agent_runner.py

# 5. Boot the ops dashboard
streamlit run app.py

# 6. (optional) refresh the showcase page numbers from this run
python update_showcase_metrics.py
```

## Files

| File | Role |
|---|---|
| `generate_data.py` | Synthetic data engine — 5,000 transactions, 500 disputes with hidden ground truth across Fraud / Merchant Error / Policy Violation |
| `train_model.py` | Plain-PyTorch fine-tune of `distilbert-base-uncased`, 60/20/20 stratified split, saves model + calibration scores |
| `ai_classifier.py` | Inference + conformal wrapper: p-values, credibility, calibrated confidence, prediction sets |
| `agent_runner.py` | The agent: reads Pending disputes, classifies, executes `UPDATE` autonomously above threshold, audit-logs every decision |
| `app.py` | Streamlit ops dashboard: automation rate, hours saved, auto-resolved table, human review queue, calibration metrics, live playground |
| `update_showcase_metrics.py` | Rewrites the numbers on `docs/index.html` from your latest local run |
| `docs/index.html` | GitHub Pages case-study page |

## Design decisions

- **The agent never sees ground truth.** `true_category` exists only for
  training and for the dashboard's backtest panel.
- **Calibration set is disjoint from training** — the independence is what
  makes the conformal guarantee hold.
- **Class overlap is deliberate.** Dispute templates share noise clauses so
  the classifier is imperfect and the escalation path actually exercises.
- **Audit log.** Autonomous financial actions without an audit trail is a
  compliance incident waiting to happen.
- **Thresholding on confidence, not softmax** — see above.

## Deploying the live demo (Hugging Face Spaces, free)

The dashboard only needs `app.py`, `fintech.db`, and optionally `model/`
(for the live playground tab). Create a Streamlit Space and push:

```bash
pip install -U "huggingface_hub[cli]"
hf auth login
# create a Space (SDK: Streamlit) at huggingface.co/new-space, then:
git clone https://huggingface.co/spaces/<you>/chargeback-agent hf-space
cp app.py requirements.txt fintech.db hf-space/
cp -r .streamlit model hf-space/        # model/ needs git lfs
cd hf-space && git lfs track "model/*.safetensors" "fintech.db"
git add . && git commit -m "deploy" && git push
```

See `deploy/HF_SPACE_README.md` for the Space front-matter.

---
*Portfolio project by [Ray Mahbub](https://github.com/ray98872) — MSc thesis:
GPU-accelerated conformal prediction.*
