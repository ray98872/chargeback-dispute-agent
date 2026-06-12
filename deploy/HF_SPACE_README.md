---
title: Chargeback Dispute Agent
emoji: ⚖️
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Autonomous Chargeback & Dispute Resolution Agent — Live Demo

Internal-ops dashboard for an autonomous dispute resolution agent.
DistilBERT + split conformal prediction; the agent auto-refunds only when
calibrated confidence exceeds 95%.

Source & write-up: https://github.com/ray98872/chargeback-dispute-agent

> **Note for deployment:** copy this file to the root of your Hugging Face
> Space as `README.md` (the YAML front-matter configures the Space), along
> with `deploy/Dockerfile` (to the root as `Dockerfile`), `app.py`,
> `requirements.txt`, `.streamlit/`, `fintech.db`, and `model/` (via git-lfs)
> for the live playground tab.
