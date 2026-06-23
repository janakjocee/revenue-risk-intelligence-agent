# Revenue Risk Intelligence Agent

A portfolio-grade AI/data science project for customer-success teams. The system combines churn prediction, revenue-at-risk scoring, retrieval-augmented evidence, agentic recommendations, observability, and a Streamlit dashboard.

All data in this repository is synthetic demo data.

## Why This Matters

Customer-success teams need to know which accounts are at risk, why the risk exists, what evidence supports the assessment, and what action to take next. This project turns scattered account signals into a decision-support workflow that is explainable, testable, and usable by business stakeholders.

## Planned Capabilities

- Synthetic customer-success dataset generation
- Churn-risk machine learning model
- Revenue-at-risk scoring engine
- Synthetic document corpus for RAG
- Local fallback retrieval that works without paid APIs
- Agent workflow for grounded explanations and next-best actions
- FastAPI backend
- Streamlit business dashboard
- Evaluation and observability layer
- Docker and GitHub Actions CI

## Architecture

```text
Synthetic customer data + synthetic notes
        |
        v
ML churn model -----> risk scoring engine
        |                    |
        v                    v
RAG corpus + retriever -> agent workflow -> FastAPI + Streamlit
                                  |
                                  v
                         observability logs
```

## Tech Stack

Python, pandas, NumPy, scikit-learn, FastAPI, Streamlit, pytest, Docker, and a local TF-IDF retrieval fallback.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Further setup and demo instructions will be expanded as each milestone lands.

