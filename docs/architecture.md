# Architecture

The Revenue Risk Intelligence Agent is a synthetic, end-to-end customer-success decision system.

## Flow

1. Generate synthetic customer account data and synthetic evidence documents.
2. Train a churn model on structured customer-success signals.
3. Score each customer for churn probability and revenue at risk.
4. Retrieve relevant support, contract, onboarding, usage, and retention evidence.
5. Use an agent workflow to produce grounded explanations and next-best actions.
6. Serve the workflow through FastAPI and a Streamlit dashboard.
7. Log each run for evaluation and observability.

## Design Principles

- Synthetic/demo data only.
- Local-first implementation that runs without an API key.
- Clear separation between ML, retrieval, agent reasoning, API, and dashboard layers.
- Explainable outputs for customer-success and analytics audiences.

