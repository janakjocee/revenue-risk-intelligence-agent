# Portfolio Case Study

## Problem

Customer-success teams need to prioritise renewal risk, but relevant signals are scattered across product usage, support tickets, payment behaviour, contract context, onboarding notes, relationship sentiment, and retention playbooks.

## Why It Matters

Churn is both a revenue problem and an operating problem. A team needs to know which customers are at risk, why they are at risk, how much revenue is exposed, what evidence supports the view, and what action should happen next.

## Data

The project uses synthetic demo data only. It includes structured customer-success fields such as contract value, monthly recurring revenue, tenure, product usage, ticket count, unresolved tickets, payment delay, NPS, contract type, last login, onboarding completion, account health, and churn label.

It also generates synthetic documents:

- support ticket summaries
- customer-success call notes
- pricing policy notes
- retention playbooks
- contract notes
- onboarding notes
- product usage summaries
- renewal risk notes

## Machine Learning Approach

A scikit-learn pipeline preprocesses numeric and categorical features and trains a churn classifier. Predictions are converted into business outputs: churn probability, risk band, revenue at risk, top risk drivers, and recommended action category.

## RAG Approach

The retrieval layer uses a hybrid retrieval interface with a local TF-IDF fallback and optional semantic extension point. Metadata filtering by customer and document type keeps evidence account-specific.

## Agent Workflow

The agent combines customer profile, churn score, revenue at risk, top risk drivers, retrieved evidence, and user question. It returns:

- answer
- cited evidence
- risk explanation
- recommended actions
- customer-success email draft
- caveats
- groundedness evaluation

The optional OpenAI-compatible provider can be enabled with environment variables, but mock/local mode is the default.

## Evaluation

The model is evaluated with ROC-AUC, precision, recall, F1, and confusion matrix. Retrieval is evaluated with a synthetic question set using precision@k, recall@k, expected theme match, expected document type match, latency, groundedness heuristic, and evidence coverage.

## Observability

Every agent run logs run ID, timestamp, customer ID, question, retrieved document IDs, retrieved document types, risk band, latency, response length, feedback placeholder, and provider mode.

## Feedback Loop

The Streamlit dashboard captures human feedback with rating and reason. Feedback is stored separately from run logs and summarised in the observability page.

## What-if Simulator

Users can adjust unresolved tickets, product usage, payment delay, NPS, account health, and contract type. The simulator recalculates churn probability, risk band, revenue at risk, and recommended actions.

## Account Brief Export

The dashboard can generate a Markdown account brief containing customer profile, risk score, revenue at risk, drivers, retrieved evidence, actions, email draft, and caveats.

## Deployment Readiness

The repo includes Streamlit Community Cloud support files, Docker, Docker Compose, Makefile commands, GitHub Actions CI, and a deployment guide. The public demo is designed to work without secrets.

## Business Impact

The project shows how applied data science can move from prediction to decision support. It connects model output with evidence, recommendations, feedback, and a business dashboard.

## Limitations

- Synthetic data only.
- TF-IDF is the default retrieval backend.
- Mock LLM provider is the default.
- No authentication or production data governance.
- Recommendations are decision-support guidance, not automated decisions.

## Next Steps

- Deploy the Streamlit dashboard publicly.
- Add optional embedding retrieval.
- Add labelled relevance judgments.
- Add real customer-success workflow research.
- Add durable database logging and production monitoring.

