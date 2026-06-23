# Portfolio Case Study

## Problem

Customer-success teams need to prioritise renewal risk but often rely on scattered signals across product usage, support tickets, payment behaviour, contract notes, and relationship context.

## Why It Matters

Churn risk is a revenue problem and an operating problem. Teams need to know not just which account is risky, but why, what evidence supports the view, and what action to take next.

## Data

The project uses synthetic customer-success data with account, usage, support, payment, sentiment, contract, onboarding, health, and churn fields. It also generates synthetic support summaries, customer-success notes, contract notes, pricing notes, onboarding notes, usage summaries, renewal risk notes, and retention playbooks.

## Machine Learning Approach

A scikit-learn pipeline trains a churn classifier with preprocessing for numeric and categorical features. The model outputs churn probability, which is converted into a business risk band and revenue-at-risk estimate.

## RAG Approach

The RAG layer uses a hybrid retrieval interface with a local TF-IDF fallback and optional semantic retrieval extension point. Retrieval supports customer-level and document-type metadata filtering so evidence remains specific to the selected account.

## Agent Workflow

The agent combines:

- customer profile
- churn score
- revenue at risk
- top structured drivers
- retrieved evidence
- user question

It returns a grounded answer, cited evidence, caveats, recommended actions, and an optional customer-success email draft.

The workflow also includes an optional OpenAI-compatible provider layer. Without an API key, the app uses a deterministic mock provider so the project remains runnable for recruiters and reviewers.

## Evaluation

The model is evaluated with ROC-AUC, precision, recall, F1, and confusion matrix. Retrieval is evaluated with customer-scoped risk-theme queries and latency checks.

## Observability

Every agent run is logged to JSONL with timestamp, question, customer ID, retrieved document IDs, risk band, latency, response length, and feedback placeholder.

The upgraded version adds human feedback logging with run ID, customer ID, rating, feedback reason, question, and risk band. This creates a simple human-in-the-loop improvement loop.

## Product Features

- Executive dashboard for risk and revenue-at-risk monitoring
- Customer workspace with agent answers and cited evidence
- What-if simulator for intervention planning
- Markdown account brief export
- Evaluation and observability pages
- Human feedback collection

## Business Impact

The workflow shows how an applied data scientist can connect predictive modelling to operational action: prioritising accounts, explaining risk, surfacing evidence, and helping teams act faster.

## What I Learned

This project demonstrates the importance of linking ML outputs to evidence, user workflows, evaluation, and observability. A prediction alone is rarely enough; business users need context and recommended action.

## Limitations

The data is synthetic, the semantic retrieval provider is an extension point, and the default agent is deterministic. A production version would need real CRM/product data, stronger evaluation, privacy controls, and human-in-the-loop governance.

## Next Steps

- Add optional embedding and LLM providers.
- Expand evaluation with labelled relevance judgments.
- Add richer account histories and time-series usage patterns.
- Add deployment-ready authentication and monitoring.
- Create Power BI-ready export tables for stakeholder reporting.
