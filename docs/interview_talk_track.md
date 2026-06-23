# Interview Talk Track

## 30-Second Explanation

Revenue Risk Intelligence Agent is a customer-success decision-support system. It predicts churn risk, estimates revenue at risk, retrieves synthetic account evidence, recommends retention actions, drafts a customer-success email, and shows everything in a Streamlit dashboard.

## 60-Second Explanation

I built this project to show how applied data science can move beyond a model notebook. It combines a scikit-learn churn model, a revenue-at-risk scoring engine, RAG over synthetic support and account notes, an agent workflow, groundedness checks, human feedback logging, FastAPI, Streamlit, Docker, CI, and deployment-ready documentation. The default demo works without paid API keys using local TF-IDF retrieval and a mock LLM provider.

## 2-Minute Technical Explanation

The system starts by generating synthetic customer-success data and synthetic account documents. A scikit-learn pipeline preprocesses numeric and categorical features and trains a churn classifier. The scoring layer converts churn probability into risk band, revenue at risk, top drivers, and recommended action category. The RAG layer uses a hybrid retriever interface with TF-IDF fallback and metadata filtering by customer ID and document type. The agent combines the customer profile, structured risk score, retrieved evidence, and user question into a grounded account answer with caveats, recommended actions, and an optional email draft. FastAPI exposes the workflow through endpoints, while Streamlit provides the business dashboard, what-if simulator, account brief export, evaluation, observability, and feedback views.

## Business Explanation

The business value is prioritisation and action. A customer-success team can identify high-risk accounts, see why revenue is at risk, inspect supporting evidence, decide what intervention to take, and export a concise account brief.

## ML Explanation

The ML model predicts churn on synthetic customer-success features such as product usage, support tickets, payment delay, NPS, contract type, onboarding completion, tenure, and account health. The model is evaluated with ROC-AUC, precision, recall, F1, and confusion matrix.

## RAG Explanation

The RAG layer retrieves synthetic account evidence from support summaries, call notes, contract notes, onboarding notes, product usage summaries, and retention playbooks. Metadata filtering keeps retrieved evidence account-specific.

## Evaluation Explanation

Evaluation includes model metrics, RAG evaluation questions, precision@k, recall@k, expected theme and document-type matches, latency, evidence coverage, and groundedness heuristics.

## Observability Explanation

Each agent run logs run ID, timestamp, customer ID, question, retrieved document IDs, risk band, latency, response length, and provider mode. Feedback is captured separately with rating and reason.

## What Was Hardest

The hardest part was making the system feel like a coherent decision product rather than disconnected ML, RAG, and dashboard components.

## What I Would Improve Next

I would add labelled human relevance judgments, optional embedding retrieval, richer time-series product usage data, and production-grade authentication and monitoring.

## Why This Is Not Just A Chatbot

It combines structured model output, retrieval-grounded evidence, deterministic recommendations, evaluation, observability, feedback, account brief export, and what-if simulation. The user gets a business workflow, not only a text response.

## How Hallucination Risk Is Reduced

The agent cites retrieved evidence, includes caveats when evidence is weak, uses structured risk scores for key claims, and has a deterministic groundedness evaluator.

## How I Would Productionise It

I would connect real CRM and product telemetry, add access control, move logs to a durable database, add tracing, monitor model drift, add human review workflows, and validate recommendations with customer-success stakeholders.

