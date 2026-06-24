# Deployment Guide

This project is designed to run as a public demo without paid API keys. By default it uses synthetic data, a saved scikit-learn model, local TF-IDF retrieval, and a deterministic mock LLM provider.

## Streamlit Community Cloud

Recommended public demo target: Streamlit Community Cloud.

Current status: deployed at https://revenue-risk.streamlit.app/

1. Push the repository to GitHub.
2. Go to Streamlit Community Cloud and create a new app.
3. Select repository `janakjocee/revenue-risk-intelligence-agent`.
4. Set the main file path to `app/streamlit_app.py`.
5. Select Python 3.11 in the Streamlit Community Cloud app settings.
6. No secrets are required for the default demo.
7. Deploy the app.

Note: Streamlit Community Cloud chooses the Python version from the app settings UI. The repo keeps `runtime.txt` for platforms that support it, but the live Streamlit deployment uses the UI-selected Python 3.11 setting.

The `packages.txt` file is intentionally empty because this demo does not require apt system packages. Do not add comments to `packages.txt`; Streamlit Cloud passes every line to `apt-get`.

Optional secrets for LLM mode:

```toml
OPENAI_API_KEY="..."
OPENAI_BASE_URL="https://api.openai.com/v1"
OPENAI_MODEL="gpt-4o-mini"
LLM_PROVIDER="openai"
```

If no secrets are set, the app uses:

```text
LLM_PROVIDER=mock
EMBEDDING_PROVIDER=tfidf
```

## FastAPI Deployment Option

FastAPI can be deployed separately on Render, Railway, Fly.io, or another container host.

Suggested command:

```bash
python3 -m uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
```

For Render:

- Build command: `pip install -r requirements.txt`
- Start command: `python3 -m uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
- Environment: Python 3.11
- Secrets: optional only

## Docker Deployment

Local Docker Compose runs both services:

```bash
docker compose up --build
```

Ports:

- FastAPI: `http://localhost:8000`
- Streamlit: `http://localhost:8501`

Verification note: Docker commands could not be executed in the final release environment because the `docker` CLI was not installed. The Dockerfile and Compose configuration were reviewed, and the expected command is shown above.

## Environment Variables

```bash
OPENAI_API_KEY=
OPENAI_BASE_URL=
OPENAI_MODEL=
LLM_PROVIDER=mock
EMBEDDING_PROVIDER=tfidf
DATA_DIR=data
MODEL_PATH=data/processed/churn_model.joblib
RETRIEVER_PATH=data/processed/tfidf_retriever.joblib
OBSERVABILITY_LOG_PATH=data/processed/agent_runs.jsonl
LOG_LEVEL=INFO
```

## Regenerate Demo Artifacts

```bash
make data
make train
make score
make retriever
make evaluate
make test
```

## Screenshot Capture

The repository includes a full-dashboard screenshot at `docs/assets/overview-dashboard.png`.

After the Streamlit deployment is live, capture additional screenshots from:

- Overview
- Account Deep Dive
- Ask The Agent
- What-If Simulator
- Evaluation
- Observability

Recommended naming:

```text
docs/assets/overview-dashboard.png
docs/assets/account-deep-dive.png
docs/assets/ask-agent.png
docs/assets/what-if-simulator.png
docs/assets/evaluation.png
docs/assets/observability.png
```

Use browser full-page screenshots where possible so the portfolio page shows the dashboard as a complete product rather than cropped components.

## Missing Artifact Troubleshooting

If the Streamlit app reports missing artifacts, run:

```bash
make data
make train
make score
make retriever
make evaluate
```

The dashboard needs these files:

- `data/processed/customer_features.csv`
- `data/processed/customer_risk_scores.csv`
- `data/processed/churn_model.joblib`
- `data/processed/tfidf_retriever.joblib`

## Known Limitations

- All data is synthetic.
- The default retrieval backend is TF-IDF, not a paid embedding service.
- The default LLM provider is deterministic mock mode.
- Streamlit Community Cloud is best for the dashboard; production FastAPI hosting should use a separate API service with auth, logging, and persistence.
- Docker was not runtime-tested in the Codex environment because Docker was unavailable there.
