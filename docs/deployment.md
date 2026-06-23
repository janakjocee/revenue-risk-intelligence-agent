# Deployment Guide

This project is designed to run as a public demo without paid API keys. By default it uses synthetic data, a saved scikit-learn model, local TF-IDF retrieval, and a deterministic mock LLM provider.

## Streamlit Community Cloud

Recommended public demo target: Streamlit Community Cloud.

1. Push the repository to GitHub.
2. Go to Streamlit Community Cloud and create a new app.
3. Select repository `janakjocee/revenue-risk-intelligence-agent`.
4. Set the main file path to `app/streamlit_app.py`.
5. Use Python 3.11. The repo includes `runtime.txt`.
6. No secrets are required for the default demo.
7. Deploy the app.

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
