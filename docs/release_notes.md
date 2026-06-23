# Release Notes

## v1.0 Portfolio Release

Revenue Risk Intelligence Agent is ready as a public portfolio project and recruiter-facing case study.

## Features

- Synthetic customer-success data and document generation
- Churn ML model
- Revenue-at-risk scoring
- Hybrid retrieval interface with TF-IDF fallback
- Optional OpenAI-compatible provider with mock fallback
- Agent workflow with cited evidence and caveats
- Groundedness evaluator
- RAG evaluation dataset
- Human feedback logging
- Account brief export
- What-if simulator
- FastAPI backend
- Streamlit dashboard
- Docker, Docker Compose, Makefile, GitHub Actions CI

## Test Status

The local pytest suite passes. GitHub Actions is configured to run tests on push and pull request.

## Deployment Status

The repository is prepared for Streamlit Community Cloud deployment. Deployment was not completed from this environment because Streamlit Community Cloud requires manual browser login and app creation.

Docker Compose deployment files are included. Docker was not executed in the final release environment because the Docker CLI was unavailable.

## Limitations

- Synthetic demo data only.
- Default retrieval is TF-IDF.
- Default LLM provider is mock mode.
- Production deployment would require authentication, durable storage, monitoring, and real data governance.

## Next Steps

- Deploy Streamlit dashboard publicly.
- Add optional embeddings.
- Add labelled human relevance judgments.
- Add richer time-series product telemetry.
- Add production observability and authentication.
