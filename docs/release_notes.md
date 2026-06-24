# Release Notes

## v1.0 Portfolio Release

Revenue Risk Intelligence Agent is ready as a public portfolio project and recruiter-facing case study.

Author: Janak Raj Joshi, Applied Data Scientist

- Portfolio: https://janakjocee.vercel.app/
- GitHub: https://github.com/janakjocee
- LinkedIn: https://www.linkedin.com/in/janakjocee

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

The Streamlit dashboard is deployed at https://revenue-risk.streamlit.app/

Deployment fix applied: `packages.txt` is empty so Streamlit Cloud does not pass comment text to `apt-get`, and the app setting is configured for Python 3.11.

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
