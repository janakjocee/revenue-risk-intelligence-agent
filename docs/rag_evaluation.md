# RAG Evaluation

All retrieval data is synthetic demo data. The current implementation uses a local TF-IDF retriever so the project works without an API key.

## Evaluation Method

The evaluation script runs theme-specific queries against customers known to have that risk theme and checks whether retrieved documents match the expected theme. This reflects the production workflow, where the agent retrieves evidence within a selected customer account.

Metrics:

- `precision_at_k`: share of retrieved documents matching the expected risk theme.
- `latency_ms`: time to retrieve results for each query.
- `top_document_ids`: sample retrieved documents for manual inspection.

Run:

```bash
python3 scripts/evaluate_rag.py
```

## Sample Results

Results are saved to `data/processed/rag_evaluation_summary.csv`.

The evaluation is intentionally simple and transparent. It is useful for a portfolio demo because it shows how retrieval can be checked, but it is not a substitute for human relevance review or production-grade RAG evaluation.

## Groundedness Approach

The agent response cites retrieved document IDs and includes a caveat when evidence is weak. Unsupported claims are avoided by relying on structured risk scores and retrieved synthetic evidence.

## Limitations

- Synthetic documents are templated, so retrieval is easier than in a real messy CRM environment.
- TF-IDF does not capture deep semantic similarity.
- No human relevance labels are included.
- Groundedness checks are heuristic.

## Future Improvements

- Add human-labeled relevance judgments.
- Add embedding-based retrieval when an API key is available.
- Track answer-level citation coverage.
- Add regression tests for retrieval quality over time.
