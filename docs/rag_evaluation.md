# RAG Evaluation

All retrieval data is synthetic demo data. The current implementation uses a local TF-IDF retriever so the project works without an API key.

## Evaluation Method

The evaluation script runs a small synthetic question set against account-specific evidence and checks whether retrieved documents match the expected risk theme and document type. This reflects the production workflow, where the agent retrieves evidence within a selected customer account.

Evaluation dataset:

`data/evaluation/rag_eval_questions.csv`

Fields:

- question ID
- customer ID
- question
- expected risk theme
- expected document type
- expected answer points

Metrics:

- `precision_at_k`: share of retrieved documents matching the expected risk theme.
- `recall_at_k`: whether at least one expected-theme document was retrieved.
- `expected_theme_match`: whether retrieval found the expected risk theme.
- `expected_document_type_match`: whether retrieval found the expected document type.
- `latency_ms`: time to retrieve results for each query.
- `groundedness_heuristic`: overlap between retrieved evidence and expected answer points.
- `evidence_coverage_score`: share of expected answer points present in retrieved evidence.
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
