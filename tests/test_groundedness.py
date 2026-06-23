from src.evaluation.groundedness import evaluate_groundedness


def test_groundedness_detects_evidence_and_score():
    result = {
        "answer": "The account has support risk based on unresolved tickets.",
        "caveats": "Answer is grounded in retrieved synthetic account evidence.",
        "risk_score": {"top_risk_drivers": ["Unresolved support tickets"]},
        "cited_evidence": [
            {
                "document_id": "DOC-1",
                "text": "There are unresolved support tickets before renewal.",
                "score": 0.7,
            }
        ],
    }
    evaluation = evaluate_groundedness(result)
    assert evaluation["cites_retrieved_evidence"] is True
    assert evaluation["unsupported_mentions"] == []
    assert evaluation["groundedness_score"] >= 0.75


def test_groundedness_flags_unsupported_claims():
    result = {
        "answer": "The customer will definitely churn because of fraud.",
        "caveats": "",
        "risk_score": {"top_risk_drivers": ["Low product usage"]},
        "cited_evidence": [],
    }
    evaluation = evaluate_groundedness(result)
    assert "definitely churn" in evaluation["unsupported_mentions"]
    assert "fraud" in evaluation["unsupported_mentions"]
    assert evaluation["groundedness_score"] < 0.75

