from datetime import datetime, timedelta

from api.search_service import rerank_incidents, score_priority


def build_incident(
    incident_id,
    *,
    similarity,
    priority,
    status="resolved",
    created_at=None,
):
    return {
        "id": incident_id,
        "ticket_id": f"TKT-{incident_id}",
        "status": status,
        "issue_type": "account_access",
        "product_area": "login_auth",
        "priority": priority,
        "initial_message": "Example incident",
        "resolution_summary": "Example resolution",
        "created_at": created_at or datetime.now() - timedelta(days=7),
        "resolution_time_hours": 2.0,
        "customer_sentiment": "neutral",
        "csat_score": 4,
        "similarity_score": similarity,
    }


def test_score_priority_maps_expected_weights():
    assert score_priority("critical") == 1.0
    assert score_priority("urgent") == 0.9
    assert score_priority("high") == 0.75
    assert score_priority("medium") == 0.55
    assert score_priority("low") == 0.35
    assert score_priority(None) == 0.2
    assert score_priority("unknown") == 0.2


def test_rerank_incidents_includes_priority_signal():
    incidents = [
        build_incident(1, similarity=0.82, priority="low"),
        build_incident(2, similarity=0.79, priority="critical"),
    ]

    ranked = rerank_incidents(incidents, top_n=2)

    assert ranked[0]["id"] == 2
    assert ranked[0]["scores"]["priority"] > ranked[1]["scores"]["priority"]
    assert ranked[0]["scores"]["final"] > ranked[1]["scores"]["final"]


def test_rerank_preserves_similarity_when_other_signals_match():
    incidents = [
        build_incident(1, similarity=0.9, priority="high"),
        build_incident(2, similarity=0.7, priority="high"),
    ]

    ranked = rerank_incidents(incidents, top_n=2)

    assert ranked[0]["id"] == 1
    assert ranked[0]["scores"]["similarity"] > ranked[1]["scores"]["similarity"]
