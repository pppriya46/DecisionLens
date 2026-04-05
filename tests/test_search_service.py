from datetime import datetime, timedelta

from pydantic import ValidationError

from api.models import SubmitDuplicateReviewRequest
from api.search_service import (
    classify_duplicate_match,
    collapse_repetitive_results,
    dedupe_duplicate_candidates,
    filter_duplicate_candidates,
    normalize_support_text,
    rerank_duplicate_candidates,
    rerank_incidents,
    score_keyword_overlap,
)


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


def build_duplicate_candidate(
    incident_id,
    *,
    similarity,
    issue_type="account_access",
    product_area="login_auth",
):
    return {
        "id": incident_id,
        "ticket_id": f"TKT-{incident_id}",
        "status": "open",
        "issue_type": issue_type,
        "product_area": product_area,
        "priority": "medium",
        "initial_message": "Users cannot log in after the SSO rollout.",
        "created_at": datetime.now() - timedelta(days=2),
        "similarity_score": similarity,
    }


def test_rerank_incidents_prefers_resolved_recent_cases_when_similarity_is_close():
    incidents = [
        build_incident(
            1,
            similarity=0.82,
            priority="low",
            status="resolved",
            created_at=datetime.now() - timedelta(days=5),
        ),
        build_incident(
            2,
            similarity=0.81,
            priority="critical",
            status="open",
            created_at=datetime.now() - timedelta(days=300),
        ),
    ]
    incidents[0]["initial_message"] = "Login fails after password reset."
    incidents[1]["initial_message"] = "Dashboard export times out for all users."

    ranked = rerank_incidents(incidents, query_text="Login fails after password reset.", top_n=2)

    assert ranked[0]["id"] == 1
    assert ranked[0]["scores"]["status"] > ranked[1]["scores"]["status"]
    assert ranked[0]["scores"]["final"] > ranked[1]["scores"]["final"]


def test_rerank_preserves_similarity_when_other_signals_match():
    incidents = [
        build_incident(1, similarity=0.9, priority="high"),
        build_incident(2, similarity=0.7, priority="high"),
    ]
    incidents[0]["initial_message"] = "Login fails after password reset."
    incidents[1]["initial_message"] = "Dashboard export times out for all users."

    ranked = rerank_incidents(incidents, query_text="Login fails after password reset.", top_n=2)

    assert ranked[0]["id"] == 1
    assert ranked[0]["scores"]["similarity"] > ranked[1]["scores"]["similarity"]


def test_rerank_incidents_prefers_keyword_overlap_for_close_matches():
    incidents = [
        build_incident(1, similarity=0.79, priority="medium", status="resolved"),
        build_incident(2, similarity=0.78, priority="medium", status="resolved"),
    ]
    incidents[0]["initial_message"] = "Something seems off, can you please check my workspace?"
    incidents[1]["initial_message"] = "Login fails after password reset and 2FA challenge."

    ranked = rerank_incidents(
        incidents,
        query_text="Password reset fails during login with 2FA",
        query_category="account_access",
        top_n=2,
    )

    assert ranked[0]["id"] == 2
    assert ranked[0]["scores"]["keyword_overlap"] > ranked[1]["scores"]["keyword_overlap"]


def test_rerank_incidents_prefers_matching_issue_category_when_similarity_is_close():
    incidents = [
        build_incident(1, similarity=0.8, priority="medium", status="resolved"),
        build_incident(2, similarity=0.79, priority="medium", status="resolved"),
    ]
    incidents[0]["issue_type"] = "performance"
    incidents[0]["initial_message"] = "Login fails after password reset."
    incidents[1]["issue_type"] = "account_access"
    incidents[1]["initial_message"] = "Login fails after password reset."

    ranked = rerank_incidents(
        incidents,
        query_text="Login fails after password reset.",
        query_category="account_access",
        top_n=2,
    )

    assert ranked[0]["id"] == 2
    assert ranked[0]["scores"]["category_match"] > ranked[1]["scores"]["category_match"]


def test_normalize_support_text_merges_common_variants():
    assert normalize_support_text("I can't log in to reset my password") == "i cannot login to reset my password"
    assert normalize_support_text("I am not able to sign in") == "i am cannot login"


def test_collapse_repetitive_results_keeps_representative_match():
    results = [
        {
            "id": 1,
            "ticket_id": "TKT-1",
            "status": "resolved",
            "issue_type": "account_access",
            "product_area": "login_auth",
            "priority": "medium",
            "description": "I cannot log in; the system says my password is incorrect.",
            "resolution": "Reset password",
            "created_at": None,
            "sentiment": "neutral",
            "csat_score": 4,
            "related_ticket_count": 1,
            "scores": {"final": 0.91, "similarity": 0.9, "status": 1.0, "recency": 0.8},
        },
        {
            "id": 2,
            "ticket_id": "TKT-2",
            "status": "resolved",
            "issue_type": "account_access",
            "product_area": "login_auth",
            "priority": "medium",
            "description": "I can't login, the system says my password is incorrect.",
            "resolution": "Reset password",
            "created_at": None,
            "sentiment": "neutral",
            "csat_score": 4,
            "related_ticket_count": 1,
            "scores": {"final": 0.87, "similarity": 0.86, "status": 1.0, "recency": 0.8},
        },
    ]

    collapsed = collapse_repetitive_results(results, top_n=5)

    assert len(collapsed) == 1
    assert collapsed[0]["id"] == 1
    assert collapsed[0]["related_ticket_count"] == 2
    assert collapsed[0]["grouped_ticket_ids"] == ["TKT-1", "TKT-2"]


def test_dedupe_duplicate_candidates_removes_exact_template_clones():
    candidates = [
        build_duplicate_candidate(1, similarity=0.95, issue_type="account_access", product_area="login_auth"),
        build_duplicate_candidate(2, similarity=0.94, issue_type="account_access", product_area="login_auth"),
        build_duplicate_candidate(3, similarity=0.9, issue_type="account_access", product_area="billing"),
    ]
    candidates[1]["ticket_id"] = "TKT-2"
    candidates[2]["initial_message"] = "My 2FA code is not working when I try to sign in."

    deduped = dedupe_duplicate_candidates(candidates, limit=20)

    assert [item["ticket_id"] for item in deduped] == ["TKT-1", "TKT-3"]


def test_rerank_duplicate_candidates_prefers_matching_metadata_when_scores_are_close():
    incidents = [
        build_duplicate_candidate(1, similarity=0.84, issue_type="billing", product_area="payments"),
        build_duplicate_candidate(2, similarity=0.83, issue_type="login", product_area="identity"),
    ]

    ranked = rerank_duplicate_candidates(
        incidents,
        query_text="Billing issue caused duplicate charges on the same payment.",
        query_issue_type="billing",
        query_product_area="payments",
        top_n=2,
    )

    assert ranked[0]["id"] == 1
    assert ranked[0]["duplicate_score"] > ranked[1]["duplicate_score"]
    assert "Same issue category" in ranked[0]["match_reasons"]
    assert "Same affected area" in ranked[0]["match_reasons"]


def test_rerank_duplicate_candidates_penalizes_conflicting_metadata_when_user_supplies_it():
    incidents = [
        build_duplicate_candidate(1, similarity=0.9, issue_type="billing", product_area="payments"),
        build_duplicate_candidate(2, similarity=0.84, issue_type="account_access", product_area="login_auth"),
    ]
    incidents[0]["initial_message"] = "Users cannot log in after the SSO rollout."
    incidents[1]["initial_message"] = "Users cannot log in after the SSO rollout."

    ranked = rerank_duplicate_candidates(
        incidents,
        query_text="Users cannot log in after the SSO rollout.",
        query_issue_type="account_access",
        query_product_area="login_auth",
        top_n=2,
    )

    assert ranked[0]["id"] == 2
    assert "Different affected area" in ranked[1]["match_reasons"]


def test_rerank_duplicate_candidates_keeps_metadata_optional():
    incidents = [
        build_duplicate_candidate(1, similarity=0.82, issue_type="billing", product_area="payments"),
        build_duplicate_candidate(2, similarity=0.8, issue_type="account_access", product_area="login_auth"),
    ]
    incidents[0]["initial_message"] = "Users cannot log in after the SSO rollout."
    incidents[1]["initial_message"] = "Generic support request with little overlap."

    ranked = rerank_duplicate_candidates(
        incidents,
        query_text="Users cannot log in after the SSO rollout.",
        top_n=2,
    )

    assert ranked[0]["id"] == 1
    assert "Different affected area" not in ranked[0]["match_reasons"]


def test_rerank_duplicate_candidates_includes_wording_reason():
    incidents = [
        build_duplicate_candidate(1, similarity=0.87),
    ]

    ranked = rerank_duplicate_candidates(
        incidents,
        query_text="Users cannot log in after the SSO rollout.",
        top_n=1,
    )

    assert "Very similar wording" in ranked[0]["match_reasons"]


def test_score_keyword_overlap_favors_shared_terms():
    overlap = score_keyword_overlap(
        "windows update keeps failing after restart",
        "Windows update failure after restart with error code",
    )

    assert overlap > 0.3


def test_rerank_duplicate_candidates_prefers_keyword_overlap_for_close_scores():
    incidents = [
        build_duplicate_candidate(
            1,
            similarity=0.78,
            issue_type="other",
            product_area="notifications",
        ),
        build_duplicate_candidate(
            2,
            similarity=0.77,
            issue_type="system update",
            product_area="windows",
        ),
    ]
    incidents[0]["initial_message"] = "Something seems off, can you please check my workspace?"
    incidents[1]["initial_message"] = "Windows update keeps failing after restart with an install error."

    ranked = rerank_duplicate_candidates(
        incidents,
        query_text="I am not able to update my windows after restart",
        top_n=2,
    )

    assert ranked[0]["id"] == 2
    assert "Shared key terms" in ranked[0]["match_reasons"] or "Some shared key terms" in ranked[0]["match_reasons"]


def test_classify_duplicate_match_thresholds():
    assert classify_duplicate_match(0.94)[0] == "likely_duplicate"
    assert classify_duplicate_match(0.9)[0] == "possibly_related"
    assert classify_duplicate_match(0.7)[0] == "new_issue"
    assert classify_duplicate_match(None)[0] == "new_issue"


def test_filter_duplicate_candidates_hides_weak_new_issue_matches():
    ranked_results = [
        {
            "id": 1,
            "ticket_id": "TKT-1",
            "status": "resolved",
            "issue_type": "other",
            "product_area": "notifications",
            "priority": "medium",
            "description": "Something seems off, can you please check my workspace?",
            "created_at": None,
            "similarity_score": 0.31,
            "keyword_overlap": 0.0,
            "duplicate_score": 0.28,
            "match_reasons": ["Closest available historical match"],
        }
    ]

    filtered = filter_duplicate_candidates(ranked_results, "new_issue")

    assert filtered == []


def test_submit_duplicate_review_request_accepts_known_decisions():
    request = SubmitDuplicateReviewRequest(
        reported_ticket_id="TKT-2026-1001",
        query_text="Users cannot log in after the latest SSO rollout and password reset fails.",
        matched_incident_id=42,
        decision="duplicate",
    )

    assert request.decision.value == "duplicate"


def test_submit_duplicate_review_request_rejects_unknown_decisions():
    try:
        SubmitDuplicateReviewRequest(
            reported_ticket_id="TKT-2026-1001",
            query_text="Users cannot log in after the latest SSO rollout and password reset fails.",
            matched_incident_id=42,
            decision="merge_now",
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("Expected invalid duplicate review decision to fail validation")
