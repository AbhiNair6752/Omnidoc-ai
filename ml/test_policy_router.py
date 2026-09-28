from app.graph.nodes import route_after_policy

approved_state = {
    "decision": {
        "decision": "APPROVED",
        "reason": "Invoice is valid"
    }
}

manual_review_state = {
    "decision": {
        "decision": "MANUAL_REVIEW",
        "reason": "invoice requires manual approval"
    }
}

rejected_state = {
    "decision": {
        "decision": "REJECTED",
        "reason": "Invoice contains invalid information"
    }
}

print(
    "APPROVED.route:",
    route_after_policy(
        approved_state
    )
)

print(
    "MANUAL_REVIEW route:",
    route_after_policy(
        manual_review_state
    )
)


print(
    "REJECTED route:",
    route_after_policy(
        rejected_state
    )
)