from __future__ import annotations

from typing import Any


def recommend_actions(customer: dict[str, Any], score: dict[str, Any]) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    drivers = set(score.get("top_risk_drivers", []))

    if "Unresolved support tickets" in drivers:
        actions.append(
            {
                "action": "Assign technical support review",
                "why": "Open support issues are a direct renewal-risk signal.",
                "owner": "Support Lead",
            }
        )
    if "Low product usage" in drivers or "Recent login inactivity" in drivers:
        actions.append(
            {
                "action": "Investigate product usage decline",
                "why": "Low adoption reduces perceived value before renewal.",
                "owner": "Customer Success Manager",
            }
        )
    if "Onboarding incomplete" in drivers:
        actions.append(
            {
                "action": "Offer onboarding support",
                "why": "Incomplete onboarding suggests the customer may not have reached value.",
                "owner": "Onboarding Specialist",
            }
        )
    if "Payment delay" in drivers:
        actions.append(
            {
                "action": "Offer commercial renewal support",
                "why": "Payment delays may indicate budget friction or procurement risk.",
                "owner": "Account Executive",
            }
        )
    if score["risk_band"] == "high":
        actions.append(
            {
                "action": "Schedule executive check-in",
                "why": "High-risk accounts need senior alignment and fast blocker removal.",
                "owner": "CS Director",
            }
        )

    actions.append(
        {
            "action": "Send customer-success email",
            "why": "A concise follow-up creates a clear next step and confirms priorities.",
            "owner": "Customer Success Manager",
        }
    )
    return actions[:5]

