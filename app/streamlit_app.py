from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.agent.briefing import render_account_brief
from src.agent.workflow import run_agent
from src.model.simulator import simulate_customer_scenario
from src.observability.logger import append_feedback, feedback_summary, read_agent_logs
from src.rag.retriever import load_retriever


CUSTOMERS_PATH = Path("data/processed/customer_features.csv")
SCORES_PATH = Path("data/processed/customer_risk_scores.csv")
RETRIEVER_PATH = Path("data/processed/tfidf_retriever.joblib")
RAG_EVAL_PATH = Path("data/processed/rag_evaluation_summary.csv")
MODEL_PATH = "data/processed/churn_model.joblib"


st.set_page_config(page_title="Revenue Risk Intelligence", layout="wide")


st.markdown(
    """
    <style>
    .block-container {padding-top: 1.25rem;}
    [data-testid="stMetricValue"] {font-size: 1.65rem;}
    .section-note {color: #5f6b7a; font-size: 0.92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_customer_table() -> pd.DataFrame:
    customers = pd.read_csv(CUSTOMERS_PATH)
    scores = pd.read_csv(SCORES_PATH)
    return customers.merge(scores, on="customer_id", how="left")


@st.cache_data
def load_rag_evaluation() -> pd.DataFrame:
    return pd.read_csv(RAG_EVAL_PATH) if RAG_EVAL_PATH.exists() else pd.DataFrame()


@st.cache_resource
def cached_retriever():
    return load_retriever(RETRIEVER_PATH)


def selected_customer(customers: pd.DataFrame) -> dict:
    selected_customer_id = st.sidebar.selectbox(
        "Account",
        customers.sort_values("revenue_at_risk", ascending=False)["customer_id"],
        format_func=lambda cid: f"{cid} - {customers.loc[customers['customer_id'].eq(cid), 'company_name'].iloc[0]}",
    )
    return customers.loc[customers["customer_id"].eq(selected_customer_id)].iloc[0].to_dict()


def show_kpis(customers: pd.DataFrame) -> None:
    high_risk = int((customers["risk_band"] == "high").sum())
    medium_risk = int((customers["risk_band"] == "medium").sum())
    total_revenue_at_risk = customers["revenue_at_risk"].sum()
    avg_probability = customers["churn_probability"].mean()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accounts", f"{len(customers):,}")
    col2.metric("High Risk", f"{high_risk:,}")
    col3.metric("Medium Risk", f"{medium_risk:,}")
    col4.metric("Revenue at Risk", f"GBP {total_revenue_at_risk:,.0f}", f"{avg_probability:.1%} avg churn")


customers = load_customer_table()
retriever = cached_retriever()

st.sidebar.title("Revenue Risk")
page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Customer Workspace",
        "What-if Simulator",
        "Account Brief Export",
        "Evaluation",
        "Observability",
    ],
)
customer = selected_customer(customers)

st.title("Revenue Risk Intelligence Agent")
st.caption("Synthetic customer-success decision system with ML risk scoring, RAG evidence, agent recommendations, and observability.")

if page == "Executive Overview":
    show_kpis(customers)
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Risk Distribution")
        st.bar_chart(customers["risk_band"].value_counts())
    with right:
        st.subheader("Revenue at Risk by Industry")
        st.bar_chart(customers.groupby("industry")["revenue_at_risk"].sum().sort_values(ascending=False))

    st.subheader("Prioritised Account Queue")
    risk_filter = st.multiselect("Risk band", ["high", "medium", "low"], default=["high", "medium"])
    filtered = customers[customers["risk_band"].isin(risk_filter)]
    st.dataframe(
        filtered[
            [
                "customer_id",
                "company_name",
                "industry",
                "risk_band",
                "churn_probability",
                "revenue_at_risk",
                "top_risk_drivers",
                "recommended_action_category",
            ]
        ].sort_values("revenue_at_risk", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

if page == "Customer Workspace":
    st.subheader(customer["company_name"])
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Risk Band", customer["risk_band"].upper())
    col2.metric("Churn Probability", f"{customer['churn_probability']:.1%}")
    col3.metric("Revenue at Risk", f"GBP {customer['revenue_at_risk']:,.0f}")
    col4.metric("Health Score", f"{customer['account_health_score']:.1f}")

    profile_cols = [
        "industry",
        "contract_type",
        "monthly_recurring_revenue",
        "tenure_months",
        "product_usage_score",
        "unresolved_ticket_count",
        "payment_delay_days",
        "nps_score",
    ]
    st.dataframe(pd.DataFrame([customer])[profile_cols], use_container_width=True, hide_index=True)

    question = st.text_area(
        "Account question",
        value="Why is this customer at risk and what should the customer-success team do next?",
        height=90,
    )
    include_email = st.checkbox("Draft customer-success email", value=True)

    if st.button("Run Revenue Risk Agent", type="primary"):
        st.session_state["agent_result"] = run_agent(
            customer,
            question,
            retriever,
            model_path=MODEL_PATH,
            include_email=include_email,
        )

    result = st.session_state.get("agent_result")
    if result:
        st.subheader("Agent Answer")
        st.write(result["answer"])
        st.caption(result["caveats"])
        st.progress(result["groundedness_evaluation"]["groundedness_score"], text="Groundedness heuristic")

        action_col, evidence_col = st.columns([0.9, 1.1])
        with action_col:
            st.subheader("Recommended Next Action")
            for action in result["recommended_actions"]:
                st.markdown(f"**{action['action']}**  \n{action['why']}  \nOwner: {action['owner']}")
            st.subheader("Email Draft")
            st.text_area("Draft", value=result["email_draft"] or "Email draft was not requested.", height=220)
        with evidence_col:
            st.subheader("Evidence")
            evidence = pd.DataFrame(result["cited_evidence"])
            st.dataframe(
                evidence[["document_id", "document_type", "risk_theme", "score", "explanation", "text"]],
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Human Feedback")
        rating = st.slider("Rating", min_value=1, max_value=5, value=4)
        feedback_reason = st.text_area("Feedback reason", height=80)
        if st.button("Submit Feedback"):
            append_feedback(
                {
                    "run_id": result["run_id"],
                    "customer_id": result["customer_id"],
                    "rating": rating,
                    "feedback_reason": feedback_reason,
                    "question": result["question"],
                    "risk_band": result["risk_score"]["risk_band"],
                }
            )
            st.success("Feedback saved.")

if page == "What-if Simulator":
    st.subheader(f"What-if Simulator: {customer['company_name']}")
    st.markdown('<p class="section-note">Adjust business levers and compare the scenario score to the current baseline.</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    adjustments = {
        "unresolved_ticket_count": col1.number_input("Unresolved Tickets", 0, 50, int(customer["unresolved_ticket_count"])),
        "product_usage_score": col1.slider("Product Usage Score", 0.0, 100.0, float(customer["product_usage_score"])),
        "payment_delay_days": col2.slider("Payment Delay Days", 0.0, 120.0, float(customer["payment_delay_days"])),
        "nps_score": col2.slider("NPS", -100.0, 100.0, float(customer["nps_score"])),
        "account_health_score": col3.slider("Account Health Score", 0.0, 100.0, float(customer["account_health_score"])),
        "contract_type": col3.selectbox("Contract Type", ["monthly", "annual", "multi-year"], index=["monthly", "annual", "multi-year"].index(customer["contract_type"])),
    }
    simulation = simulate_customer_scenario(customer, adjustments, model_path=MODEL_PATH)
    base, scenario = simulation["baseline"], simulation["scenario"]
    k1, k2, k3 = st.columns(3)
    k1.metric("Scenario Churn", f"{scenario['churn_probability']:.1%}", f"{scenario['churn_probability'] - base['churn_probability']:+.1%}")
    k2.metric("Scenario Risk Band", scenario["risk_band"].upper())
    k3.metric("Scenario Revenue at Risk", f"GBP {scenario['revenue_at_risk']:,.0f}", f"GBP {scenario['revenue_at_risk'] - base['revenue_at_risk']:,.0f}")
    st.subheader("Scenario Recommended Actions")
    for action in simulation["recommended_actions"]:
        st.markdown(f"**{action['action']}**  \n{action['why']}  \nOwner: {action['owner']}")

if page == "Account Brief Export":
    st.subheader(f"Account Brief Export: {customer['company_name']}")
    if st.button("Generate Account Brief", type="primary"):
        result = run_agent(
            customer,
            "Prepare an account brief with renewal risk, evidence, recommendations, and email draft.",
            retriever,
            model_path=MODEL_PATH,
            include_email=True,
        )
        st.session_state["brief_markdown"] = render_account_brief(customer, result)

    brief_markdown = st.session_state.get("brief_markdown")
    if brief_markdown:
        st.download_button(
            "Download Markdown Brief",
            data=brief_markdown,
            file_name=f"{customer['customer_id']}_account_brief.md",
            mime="text/markdown",
        )
        st.markdown(brief_markdown)

if page == "Evaluation":
    st.subheader("Evaluation")
    rag_eval = load_rag_evaluation()
    if rag_eval.empty:
        st.warning("No RAG evaluation summary found. Run `python3 scripts/evaluate_rag.py`.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Precision@K", f"{rag_eval['precision_at_k'].mean():.2f}")
        col2.metric("Theme Match Rate", f"{rag_eval['expected_theme_match'].mean():.0%}")
        col3.metric("Avg Evidence Coverage", f"{rag_eval['evidence_coverage_score'].mean():.2f}")
        st.dataframe(rag_eval, use_container_width=True, hide_index=True)

if page == "Observability":
    st.subheader("Observability")
    logs = pd.DataFrame(read_agent_logs())
    summary = feedback_summary()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Agent Runs", f"{len(logs):,}")
    col2.metric("Avg Latency", f"{logs['latency_ms'].mean():.0f} ms" if not logs.empty else "N/A")
    col3.metric("Feedback Count", f"{summary['feedback_count']:,}")
    col4.metric("Avg Rating", summary["average_rating"] if summary["average_rating"] is not None else "N/A")
    if not logs.empty:
        st.subheader("Risk Bands in Agent Runs")
        st.bar_chart(logs["risk_band"].value_counts())
        st.subheader("Recent Questions")
        st.dataframe(logs.tail(30), use_container_width=True, hide_index=True)
