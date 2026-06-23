from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.agent.workflow import run_agent
from src.observability.logger import read_agent_logs
from src.rag.retriever import load_retriever


CUSTOMERS_PATH = Path("data/processed/customer_features.csv")
SCORES_PATH = Path("data/processed/customer_risk_scores.csv")
RETRIEVER_PATH = Path("data/processed/tfidf_retriever.joblib")
MODEL_PATH = "data/processed/churn_model.joblib"


st.set_page_config(page_title="Revenue Risk Intelligence", page_icon="📊", layout="wide")


@st.cache_data
def load_customer_table() -> pd.DataFrame:
    customers = pd.read_csv(CUSTOMERS_PATH)
    scores = pd.read_csv(SCORES_PATH)
    return customers.merge(scores, on="customer_id", how="left")


@st.cache_resource
def cached_retriever():
    return load_retriever(RETRIEVER_PATH)


customers = load_customer_table()
retriever = cached_retriever()

st.title("Revenue Risk Intelligence Agent")
st.caption("Synthetic demo data. Built for customer-success decision support, not as a basic chatbot.")

tab_overview, tab_customer, tab_observability = st.tabs(["Overview", "Customer Workspace", "Evaluation & Observability"])

with tab_overview:
    high_risk = int((customers["risk_band"] == "high").sum())
    total_revenue_at_risk = customers["revenue_at_risk"].sum()
    avg_probability = customers["churn_probability"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers", f"{len(customers):,}")
    col2.metric("High-risk accounts", f"{high_risk:,}")
    col3.metric("Revenue at risk", f"GBP {total_revenue_at_risk:,.0f}")
    col4.metric("Avg churn probability", f"{avg_probability:.1%}")

    st.subheader("Customer Risk Table")
    risk_filter = st.multiselect("Risk band", ["high", "medium", "low"], default=["high", "medium", "low"])
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

with tab_customer:
    selected_customer_id = st.selectbox(
        "Customer",
        customers.sort_values("revenue_at_risk", ascending=False)["customer_id"],
        format_func=lambda cid: f"{cid} - {customers.loc[customers['customer_id'].eq(cid), 'company_name'].iloc[0]}",
    )
    row = customers.loc[customers["customer_id"].eq(selected_customer_id)].iloc[0].to_dict()

    col1, col2, col3 = st.columns([1.2, 1, 1])
    col1.metric("Risk band", row["risk_band"].upper())
    col2.metric("Churn probability", f"{row['churn_probability']:.1%}")
    col3.metric("Revenue at risk", f"GBP {row['revenue_at_risk']:,.0f}")

    st.subheader(row["company_name"])
    profile_cols = [
        "industry",
        "contract_type",
        "monthly_recurring_revenue",
        "tenure_months",
        "product_usage_score",
        "unresolved_ticket_count",
        "payment_delay_days",
        "nps_score",
        "account_health_score",
    ]
    st.dataframe(pd.DataFrame([row])[profile_cols], use_container_width=True, hide_index=True)

    question = st.text_area(
        "Ask about this account",
        value="Why is this customer at risk and what should the customer-success team do next?",
        height=90,
    )
    include_email = st.checkbox("Draft a customer-success email", value=True)

    if st.button("Run agent", type="primary"):
        result = run_agent(row, question, retriever, model_path=MODEL_PATH, include_email=include_email)
        st.session_state["agent_result"] = result

    result = st.session_state.get("agent_result")
    if result:
        st.subheader("Agent Answer")
        st.write(result["answer"])
        st.info(result["caveats"])

        action_col, email_col = st.columns(2)
        with action_col:
            st.subheader("Recommended Actions")
            for action in result["recommended_actions"]:
                st.markdown(f"**{action['action']}**  \n{action['why']}  \nOwner: {action['owner']}")
        with email_col:
            st.subheader("Email Draft")
            st.text_area("Draft", value=result["email_draft"] or "Email draft was not requested.", height=240)

        st.subheader("Retrieved Evidence")
        evidence = pd.DataFrame(result["cited_evidence"])
        st.dataframe(
            evidence[["document_id", "document_type", "risk_theme", "score", "text"]],
            use_container_width=True,
            hide_index=True,
        )

with tab_observability:
    logs = pd.DataFrame(read_agent_logs())
    st.subheader("Evaluation Summary")
    st.write("Retrieval backend: local TF-IDF. Answers cite synthetic documents and include caveats when evidence is weak.")
    if logs.empty:
        st.warning("No agent runs logged yet. Run the agent from the Customer Workspace tab.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Agent runs", f"{len(logs):,}")
        col2.metric("Average latency", f"{logs['latency_ms'].mean():.0f} ms")
        col3.metric("Average response length", f"{logs['response_length'].mean():.0f} chars")
        st.bar_chart(logs["risk_band"].value_counts())
        st.subheader("Recent Questions")
        st.dataframe(logs.tail(20), use_container_width=True, hide_index=True)

