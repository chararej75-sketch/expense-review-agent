"""
app.py -- Streamlit web app for the Expense Review Agent.

Deploy this file to Streamlit Community Cloud. Users upload (or use the
built-in sample) expense-transaction data and get back the agent's risk
scores, flags, explanations, and a management summary -- no accounting
or ML background required to use it.
"""

import os

import pandas as pd
import streamlit as st

from agent import AgentConfig, DEFAULT_CATEGORY_LIMITS, ExpenseReviewAgent

# If an ANTHROPIC_API_KEY was provided via Streamlit secrets, expose it as an
# env var so agent.py's optional LLM narrative can pick it up. This key is
# entirely optional -- the app works fully without it.
try:
    if "ANTHROPIC_API_KEY" in st.secrets and st.secrets["ANTHROPIC_API_KEY"]:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    pass  # no secrets.toml configured -- that's fine, feature just stays off

st.set_page_config(
    page_title="Expense Review Agent",
    page_icon="🧾",
    layout="wide",
)

# --------------------------------------------------------------------- #
# Sidebar: data input + policy configuration
# --------------------------------------------------------------------- #
st.sidebar.title("🧾 Expense Review Agent")
st.sidebar.caption("AI-assisted triage for Accounts Payable / expense audit")

st.sidebar.header("1. Data")
data_source = st.sidebar.radio(
    "Choose a data source",
    ["Use sample data", "Upload CSV"],
    index=0,
)

uploaded_file = None
if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload expense transactions (.csv)",
        type=["csv"],
        help=(
            "Required columns: transaction_id, employee_id, employee_name, "
            "department, category, vendor, amount, date, payment_method, "
            "receipt_attached, approved_by"
        ),
    )

st.sidebar.header("2. Policy limits ($)")
st.sidebar.caption("Adjust category spending limits used for rule checks.")
category_limits = {}
for category, default_limit in DEFAULT_CATEGORY_LIMITS.items():
    category_limits[category] = st.sidebar.number_input(
        category, min_value=0, value=int(default_limit), step=25
    )

st.sidebar.header("3. Detection settings")
contamination = st.sidebar.slider(
    "Expected anomaly rate (ML model)", 0.02, 0.30, 0.10, 0.01,
    help="Roughly what % of transactions the anomaly model should expect to be unusual.",
)
high_thresh = st.sidebar.slider("High-risk score threshold", 40, 95, 65, 5)
med_thresh = st.sidebar.slider("Medium-risk score threshold", 10, high_thresh - 5, 35, 5)

use_ai_summary = st.sidebar.toggle(
    "Use Claude AI for narrative summary (optional)",
    value=False,
    help=(
        "Requires an ANTHROPIC_API_KEY set in Streamlit secrets. If not set, "
        "a template-based summary is shown instead -- the app works fully without it."
    ),
)

st.sidebar.divider()
st.sidebar.caption(
    "This agent provides decision SUPPORT only. It never approves or pays "
    "transactions -- a human must review flagged items."
)

# --------------------------------------------------------------------- #
# Load data
# --------------------------------------------------------------------- #
@st.cache_data
def load_sample_data():
    return pd.read_csv("data/sample_expenses.csv")


if data_source == "Upload CSV":
    if uploaded_file is None:
        st.title("🧾 Expense Review Agent")
        st.info(
            "Upload a CSV of expense transactions from the sidebar to begin, "
            "or switch to **Use sample data** to try the agent immediately."
        )
        with st.expander("Required CSV column format"):
            st.code(
                "transaction_id, employee_id, employee_name, department, category, "
                "vendor, amount, date, payment_method, receipt_attached, approved_by"
            )
        st.stop()
    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the uploaded file: {e}")
        st.stop()
else:
    raw_df = load_sample_data()

# --------------------------------------------------------------------- #
# Run the agent
# --------------------------------------------------------------------- #
st.title("🧾 Expense Review Agent")
st.caption(
    "AI decision-support agent for expense claim review: rule-based policy "
    "checks + Isolation Forest anomaly detection, blended into a risk score "
    "and plain-language explanation for every transaction."
)

config = AgentConfig(
    category_limits=category_limits,
    contamination=contamination,
    high_risk_threshold=high_thresh,
    medium_risk_threshold=med_thresh,
)
agent = ExpenseReviewAgent(config)

try:
    with st.spinner("Agent reviewing transactions..."):
        scored_df = agent.run(raw_df)
except ValueError as e:
    st.error(f"Data validation error: {e}")
    st.stop()

stats = agent.summary_stats(scored_df)

# --------------------------------------------------------------------- #
# Top-line metrics
# --------------------------------------------------------------------- #
c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions reviewed", stats["total_transactions"])
c2.metric("High risk", stats["high_risk"])
c3.metric("Medium risk", stats["medium_risk"])
c4.metric("Flagged $ (needs review)", f"${stats['flagged_amount']:,.0f}")

# --------------------------------------------------------------------- #
# Narrative summary
# --------------------------------------------------------------------- #
st.subheader("Agent summary")
summary_text = agent.narrative_summary(scored_df, use_llm=use_ai_summary)
st.markdown(summary_text.replace("\n", "\n\n"))

st.divider()

# --------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------- #
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("**Risk tier distribution**")
    tier_counts = scored_df["risk_tier"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
    st.bar_chart(tier_counts)

with chart_col2:
    st.markdown("**Flagged amount by department**")
    dept_flagged = (
        scored_df[scored_df["risk_tier"] != "Low"]
        .groupby("department")["amount"]
        .sum()
        .sort_values(ascending=False)
    )
    st.bar_chart(dept_flagged)

st.divider()

# --------------------------------------------------------------------- #
# Results table + filters
# --------------------------------------------------------------------- #
st.subheader("Transaction-level results")

filt_col1, filt_col2, filt_col3 = st.columns(3)
with filt_col1:
    tier_filter = st.multiselect(
        "Risk tier", ["High", "Medium", "Low"], default=["High", "Medium"]
    )
with filt_col2:
    dept_filter = st.multiselect(
        "Department", sorted(scored_df["department"].unique().tolist()), default=[]
    )
with filt_col3:
    search = st.text_input("Search employee / vendor / transaction ID")

filtered = scored_df.copy()
if tier_filter:
    filtered = filtered[filtered["risk_tier"].isin(tier_filter)]
if dept_filter:
    filtered = filtered[filtered["department"].isin(dept_filter)]
if search:
    s = search.lower()
    filtered = filtered[
        filtered["employee_name"].str.lower().str.contains(s)
        | filtered["vendor"].str.lower().str.contains(s)
        | filtered["transaction_id"].str.lower().str.contains(s)
    ]

display_cols = [
    "transaction_id", "date", "employee_name", "department", "category",
    "vendor", "amount", "risk_score", "risk_tier", "recommended_action",
]
filtered_sorted = filtered.sort_values("risk_score", ascending=False)
st.dataframe(
    filtered_sorted[display_cols],
    use_container_width=True,
    hide_index=True,
    column_config={
        "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
        "risk_score": st.column_config.ProgressColumn(
            "Risk score", min_value=0, max_value=100, format="%.0f"
        ),
    },
)

st.download_button(
    "Download flagged report (CSV)",
    data=filtered_sorted.drop(columns=["rule_flags"]).to_csv(index=False).encode("utf-8"),
    file_name="expense_review_flagged_report.csv",
    mime="text/csv",
)

st.divider()

# --------------------------------------------------------------------- #
# Transaction detail / explanation view
# --------------------------------------------------------------------- #
st.subheader("Transaction detail")
if len(filtered_sorted) == 0:
    st.info("No transactions match the current filters.")
else:
    selected_id = st.selectbox(
        "Select a transaction to see the agent's full reasoning",
        filtered_sorted["transaction_id"].tolist(),
    )
    row = scored_df[scored_df["transaction_id"] == selected_id].iloc[0]

    d1, d2 = st.columns([2, 1])
    with d1:
        st.markdown(f"**{row['employee_name']}** ({row['department']}) — {row['category']} at {row['vendor']}")
        st.markdown(f"Amount: **${row['amount']:,.2f}**  |  Date: {row['date'].strftime('%Y-%m-%d')}  |  Approved by: {row['approved_by']}")
        st.markdown(f"**Recommended action:** {row['recommended_action']}")
        st.markdown("**Agent explanation:**")
        if row["rule_flags"]:
            for f in row["rule_flags"]:
                st.markdown(f"- {f}")
        else:
            st.markdown("- No rule-based policy violations detected.")
        st.markdown(f"- ML anomaly score: {row['ml_anomaly_score']:.0f}/100")
    with d2:
        st.metric("Risk score", f"{row['risk_score']:.0f}/100")
        st.metric("Risk tier", row["risk_tier"])

st.divider()
st.caption(
    "Built for accounting decision support. Uses simulated/synthetic data only. "
    "No confidential or real client data is included in this deployment."
)
