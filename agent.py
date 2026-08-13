"""
agent.py

Core logic for the Expense Review Agent: an AI-assisted decision-support
agent for the Accounts Payable / Financial Controls function.

BUSINESS PROBLEM
-----------------
Manually reviewing every employee expense claim for policy violations,
duplicate/split submissions, missing documentation, and unusual spending
is slow and error-prone, especially as transaction volume grows. Finance
teams need a first-pass triage: which claims are safe to auto-approve,
and which need a human's attention before reimbursement or ledger posting.

WHAT THE AGENT DOES
--------------------
Given a batch of expense transactions, the agent:
  1. Applies deterministic accounting/policy rules (explainable, auditable)
     - category spending limit exceeded
     - missing receipt above disclosure threshold
     - suspiciously round amounts (possible estimation/fabrication)
     - self-approval (segregation-of-duties violation)
     - weekend/holiday submission
     - duplicate or "structured" (split just under the limit) submissions
  2. Runs an unsupervised ML anomaly model (Isolation Forest) over encoded
     transaction features to catch anomalies that don't match a fixed rule
     (e.g. an employee's amount is statistically unusual for their own
     category/department history).
  3. Combines rule flags + ML anomaly score into a single 0-100 risk score
     and a risk tier (Low / Medium / High).
  4. Produces a plain-language, auditable explanation and a recommended
     action (Auto-Approve / Manual Review / Escalate) for every transaction.

This is decision SUPPORT, not autonomous payment execution: the agent
never approves or pays anything itself, it only recommends and explains,
which keeps a human accountant in the loop as required by internal control
best practice (segregation of duties).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OrdinalEncoder

DEFAULT_CATEGORY_LIMITS = {
    "Meals & Entertainment": 150,
    "Travel - Airfare": 1200,
    "Travel - Lodging": 400,
    "Ground Transport": 120,
    "Office Supplies": 250,
    "Software & Subscriptions": 500,
    "Client Gifts": 100,
    "Training & Conferences": 900,
}

RECEIPT_REQUIRED_ABOVE = 75.0
ROUND_NUMBER_THRESHOLD = 100.0  # amounts >= this that are exact multiples of 50/100
STRUCTURING_WINDOW_DAYS = 3

REQUIRED_COLUMNS = [
    "transaction_id", "employee_id", "employee_name", "department",
    "category", "vendor", "amount", "date", "payment_method",
    "receipt_attached", "approved_by",
]


@dataclass
class AgentConfig:
    category_limits: dict = field(default_factory=lambda: dict(DEFAULT_CATEGORY_LIMITS))
    receipt_required_above: float = RECEIPT_REQUIRED_ABOVE
    contamination: float = 0.10  # expected proportion of anomalies for Isolation Forest
    high_risk_threshold: int = 65
    medium_risk_threshold: int = 35


class ExpenseReviewAgent:
    """AI agent that scores, flags, and explains expense transactions."""

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full agent pipeline and return an enriched DataFrame."""
        df = self._validate_and_prepare(df)
        df = self._apply_rules(df)
        df = self._apply_ml_anomaly_score(df)
        df = self._score_and_recommend(df)
        return df

    def summary_stats(self, scored_df: pd.DataFrame) -> dict:
        total = len(scored_df)
        tier_counts = scored_df["risk_tier"].value_counts().to_dict()
        total_flagged_amount = scored_df.loc[scored_df["risk_tier"] != "Low", "amount"].sum()
        return {
            "total_transactions": total,
            "high_risk": int(tier_counts.get("High", 0)),
            "medium_risk": int(tier_counts.get("Medium", 0)),
            "low_risk": int(tier_counts.get("Low", 0)),
            "total_amount_reviewed": float(scored_df["amount"].sum()),
            "flagged_amount": float(total_flagged_amount),
            "avg_risk_score": float(scored_df["risk_score"].mean()),
        }

    def narrative_summary(self, scored_df: pd.DataFrame, use_llm: bool = False) -> str:
        """
        Produce a short management-facing narrative summary.
        If use_llm is True and an ANTHROPIC_API_KEY is available in the
        environment, a Claude-generated narrative is used. Otherwise a
        deterministic template-based summary is returned (no API key
        required -- this is the default and always works).
        """
        stats = self.summary_stats(scored_df)
        top_flags = (
            scored_df[scored_df["risk_tier"] == "High"]
            .sort_values("risk_score", ascending=False)
            .head(5)
        )

        if use_llm:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                try:
                    return self._llm_narrative_summary(stats, top_flags, api_key)
                except Exception as exc:  # graceful fallback, never crash the app
                    fallback = self._template_narrative_summary(stats, top_flags)
                    return fallback + f"\n\n_(AI narrative unavailable: {exc}. Showing template summary instead.)_"

        return self._template_narrative_summary(stats, top_flags)

    # ------------------------------------------------------------------ #
    # Pipeline steps
    # ------------------------------------------------------------------ #
    def _validate_and_prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Input data is missing required column(s): {missing}. "
                f"Expected columns: {REQUIRED_COLUMNS}"
            )

        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        if df["amount"].isna().any():
            bad = df[df["amount"].isna()]["transaction_id"].tolist()
            raise ValueError(f"Non-numeric amount values found in rows: {bad}")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isna().any():
            bad = df[df["date"].isna()]["transaction_id"].tolist()
            raise ValueError(f"Unparseable date values found in rows: {bad}")

        df["receipt_attached"] = df["receipt_attached"].astype(bool)
        df["day_of_week"] = df["date"].dt.dayofweek  # 0=Mon ... 6=Sun
        df["is_weekend"] = df["day_of_week"] >= 5
        return df.sort_values("date").reset_index(drop=True)

    def _apply_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        flags_col = [[] for _ in range(len(df))]

        for idx, row in df.iterrows():
            flags = []

            limit = self.config.category_limits.get(row["category"])
            if limit is not None and row["amount"] > limit:
                pct_over = (row["amount"] / limit - 1) * 100
                flags.append(
                    f"Exceeds {row['category']} policy limit of ${limit:,.0f} "
                    f"by {pct_over:.0f}% (claimed ${row['amount']:,.2f})"
                )

            if (not row["receipt_attached"]) and row["amount"] >= self.config.receipt_required_above:
                flags.append(
                    f"No receipt attached for a claim of ${row['amount']:,.2f} "
                    f"(receipts required above ${self.config.receipt_required_above:,.0f})"
                )

            if row["amount"] >= ROUND_NUMBER_THRESHOLD and row["amount"] % 50 == 0:
                flags.append(f"Suspiciously round amount (${row['amount']:,.2f}) -- possible estimate rather than actual receipt")

            if str(row["approved_by"]).strip().lower() == str(row["employee_name"]).strip().lower():
                flags.append("Self-approved: approver is the same person as the claimant (segregation-of-duties violation)")

            if row["is_weekend"]:
                flags.append(f"Submitted/dated on a weekend ({row['date'].strftime('%A, %Y-%m-%d')})")

            flags_col[idx] = flags

        df["rule_flags"] = flags_col
        df["rule_flag_count"] = df["rule_flags"].apply(len)

        df = self._flag_duplicates_and_structuring(df)
        return df

    def _flag_duplicates_and_structuring(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag same employee+vendor+category submissions close together in
        time whose combined amount would have exceeded the category limit
        (a classic 'structuring' / split-transaction red flag), and exact
        duplicate submissions."""
        extra_flags = [[] for _ in range(len(df))]

        grouped = df.groupby(["employee_id", "vendor", "category"])
        for _, group in grouped:
            if len(group) < 2:
                continue
            group_sorted = group.sort_values("date")
            rows = list(group_sorted.iterrows())
            for i in range(len(rows)):
                idx_i, row_i = rows[i]
                for j in range(i + 1, len(rows)):
                    idx_j, row_j = rows[j]
                    day_gap = abs((row_j["date"] - row_i["date"]).days)
                    if day_gap > STRUCTURING_WINDOW_DAYS:
                        continue

                    # exact duplicate
                    if abs(row_i["amount"] - row_j["amount"]) < 0.01:
                        extra_flags[idx_i].append(
                            f"Possible duplicate of {row_j['transaction_id']} (same amount, {day_gap}d apart)"
                        )
                        extra_flags[idx_j].append(
                            f"Possible duplicate of {row_i['transaction_id']} (same amount, {day_gap}d apart)"
                        )
                        continue

                    # structuring: combined amount breaches the category limit,
                    # but individually each is under it
                    limit = self.config.category_limits.get(row_i["category"])
                    if limit is not None:
                        combined = row_i["amount"] + row_j["amount"]
                        if row_i["amount"] < limit and row_j["amount"] < limit and combined > limit:
                            extra_flags[idx_i].append(
                                f"Possible split/structured transaction with {row_j['transaction_id']} "
                                f"-- combined ${combined:,.2f} exceeds the ${limit:,.0f} limit ({day_gap}d apart)"
                            )
                            extra_flags[idx_j].append(
                                f"Possible split/structured transaction with {row_i['transaction_id']} "
                                f"-- combined ${combined:,.2f} exceeds the ${limit:,.0f} limit ({day_gap}d apart)"
                            )

        for i in range(len(df)):
            if extra_flags[i]:
                df.at[i, "rule_flags"] = df.at[i, "rule_flags"] + extra_flags[i]
        df["rule_flag_count"] = df["rule_flags"].apply(len)
        return df

    def _apply_ml_anomaly_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Unsupervised anomaly detection with Isolation Forest over encoded
        transaction features. Produces `ml_anomaly_score` in [0, 100], where
        higher means more anomalous relative to the rest of the batch."""
        n = len(df)
        if n < 10:
            # too little data for a meaningful model; skip ML scoring
            df["ml_anomaly_score"] = 0.0
            return df

        features = pd.DataFrame(index=df.index)
        features["amount"] = df["amount"]
        features["day_of_week"] = df["day_of_week"]

        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        cat_cols = ["category", "department", "vendor", "payment_method"]
        features[cat_cols] = encoder.fit_transform(df[cat_cols].astype(str))

        model = IsolationForest(
            n_estimators=200,
            contamination=self.config.contamination,
            random_state=42,
        )
        model.fit(features)
        raw_scores = model.decision_function(features)  # higher = more normal

        # normalize to 0-100 anomaly score, higher = more anomalous
        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s - min_s < 1e-9:
            normalized = np.zeros(n)
        else:
            normalized = (max_s - raw_scores) / (max_s - min_s) * 100

        df["ml_anomaly_score"] = np.round(normalized, 1)
        return df

    def _score_and_recommend(self, df: pd.DataFrame) -> pd.DataFrame:
        # Weighted blend: rule violations are the strongest signal because
        # they are policy-certain; the ML score adds a softer statistical
        # signal for things no fixed rule captures.
        rule_component = df["rule_flag_count"].clip(upper=4) / 4 * 70
        ml_component = df["ml_anomaly_score"] / 100 * 30
        risk_score = (rule_component + ml_component).round(1)
        df["risk_score"] = risk_score.clip(0, 100)

        def tier(score):
            if score >= self.config.high_risk_threshold:
                return "High"
            if score >= self.config.medium_risk_threshold:
                return "Medium"
            return "Low"

        df["risk_tier"] = df["risk_score"].apply(tier)

        def recommendation(row):
            if row["risk_tier"] == "High":
                return "Escalate to Finance Manager"
            if row["risk_tier"] == "Medium":
                return "Manual Review Required"
            return "Auto-Approve"

        df["recommended_action"] = df.apply(recommendation, axis=1)

        def explain(row):
            if not row["rule_flags"] and row["ml_anomaly_score"] < 40:
                return "No policy violations detected; spending pattern is consistent with normal activity."
            parts = list(row["rule_flags"])
            if row["ml_anomaly_score"] >= 40:
                parts.append(
                    f"Statistical anomaly model flagged this transaction as unusual "
                    f"relative to peer transactions (anomaly score {row['ml_anomaly_score']:.0f}/100)."
                )
            return " | ".join(parts)

        df["explanation"] = df.apply(explain, axis=1)
        return df

    # ------------------------------------------------------------------ #
    # Narrative summary helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _template_narrative_summary(stats: dict, top_flags: pd.DataFrame) -> str:
        lines = [
            f"Reviewed {stats['total_transactions']} transactions totaling "
            f"${stats['total_amount_reviewed']:,.2f}.",
            f"{stats['high_risk']} flagged High risk, {stats['medium_risk']} Medium risk, "
            f"{stats['low_risk']} Low risk (auto-approvable).",
            f"${stats['flagged_amount']:,.2f} in claimed spend requires review before reimbursement.",
        ]
        if len(top_flags) > 0:
            lines.append("Top items needing immediate attention:")
            for _, r in top_flags.iterrows():
                lines.append(
                    f"  - {r['transaction_id']} ({r['employee_name']}, {r['category']}, "
                    f"${r['amount']:,.2f}): {r['explanation']}"
                )
        return "\n".join(lines)

    @staticmethod
    def _llm_narrative_summary(stats: dict, top_flags: pd.DataFrame, api_key: str) -> str:
        """Optional enhancement: uses the Anthropic API to produce a more
        natural management narrative. Only called if the user has supplied
        their own ANTHROPIC_API_KEY via Streamlit secrets / environment --
        never required for the agent's core functionality."""
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        flags_text = "\n".join(
            f"- {r['transaction_id']} ({r['employee_name']}, {r['category']}, ${r['amount']:,.2f}): {r['explanation']}"
            for _, r in top_flags.iterrows()
        )
        prompt = (
            "You are an accounting controls assistant. Write a concise (<150 words) "
            "management summary of this expense review batch for a Finance Manager. "
            "Be factual and specific, no fluff.\n\n"
            f"Stats: {stats}\n\nTop flagged items:\n{flags_text}"
        )
        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
