# Expense Review Agent

An AI decision-support agent for **Accounts Payable / expense audit**: it reviews batches of employee expense claims, flags policy violations and statistical anomalies, assigns a 0-100 risk score, and recommends an action (auto-approve, manual review, or escalate) -- with a plain-language, auditable explanation for every transaction.

- **Live app:**  https://expense-review-agent-hqsfyjjy9capprqxgwkqhppq.streamlit.app
- **GitHub repo:** https://github.com/chararej75-sketch/expense-review-agent

---

## 1. The business problem

Every expense claim submitted by an employee is supposed to be checked against company policy before it's reimbursed and posted to the ledger: is the amount within the category limit, is a receipt attached, did someone other than the claimant approve it (segregation of duties), is the claim a duplicate or a "split" submission designed to dodge an approval threshold, and so on.

In practice, Accounts Payable teams review these claims manually, which does not scale as transaction volume grows, and is inconsistent between reviewers -- the same violation might get caught by one reviewer and missed by another. This creates real financial-control risk: unauthorized spend goes unnoticed, and reviewers waste time re-checking claims that are perfectly fine.

**Why an agent fits this problem well:** the checks needed are a mix of deterministic accounting rules (a spending limit is a spending limit) *and* fuzzier statistical judgment (is this specific claim unusual, even if no single rule catches it?). A rules-only system misses novel anomalies; a black-box ML-only system isn't auditable enough for a financial control. Combining both, with every flag explained in plain language, gives Finance a fast, consistent, and defensible first-pass triage -- while a human still makes every final call.

## 2. What the agent does

Given a CSV of expense transactions, the agent:

1. **Applies deterministic policy rules** (explainable, matches standard AP/audit controls):
   - Category spending limit exceeded
   - Missing receipt above a disclosure threshold
   - Suspiciously round claim amount (possible estimate rather than an actual receipt)
   - Self-approval -- approver is the same person as the claimant (segregation-of-duties violation)
   - Weekend/holiday submission
   - Duplicate submissions, and "structured"/split transactions (two claims from the same employee/vendor/category, close together in time, that individually sit under the policy limit but combined exceed it)
2. **Runs an unsupervised ML anomaly model** (`sklearn.ensemble.IsolationForest`) over encoded transaction features (amount, day of week, category, department, vendor, payment method) to catch anomalies no fixed rule captures.
3. **Blends both signals** into a single 0-100 risk score and a Low / Medium / High risk tier.
4. **Recommends an action** for each transaction: Auto-Approve, Manual Review Required, or Escalate to Finance Manager.
5. **Explains itself** in plain language for every flagged transaction, and produces a management-facing narrative summary of the batch.

The agent never approves, rejects, or pays a transaction itself -- it only recommends and explains. A human accountant remains in the loop for every decision, which is required by internal-control best practice (segregation of duties) and keeps the system auditable.

## 3. Project structure

```
expense-review-agent/
├── app.py                          # Streamlit web app (deploy this)
├── agent.py                        # Core agent logic (rules + ML + scoring), imported by app.py
├── generate_sample_data.py         # Builds the synthetic sample dataset
├── data/
│   └── sample_expenses.csv         # Simulated expense data (fully synthetic, safe to share)
├── notebook/
│   └── Expense_Review_Agent.ipynb  # Colab notebook: EDA, model development, evaluation
├── requirements.txt
├── .streamlit/
│   └── secrets.toml.example        # Template for the OPTIONAL Claude API key
├── .gitignore
└── README.md
```

## 4. How the agent decides (methodology)

| Step | Technique | Why |
|---|---|---|
| Policy checks | Deterministic rules on amount, receipt flag, approver, date, and pairwise comparisons within employee/vendor/category groups | Matches real audit controls; fully explainable and auditable |
| Anomaly detection | Isolation Forest (unsupervised) over encoded transaction features | Catches unusual patterns no fixed rule anticipates, without needing labeled fraud data |
| Risk scoring | `risk_score = 70% x (rule flags, capped) + 30% x (ML anomaly score)` | Rule violations are policy-certain and weighted higher; ML adds a softer statistical signal |
| Risk tiering | Score >= 65 -> High, >= 35 -> Medium, else Low (thresholds adjustable in the app sidebar) | Turns a continuous score into an actionable triage bucket |
| Explanation | Every flag is rendered as a plain-language sentence, concatenated per transaction | No black-box outputs -- a reviewer can see exactly why something was flagged |

Full implementation is in `agent.py`; the same logic is walked through step-by-step, with EDA and evaluation, in `notebook/Expense_Review_Agent.ipynb`.

## 5. Data

`data/sample_expenses.csv` (504 rows) is **entirely synthetic** -- generated by `generate_sample_data.py` with fictional employee names, vendors, and amounts. About 8% of rows have an intentionally injected policy violation (over-limit spend, missing receipt, round-number amount, self-approval, weekend submission, or a split/structured transaction pair) so the agent's detection can be sanity-checked against known ground truth. No real company, employee, or client data is used anywhere in this project.

**Expected input CSV columns** (for your own data, via the app's upload option):

```
transaction_id, employee_id, employee_name, department, category, vendor,
amount, date, payment_method, receipt_attached, approved_by
```

## 6. Setup instructions (local)

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd expense-review-agent
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# (re)generate the sample dataset if needed
python generate_sample_data.py

# run the app locally
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

### Running the Colab notebook

Open `notebook/Expense_Review_Agent.ipynb` in Google Colab (File -> Upload notebook, or open directly from GitHub via Colab's "GitHub" tab). The first cell has two options: clone the full repo with `!git clone`, or manually upload `agent.py` and `data/sample_expenses.csv` to the Colab session. Then run all cells top to bottom.

## 7. Deployment (Streamlit Community Cloud)

1. Push this project to a **public GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select your repo, branch `main`, and set the main file path to `app.py`.
4. (Optional) If you want to enable the "Use Claude AI for narrative summary" toggle, go to your app's **Settings -> Secrets** and add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-your-own-key-here"
   ```
   This is entirely optional -- the agent's core scoring, flagging, and template-based summary work fully without any API key.
5. Click **Deploy**. Once live, copy the app URL into this README and your submission.

## 8. Security and confidentiality

- No API keys, passwords, or credentials are committed to this repository. `.streamlit/secrets.toml` is git-ignored; only the `.example` template is committed.
- The optional Claude API key is read exclusively from Streamlit secrets (or an environment variable) at runtime -- never hard-coded.
- All data used for testing/demonstration (`data/sample_expenses.csv`) is synthetic and publicly shareable. To use the app with real company data, upload it via the app's file uploader at runtime; it is processed in-memory for that session only and is never written back to this repository.

## 9. Limitations and next steps

- The Isolation Forest is unsupervised and re-fit on whatever batch is uploaded; with very small batches (<10 rows) ML scoring is skipped and the agent falls back to rule-based checks only.
- Policy limits and risk thresholds are configurable in the app sidebar but are not currently persisted between sessions -- a production version would store per-company policy configuration.
- The agent supports single-batch CSV review; a production deployment would likely integrate directly with an ERP/expense system (e.g., via API) rather than manual CSV upload.

---

*Submitted as Project 2. Report links: GitHub repository and live Streamlit app URL are listed at the top of this README and on the last page of the submission report.*
