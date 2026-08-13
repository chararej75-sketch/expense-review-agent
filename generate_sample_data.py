"""
generate_sample_data.py

Creates a simulated, fully synthetic corporate expense-transaction dataset
for the Expense Review Agent. No real people, vendors, or company data are
used -- this is safe to commit publicly and use for grading/testing.

Run:
    python generate_sample_data.py

Output:
    data/sample_expenses.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG_SEED = 42
N_NORMAL = 460
N_ANOMALOUS = 40  # ~8% intentionally-flaggable records, mirrors real audit rates

np.random.seed(RNG_SEED)

DEPARTMENTS = ["Sales", "Marketing", "Engineering", "Operations", "Finance", "HR", "Executive"]

EMPLOYEES = [
    ("E{:03d}".format(i), name, dept)
    for i, (name, dept) in enumerate(
        [
            ("Alex Morgan", "Sales"), ("Jamie Chen", "Sales"), ("Priya Nair", "Marketing"),
            ("Sam Rivera", "Marketing"), ("Taylor Brooks", "Engineering"), ("Jordan Lee", "Engineering"),
            ("Casey Kim", "Operations"), ("Morgan Diaz", "Operations"), ("Riley Cooper", "Finance"),
            ("Drew Patel", "Finance"), ("Avery Scott", "HR"), ("Quinn Bailey", "HR"),
            ("Reese Turner", "Executive"), ("Skyler James", "Executive"), ("Cameron Ross", "Sales"),
            ("Harper Wells", "Marketing"), ("Rowan Blake", "Engineering"), ("Emerson Hale", "Operations"),
            ("Finley Grant", "Finance"), ("Dakota Reyes", "HR"),
        ],
        start=1,
    )
]

CATEGORIES = {
    "Meals & Entertainment": {"limit": 150, "vendors": ["The Grill House", "Cafe Nine", "Bistro 22", "Uptown Diner"]},
    "Travel - Airfare": {"limit": 1200, "vendors": ["Delta Air", "United Airlines", "American Airlines"]},
    "Travel - Lodging": {"limit": 400, "vendors": ["Marriott", "Hilton", "Holiday Inn"]},
    "Ground Transport": {"limit": 120, "vendors": ["Uber", "Lyft", "City Taxi"]},
    "Office Supplies": {"limit": 250, "vendors": ["Staples", "Office Depot", "Amazon Business"]},
    "Software & Subscriptions": {"limit": 500, "vendors": ["Adobe", "Microsoft", "Slack", "Zoom"]},
    "Client Gifts": {"limit": 100, "vendors": ["Gift Emporium", "Corporate Gifts Co"]},
    "Training & Conferences": {"limit": 900, "vendors": ["EventBrite", "Conf Registration Inc"]},
}

PAYMENT_METHODS = ["Corporate Card", "Personal Card - Reimbursement", "Cash - Reimbursement"]

START_DATE = datetime(2026, 1, 1)
END_DATE = datetime(2026, 6, 30)


def random_date():
    delta_days = (END_DATE - START_DATE).days
    d = START_DATE + timedelta(days=int(np.random.randint(0, delta_days)))
    return d


def make_normal_row(txn_id):
    emp_id, name, dept = EMPLOYEES[np.random.randint(len(EMPLOYEES))]
    category = np.random.choice(list(CATEGORIES.keys()))
    cat_info = CATEGORIES[category]
    vendor = np.random.choice(cat_info["vendors"])
    limit = cat_info["limit"]
    # normal spend: well under the policy limit, non-round amounts
    amount = round(np.random.uniform(0.15, 0.75) * limit + np.random.uniform(0, 5), 2)
    date = random_date()
    payment = np.random.choice(PAYMENT_METHODS, p=[0.6, 0.35, 0.05])
    receipt = True if amount < 75 and np.random.rand() < 0.9 else np.random.rand() < 0.97
    approver_pool = [e for e in EMPLOYEES if e[2] == "Finance" or e[1] != name]
    approved_by = approver_pool[np.random.randint(len(approver_pool))][1]

    return {
        "transaction_id": txn_id,
        "employee_id": emp_id,
        "employee_name": name,
        "department": dept,
        "category": category,
        "vendor": vendor,
        "amount": amount,
        "currency": "USD",
        "date": date.strftime("%Y-%m-%d"),
        "payment_method": payment,
        "description": f"{category} expense at {vendor}",
        "receipt_attached": bool(receipt),
        "approved_by": approved_by,
    }


def make_anomalous_row(txn_id, kind):
    emp_id, name, dept = EMPLOYEES[np.random.randint(len(EMPLOYEES))]
    category = np.random.choice(list(CATEGORIES.keys()))
    cat_info = CATEGORIES[category]
    vendor = np.random.choice(cat_info["vendors"])
    limit = cat_info["limit"]
    date = random_date()
    payment = np.random.choice(PAYMENT_METHODS, p=[0.5, 0.4, 0.1])
    approved_by = name  # default; overridden below for most kinds

    if kind == "over_limit":
        amount = round(limit * np.random.uniform(1.4, 3.0), 2)
        receipt = np.random.rand() < 0.6
        approved_by = "Riley Cooper"
    elif kind == "missing_receipt":
        amount = round(np.random.uniform(0.4, 0.9) * limit, 2)
        receipt = False
        approved_by = "Riley Cooper"
    elif kind == "round_number":
        amount = float(np.random.choice([100, 200, 300, 500, 750, 1000]))
        receipt = np.random.rand() < 0.5
        approved_by = "Riley Cooper"
    elif kind == "self_approved":
        amount = round(np.random.uniform(0.3, 0.8) * limit, 2)
        receipt = True
        approved_by = name  # employee approved their own expense
    elif kind == "weekend_submission":
        # force date onto a Saturday/Sunday
        d = random_date()
        while d.weekday() < 5:
            d = random_date()
        date = d
        amount = round(np.random.uniform(0.3, 0.9) * limit, 2)
        receipt = np.random.rand() < 0.7
        approved_by = "Riley Cooper"
    else:  # structuring: just-under-limit, will be paired with a duplicate below
        amount = round(limit * np.random.uniform(0.9, 0.99), 2)
        receipt = True
        approved_by = "Riley Cooper"

    row = {
        "transaction_id": txn_id,
        "employee_id": emp_id,
        "employee_name": name,
        "department": dept,
        "category": category,
        "vendor": vendor,
        "amount": amount,
        "currency": "USD",
        "date": date.strftime("%Y-%m-%d"),
        "payment_method": payment,
        "description": f"{category} expense at {vendor}",
        "receipt_attached": bool(receipt),
        "approved_by": approved_by,
    }

    if kind == "duplicate_or_structuring":
        return row, (emp_id, name, dept, category, vendor, date, payment)
    return row


def main():
    rows = []
    txn_counter = 1

    for _ in range(N_NORMAL):
        rows.append(make_normal_row(f"TXN{txn_counter:05d}"))
        txn_counter += 1

    anomaly_kinds = (
        ["over_limit"] * 10
        + ["missing_receipt"] * 8
        + ["round_number"] * 7
        + ["self_approved"] * 6
        + ["weekend_submission"] * 5
        + ["duplicate_or_structuring"] * 4
    )
    np.random.shuffle(anomaly_kinds)

    duplicate_seeds = []
    for kind in anomaly_kinds[:N_ANOMALOUS]:
        if kind == "duplicate_or_structuring":
            row, seed = make_anomalous_row(f"TXN{txn_counter:05d}", kind)
            rows.append(row)
            txn_counter += 1
            duplicate_seeds.append((row, seed))
        else:
            row = make_anomalous_row(f"TXN{txn_counter:05d}", kind)
            rows.append(row)
            txn_counter += 1

    # create true duplicate / split-transaction pairs (structuring pattern:
    # two submissions from the same employee/vendor/day that together exceed
    # the category limit, each individually just under it)
    for row, seed in duplicate_seeds:
        emp_id, name, dept, category, vendor, date, payment = seed
        dup = dict(row)
        dup["transaction_id"] = f"TXN{txn_counter:05d}"
        dup["amount"] = round(row["amount"] * np.random.uniform(0.85, 1.05), 2)
        dup["description"] = f"{category} expense at {vendor} (split submission)"
        rows.append(dup)
        txn_counter += 1

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=RNG_SEED).reset_index(drop=True)
    df["amount"] = df["amount"].astype(float)

    df.to_csv("data/sample_expenses.csv", index=False)
    print(f"Wrote {len(df)} rows to data/sample_expenses.csv")


if __name__ == "__main__":
    main()
