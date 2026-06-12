"""
Synthetic Data Engine
=====================
Creates a local SQLite database (`fintech.db`) with:
  - `transactions`: 5,000 mock banking transactions
  - `disputes`:       500 mock customer disputes linked to transactions

Each dispute has a hidden ground-truth label (`true_category`) drawn from
{Fraud, Merchant Error, Policy Violation}. The label is used ONLY for
training/calibrating the classifier - the agent never sees it at decision
time, which keeps the evaluation honest.

Usage:
    python generate_data.py [--seed 42]
"""

import argparse
import random
import sqlite3

from faker import Faker

DB_PATH = "fintech.db"
N_TRANSACTIONS = 5_000
N_DISPUTES = 500

CATEGORIES = ["Fraud", "Merchant Error", "Policy Violation"]

MERCHANT_CATEGORIES = [
    "Groceries", "Electronics", "Travel", "Dining", "Subscription",
    "Fashion", "Fuel", "Entertainment", "Healthcare", "Utilities",
]

# ---------------------------------------------------------------------------
# Dispute text generation.
# Each category has distinct vocabulary/templates plus shared "noise" clauses
# so the classes overlap a little - a perfectly separable dataset would make
# conformal prediction trivially confident and the demo dishonest.
# ---------------------------------------------------------------------------

FRAUD_TEMPLATES = [
    "I have never been to {merchant}. This charge of {amount} on {date} was not made by me. I think my card details were stolen.",
    "Unrecognised transaction at {merchant} for {amount}. My card was in my possession the whole time. Please block my card immediately.",
    "Someone has used my account at {merchant} without my permission. I did not authorise this {amount} payment.",
    "There is a payment to {merchant} I never made. I lost my card last week and believe it was used fraudulently.",
    "This {amount} charge at {merchant} is fraud. I was at work when it happened and have never shopped there.",
    "I just noticed a transaction at {merchant} that I did not make. My phone was stolen recently and I suspect my banking app was accessed.",
    "Multiple unauthorised charges from {merchant} appeared overnight. I demand these are reversed and my account secured.",
    "I received a text about a {amount} purchase at {merchant} which I never approved. This is identity theft.",
]

MERCHANT_ERROR_TEMPLATES = [
    "{merchant} charged me {amount} twice for the same order on {date}. I only made one purchase.",
    "I was billed {amount} by {merchant} but the order was cancelled before it shipped. The refund never arrived.",
    "The amount taken by {merchant} is wrong. I agreed to pay less at checkout but was charged {amount}.",
    "I returned the item to {merchant} three weeks ago and still have not received my {amount} refund.",
    "{merchant} took payment of {amount} but my order never arrived and their support is not responding.",
    "Duplicate charge from {merchant}. My banking app shows two identical payments of {amount} seconds apart.",
    "The goods from {merchant} arrived damaged and unusable. I paid {amount} and the merchant refuses to refund me.",
    "I was overcharged by {merchant} on {date}. The receipt says one price but {amount} was debited.",
]

POLICY_VIOLATION_TEMPLATES = [
    "I signed up for a free trial with {merchant} and was charged {amount} without any warning. The cancellation link does not work.",
    "{merchant} keeps charging me {amount} every month even though I cancelled the subscription in writing on {date}.",
    "The terms said no fees, but {merchant} added hidden charges totalling {amount}. This breaches their own policy.",
    "I cancelled my membership with {merchant} inside the cooling-off period but was still billed {amount}.",
    "{merchant} renewed my subscription automatically at a higher price of {amount} without notifying me as their terms require.",
    "I was charged {amount} by {merchant} after closing my account. Their policy states no charges after closure.",
    "The advertised price at {merchant} did not include mandatory fees. Final charge of {amount} violates advertised terms.",
    "{merchant} billed me {amount} during a promotional period that was supposed to be free under their offer terms.",
]

# Shared filler clauses that can appear in any category (class overlap / noise)
NOISE_CLAUSES = [
    "I have been a loyal customer for years.",
    "Please resolve this as soon as possible.",
    "This has caused me significant stress.",
    "I have attached my statement as evidence.",
    "I tried contacting support but got no reply.",
    "I expect a prompt refund.",
    "This is the second time something like this has happened.",
    "",
    "",
]

TEMPLATES = {
    "Fraud": FRAUD_TEMPLATES,
    "Merchant Error": MERCHANT_ERROR_TEMPLATES,
    "Policy Violation": POLICY_VIOLATION_TEMPLATES,
}


def build_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS disputes;
        DROP TABLE IF EXISTS transactions;
        DROP TABLE IF EXISTS agent_audit_log;

        CREATE TABLE transactions (
            transaction_id   INTEGER PRIMARY KEY,
            customer_id      TEXT    NOT NULL,
            customer_name    TEXT    NOT NULL,
            merchant         TEXT    NOT NULL,
            merchant_category TEXT   NOT NULL,
            amount           REAL    NOT NULL,
            currency         TEXT    NOT NULL DEFAULT 'GBP',
            timestamp        TEXT    NOT NULL,
            country          TEXT    NOT NULL
        );

        CREATE TABLE disputes (
            dispute_id       INTEGER PRIMARY KEY,
            transaction_id   INTEGER NOT NULL REFERENCES transactions(transaction_id),
            opened_at        TEXT    NOT NULL,
            dispute_text     TEXT    NOT NULL,
            true_category    TEXT    NOT NULL,   -- ground truth (hidden from agent)
            status           TEXT    NOT NULL DEFAULT 'Pending',
            ai_category      TEXT,
            ai_confidence    REAL,
            ai_credibility   REAL,
            resolved_at      TEXT
        );

        CREATE TABLE agent_audit_log (
            log_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            dispute_id  INTEGER NOT NULL,
            action      TEXT    NOT NULL,
            detail      TEXT,
            logged_at   TEXT    NOT NULL DEFAULT (datetime('now'))
        );
        """
    )


def generate_transactions(fake: Faker, rng: random.Random) -> list[tuple]:
    rows = []
    for txn_id in range(1, N_TRANSACTIONS + 1):
        ts = fake.date_time_between(start_date="-180d", end_date="now")
        rows.append(
            (
                txn_id,
                f"CUST-{rng.randint(10_000, 99_999)}",
                fake.name(),
                fake.company(),
                rng.choice(MERCHANT_CATEGORIES),
                round(rng.uniform(2.5, 1_850.0), 2),
                rng.choice(["GBP", "GBP", "GBP", "EUR", "USD"]),
                ts.isoformat(sep=" ", timespec="seconds"),
                fake.country_code(),
            )
        )
    return rows


def render_dispute_text(rng: random.Random, category: str, merchant: str,
                        amount: float, currency: str, date: str) -> str:
    template = rng.choice(TEMPLATES[category])
    symbol = {"GBP": "£", "EUR": "€", "USD": "$"}[currency]
    text = template.format(
        merchant=merchant,
        amount=f"{symbol}{amount:.2f}",
        date=date.split(" ")[0],
    )
    noise = rng.choice(NOISE_CLAUSES)
    if noise:
        text = f"{text} {noise}"
    return text


def generate_disputes(conn: sqlite3.Connection, fake: Faker,
                      rng: random.Random) -> list[tuple]:
    txn_rows = conn.execute(
        "SELECT transaction_id, merchant, amount, currency, timestamp "
        "FROM transactions"
    ).fetchall()
    disputed = rng.sample(txn_rows, N_DISPUTES)

    rows = []
    for dispute_id, (txn_id, merchant, amount, currency, ts) in enumerate(
        disputed, start=1
    ):
        category = rng.choice(CATEGORIES)
        opened = fake.date_time_between(start_date="-30d", end_date="now")
        rows.append(
            (
                dispute_id,
                txn_id,
                opened.isoformat(sep=" ", timespec="seconds"),
                render_dispute_text(rng, category, merchant, amount, currency, ts),
                category,
                "Pending",
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic fintech data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    fake = Faker("en_GB")
    Faker.seed(args.seed)

    conn = sqlite3.connect(DB_PATH)
    build_schema(conn)

    conn.executemany(
        "INSERT INTO transactions VALUES (?,?,?,?,?,?,?,?,?)",
        generate_transactions(fake, rng),
    )
    conn.executemany(
        "INSERT INTO disputes (dispute_id, transaction_id, opened_at, "
        "dispute_text, true_category, status) VALUES (?,?,?,?,?,?)",
        generate_disputes(conn, fake, rng),
    )
    conn.commit()

    n_txn = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    n_disp = conn.execute("SELECT COUNT(*) FROM disputes").fetchone()[0]
    by_cat = conn.execute(
        "SELECT true_category, COUNT(*) FROM disputes GROUP BY true_category"
    ).fetchall()
    conn.close()

    print(f"Created {DB_PATH}")
    print(f"  transactions: {n_txn}")
    print(f"  disputes:     {n_disp}")
    for cat, count in by_cat:
        print(f"    - {cat}: {count}")


if __name__ == "__main__":
    main()
