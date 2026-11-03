"""``docqa sample``: a small workspace to try every route of the agent.

Creates::

    <folder>/
      my_company.db                     SQLite: customers, orders
      workspace_files/
        docs/company_policies.md        -> DOC   (SLA, refunds, expenses)
        docs/data_dictionary.md         -> business rules the SQL route must apply
        code/auth_service.py            -> CODE  (Python)
        code/api_client.ts              -> CODE  (TypeScript)
        sql/schema.sql                  -> SQL   (DDL with comments)
        sql/golden_queries.sql          -> golden question-to-SQL examples

Questions to try:

* DOC  "What uptime do Enterprise customers get, and what happens if we miss it?"
* CODE "How does validate_request_token detect an expired token?"
* SQL  "Rank customers by net revenue." (the data dictionary says refunds don't count)
* SQL  "Month-over-month net revenue growth" (follows the golden example's LAG pattern)
* follow-up "Only Enterprise customers."
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

POLICIES_MD = """# Company Policies (2026)

## Service Level Agreement (SLA)

- **Enterprise customers:** 99.99% monthly uptime, 15-minute incident response.
- **Standard customers:** 99.5% monthly uptime, 4-hour incident response.
- **Service credit:** if monthly uptime falls below the guarantee, the customer gets a 25% credit on the next invoice.

## Refunds

Customers may request a refund within 30 days of an order. Refunded orders are marked with status `R` in the orders system.

## Remote work and equipment

- Full-time engineers receive a $1,500 home office stipend every 24 months.
- Expense reports must be submitted before the 25th of each month through the finance portal.
"""

DATA_DICTIONARY_MD = """# Sales Data Dictionary

How business terms map to the `customers` and `orders` tables.

## Order status codes

`orders.status` is one of:

- `C` = Completed
- `R` = Refunded (the full amount was returned)
- `P` = Pending (not yet paid)

## Metrics

- **Net revenue** is the sum of `orders.amount_usd` for orders with `status = 'C'` only.
  Refunded (`R`) and pending (`P`) orders never count as revenue.
- **Gross bookings** is the sum of `amount_usd` over all orders regardless of status.
- **Active customer**: a customer with at least one completed order in the 90 days before the reporting date.
- **Order month** is `strftime('%Y-%m', order_date)` in SQLite (`DATE_TRUNC('month', order_date)` in PostgreSQL).

## Join path

Join `orders.customer_id` to `customers.customer_id`. A customer's segment is `customers.tier`
(`Enterprise` or `Standard`).
"""

AUTH_PY = '''import hashlib
import hmac
import time


class TokenExpiredError(ValueError):
    """The request token is older than the allowed time-to-live."""


class TokenValidator:
    """Validates HMAC-signed API request tokens and their age."""

    def __init__(self, secret_key: str, ttl_seconds: int = 3600):
        self.secret_key = secret_key.encode("utf-8")
        self.ttl_seconds = ttl_seconds

    def validate_request_token(self, user_id: str, timestamp: int, signature: str) -> bool:
        # Reject tokens older than ttl_seconds (one hour by default).
        if int(time.time()) - timestamp > self.ttl_seconds:
            raise TokenExpiredError("Request token has exceeded its TTL.")

        payload = f"{user_id}:{timestamp}".encode("utf-8")
        expected = hmac.new(self.secret_key, payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature)


def calculate_retry_backoff(attempt: int) -> int:
    """Exponential backoff in seconds, capped at 60."""
    return min(60, (2 ** attempt) + 1)
'''

API_CLIENT_TS = '''export interface Order {
  orderId: number;
  customerId: number;
  amountUsd: number;
  status: "C" | "R" | "P";
}

const MAX_RETRIES = 3;

export class ApiClient {
  constructor(private baseUrl: string, private token: string) {}

  /** Fetch one customer's orders, retrying up to MAX_RETRIES times on HTTP 5xx. */
  async getOrders(customerId: number): Promise<Order[]> {
    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
      const res = await fetch(`${this.baseUrl}/customers/${customerId}/orders`, {
        headers: { Authorization: `Bearer ${this.token}` },
      });
      if (res.ok) return (await res.json()) as Order[];
      if (res.status < 500) throw new Error(`Request failed: ${res.status}`);
    }
    throw new Error(`Gave up after ${MAX_RETRIES} attempts`);
  }
}
'''

SCHEMA_SQL = """-- Sales database (SQLite). Business rules for these columns are in docs/data_dictionary.md.

CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    tier        TEXT NOT NULL,      -- 'Enterprise' or 'Standard'
    country     TEXT NOT NULL
);

CREATE TABLE orders (
    order_id     INTEGER PRIMARY KEY,
    customer_id  INTEGER NOT NULL REFERENCES customers(customer_id),
    product_name TEXT NOT NULL,
    amount_usd   REAL NOT NULL,
    order_date   TEXT NOT NULL,     -- ISO date, e.g. '2026-03-01'
    status       TEXT NOT NULL      -- 'C' completed, 'R' refunded, 'P' pending
);
"""

GOLDEN_SQL = """-- Golden SQL examples: verified questions and queries the agent copies the style of.
-- Format: a '-- Question:' line, then the query, ending with ';'.

-- Question: Month-over-month net revenue growth
WITH monthly AS (
    SELECT strftime('%Y-%m', order_date) AS month, SUM(amount_usd) AS net_revenue
    FROM orders
    WHERE status = 'C'
    GROUP BY 1
),
lagged AS (
    SELECT month, net_revenue, LAG(net_revenue) OVER (ORDER BY month) AS prev_revenue
    FROM monthly
)
SELECT month, net_revenue,
       ROUND(100.0 * (net_revenue - prev_revenue) / NULLIF(prev_revenue, 0), 1) AS growth_pct
FROM lagged
ORDER BY month;

-- Question: Top customer by net revenue in each country
WITH revenue AS (
    SELECT c.country, c.name, SUM(o.amount_usd) AS net_revenue
    FROM orders o
    JOIN customers c ON c.customer_id = o.customer_id
    WHERE o.status = 'C'
    GROUP BY c.country, c.name
),
ranked AS (
    SELECT *, RANK() OVER (PARTITION BY country ORDER BY net_revenue DESC) AS rnk
    FROM revenue
)
SELECT country, name, net_revenue FROM ranked WHERE rnk = 1 ORDER BY net_revenue DESC;
"""

CUSTOMERS = [
    (1, "Acme Corp", "Enterprise", "USA"),
    (2, "Globex Inc", "Standard", "UK"),
    (3, "Initech", "Enterprise", "India"),
    (4, "Soylent Corp", "Standard", "Canada"),
    (5, "Umbrella Ltd", "Enterprise", "UK"),
]

ORDERS = [
    (101, 1, "Cloud AI Cluster", 14500.00, "2026-01-15", "C"),
    (102, 1, "Dedicated Support", 2500.00, "2026-02-10", "C"),
    (103, 2, "Basic API Plan", 499.00, "2026-02-18", "C"),
    (104, 3, "Cloud AI Cluster", 14500.00, "2026-03-01", "C"),
    (105, 3, "Custom Vector Node", 6200.00, "2026-03-12", "R"),
    (106, 4, "Basic API Plan", 499.00, "2026-03-20", "C"),
    (107, 5, "Cloud AI Cluster", 14500.00, "2026-03-25", "P"),
    (108, 5, "Dedicated Support", 2500.00, "2026-01-30", "C"),
    (109, 2, "Basic API Plan", 499.00, "2026-03-18", "R"),
    (110, 1, "Custom Vector Node", 6200.00, "2026-03-28", "C"),
]


def create_sample_workspace(folder: Path) -> dict:
    root = Path(folder)
    docs = root / "workspace_files"
    files = {
        "docs/company_policies.md": POLICIES_MD,
        "docs/data_dictionary.md": DATA_DICTIONARY_MD,
        "code/auth_service.py": AUTH_PY,
        "code/api_client.ts": API_CLIENT_TS,
        "sql/schema.sql": SCHEMA_SQL,
        "sql/golden_queries.sql": GOLDEN_SQL,
    }
    for rel, text in files.items():
        p = docs / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    db = root / "my_company.db"
    if db.exists():
        db.unlink()
    conn = sqlite3.connect(db)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)", CUSTOMERS)
        conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)", ORDERS)
        conn.commit()
    finally:
        conn.close()
    return {
        "root": root,
        "docs": docs,
        "db": db,
        "db_url": f"sqlite:///{db.as_posix()}",
        "files": [f"workspace_files/{rel}" for rel in files],
    }
