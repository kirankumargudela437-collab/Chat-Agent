# Comparing with the previous or next row (LAG, LEAD)
id: lag-lead
cues: previous, prior, last month, change, growth, increase, decrease, difference from, compared to the previous, month over month, next, gap between, days since
use: LAG(col) OVER (ORDER BY ...) reads the previous row's value and LEAD the next one; subtract to get the change.
avoid: Self-joining on date minus one (breaks on gaps); forgetting PARTITION BY when the comparison is per customer or per product.
manual: window functions LAG LEAD previous next row

## Example
Q: Monthly revenue and the change from the previous month.
```sql
WITH monthly AS (
  SELECT strftime('%Y-%m', o.order_date) AS month, SUM(oi.quantity * oi.unit_price) AS revenue
  FROM orders o JOIN order_items oi ON oi.order_id = o.order_id GROUP BY month
)
SELECT month, revenue, revenue - LAG(revenue) OVER (ORDER BY month) AS change FROM monthly;
```
Why: LAG brings last month's revenue onto this month's row.

## Example
Q: Days between each customer's consecutive orders.
```sql
SELECT customer_id, order_id, order_date,
       julianday(order_date) - julianday(LAG(order_date) OVER (PARTITION BY customer_id ORDER BY order_date)) AS days_since_previous
FROM orders;
```
Why: PARTITION BY customer keeps the comparison within one customer.

## Example
Q: Month-over-month growth in percent.
```sql
WITH monthly AS (
  SELECT strftime('%Y-%m', o.order_date) AS month, SUM(oi.quantity * oi.unit_price) AS revenue
  FROM orders o JOIN order_items oi ON oi.order_id = o.order_id GROUP BY month
)
SELECT month, revenue,
       100.0 * (revenue - LAG(revenue) OVER (ORDER BY month)) / NULLIF(LAG(revenue) OVER (ORDER BY month), 0) AS growth_pct
FROM monthly;
```
Why: change divided by the previous value, guarded against zero.

## Example
Q: For each order, the date of the same customer's next order.
```sql
SELECT customer_id, order_id, order_date,
       LEAD(order_date) OVER (PARTITION BY customer_id ORDER BY order_date) AS next_order_date
FROM orders;
```
Why: LEAD looks one row ahead.

## Example
Q: Employees hired in order, with who was hired just before them.
```sql
SELECT name, hire_date, LAG(name) OVER (ORDER BY hire_date) AS hired_before FROM employees;
```
Why: LAG works on any column, not only numbers.
