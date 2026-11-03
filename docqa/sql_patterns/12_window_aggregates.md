# Running totals and shares (window aggregates)
id: window-aggregate
cues: running total, cumulative, so far, to date, moving average, rolling, share of, percent of total, compared with the group average, each row with the total
use: SUM/AVG/COUNT(...) OVER (...) adds a group or running value to every row without collapsing rows. ORDER BY inside OVER makes it cumulative.
avoid: GROUP BY when each row must stay; forgetting ORDER BY in OVER for running totals.

## Example
Q: Running total of order revenue by date.
```sql
WITH rev AS (
  SELECT o.order_id, o.order_date, SUM(oi.quantity * oi.unit_price) AS revenue
  FROM orders o JOIN order_items oi ON oi.order_id = o.order_id GROUP BY o.order_id, o.order_date
)
SELECT order_date, revenue, SUM(revenue) OVER (ORDER BY order_date, order_id) AS running_total FROM rev;
```
Why: SUM OVER (ORDER BY date) adds up everything so far.

## Example
Q: Each employee's salary next to their department's average.
```sql
SELECT name, dept_id, salary, AVG(salary) OVER (PARTITION BY dept_id) AS dept_avg FROM employees;
```
Why: PARTITION BY gives the per-department value on every row.

## Example
Q: Each product's share of total revenue, in percent.
```sql
WITH rev AS (
  SELECT p.name, SUM(oi.quantity * oi.unit_price) AS revenue
  FROM products p JOIN order_items oi ON oi.product_id = p.product_id GROUP BY p.product_id, p.name
)
SELECT name, revenue, 100.0 * revenue / SUM(revenue) OVER () AS pct_of_total FROM rev;
```
Why: OVER () is the grand total across all rows.

## Example
Q: Three-order moving average of order revenue.
```sql
WITH rev AS (
  SELECT o.order_id, o.order_date, SUM(oi.quantity * oi.unit_price) AS revenue
  FROM orders o JOIN order_items oi ON oi.order_id = o.order_id GROUP BY o.order_id, o.order_date
)
SELECT order_date, revenue,
       AVG(revenue) OVER (ORDER BY order_date, order_id ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS moving_avg
FROM rev;
```
Why: the ROWS frame limits the average to the current and two previous orders.

## Example
Q: Number each customer's orders in date order.
```sql
SELECT customer_id, order_id, order_date,
       COUNT(*) OVER (PARTITION BY customer_id ORDER BY order_date, order_id) AS order_number
FROM orders;
```
Why: a running count per customer.
