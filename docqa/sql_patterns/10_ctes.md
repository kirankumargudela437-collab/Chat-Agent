# Step-by-step queries (WITH / CTE)
id: cte
cues: step by step, then, among them, of those, based on that, intermediate result, average of the totals, average per, total per then, first compute
use: WITH name AS (...) gives a sub-result a name, so a complex question becomes readable steps: compute per-group values first, then filter, rank or aggregate them.
avoid: Nesting many subqueries instead; averaging rows when the question wants the average of per-group totals.

## Example
Q: What is the average revenue per customer?
```sql
WITH per_customer AS (
  SELECT o.customer_id, SUM(oi.quantity * oi.unit_price) AS revenue
  FROM orders o JOIN order_items oi ON oi.order_id = o.order_id
  GROUP BY o.customer_id
)
SELECT AVG(revenue) FROM per_customer;
```
Why: first total per customer, then the average of those totals.

## Example
Q: Which customers spent more than the average customer?
```sql
WITH per_customer AS (
  SELECT o.customer_id, SUM(oi.quantity * oi.unit_price) AS revenue
  FROM orders o JOIN order_items oi ON oi.order_id = o.order_id
  GROUP BY o.customer_id
)
SELECT c.name, pc.revenue
FROM per_customer pc JOIN customers c ON c.customer_id = pc.customer_id
WHERE pc.revenue > (SELECT AVG(revenue) FROM per_customer);
```
Why: the CTE is reused twice, once for the rows and once for the average.

## Example
Q: Monthly revenue, then only the months above 2000.
```sql
WITH monthly AS (
  SELECT strftime('%Y-%m', o.order_date) AS month, SUM(oi.quantity * oi.unit_price) AS revenue
  FROM orders o JOIN order_items oi ON oi.order_id = o.order_id
  GROUP BY month
)
SELECT month, revenue FROM monthly WHERE revenue > 2000 ORDER BY month;
```
Why: aggregate in the CTE, filter the aggregated rows outside.

## Example
Q: For each department, the number of employees and the department's share of all employees.
```sql
WITH counts AS (SELECT dept_id, COUNT(*) AS n FROM employees GROUP BY dept_id)
SELECT d.name, c.n, 100.0 * c.n / (SELECT SUM(n) FROM counts) AS pct
FROM counts c JOIN departments d ON d.dept_id = c.dept_id;
```
Why: the grand total comes from the same CTE.

## Example
Q: Customers whose first order was in 2024-01.
```sql
WITH first_order AS (SELECT customer_id, MIN(order_date) AS first_date FROM orders GROUP BY customer_id)
SELECT c.name, f.first_date
FROM first_order f JOIN customers c ON c.customer_id = f.customer_id
WHERE f.first_date >= '2024-01-01' AND f.first_date < '2024-02-01';
```
Why: find each customer's first order, then filter on it.
