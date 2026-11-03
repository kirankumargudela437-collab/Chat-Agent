# Per-group results (GROUP BY and HAVING)
id: group-by
cues: each, per, for every, by, breakdown, group, grouped, in each, for each, at least, more than, having
use: GROUP BY the column(s) that define a group; aggregate the rest. Filter groups with HAVING, rows with WHERE.
avoid: Grouping by a name when names can repeat (group by the id, show the name); putting an aggregate in WHERE; selecting columns that are neither grouped nor aggregated.
manual: GROUP BY HAVING clause aggregate grouping

## Example
Q: How many orders does each customer have?
```sql
SELECT c.customer_id, c.name, COUNT(o.order_id) AS orders
FROM customers c JOIN orders o ON o.customer_id = c.customer_id
GROUP BY c.customer_id, c.name;
```
Why: group by the key (customer_id) so two customers with the same name stay apart.

## Example
Q: Number of employees per department.
```sql
SELECT d.name, COUNT(e.emp_id) AS employees
FROM departments d JOIN employees e ON e.dept_id = d.dept_id
GROUP BY d.dept_id, d.name;
```
Why: one row per department.

## Example
Q: Which customers placed more than two orders?
```sql
SELECT customer_id, COUNT(*) AS orders FROM orders GROUP BY customer_id HAVING COUNT(*) > 2;
```
Why: a condition on the count goes in HAVING, after grouping.

## Example
Q: Revenue by product category.
```sql
SELECT cat.name, SUM(oi.quantity * oi.unit_price) AS revenue
FROM order_items oi
JOIN products p ON p.product_id = oi.product_id
JOIN categories cat ON cat.category_id = p.category_id
GROUP BY cat.category_id, cat.name;
```
Why: join down to the category, then sum per category.

## Example
Q: Average salary per city, only for cities with at least two employees.
```sql
SELECT city, AVG(salary) AS avg_salary FROM employees GROUP BY city HAVING COUNT(*) >= 2;
```
Why: WHERE cannot see COUNT(*); HAVING can.
