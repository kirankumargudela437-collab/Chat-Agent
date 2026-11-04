# Totals, counts and averages (aggregation)
id: aggregation
cues: how many, number of, count, total, sum, average, mean, maximum, minimum, overall, in total, distinct, different
use: COUNT, SUM, AVG, MIN, MAX collapse rows into one value. COUNT(DISTINCT col) counts different values.
avoid: Counting rows after a join that duplicates them (count DISTINCT the key instead); averaging an average; returning extra columns next to an aggregate without GROUP BY.
manual: aggregate functions count sum avg min max count distinct

## Example
Q: How many customers are there?
```sql
SELECT COUNT(*) FROM customers;
```
Why: COUNT(*) counts rows.

## Example
Q: How many different countries do customers come from?
```sql
SELECT COUNT(DISTINCT country) FROM customers;
```
Why: DISTINCT counts each country once.

## Example
Q: What is the total revenue of all shipped orders?
```sql
SELECT SUM(oi.quantity * oi.unit_price)
FROM order_items oi JOIN orders o ON o.order_id = oi.order_id
WHERE o.status = 'shipped';
```
Why: revenue is quantity times price per line, summed after filtering the orders.

## Example
Q: What is the average salary of employees?
```sql
SELECT AVG(salary) FROM employees;
```
Why: AVG ignores NULL salaries.

## Example
Q: How many customers have placed at least one order?
```sql
SELECT COUNT(DISTINCT customer_id) FROM orders;
```
Why: a customer with three orders must count once, so DISTINCT the customer key.
