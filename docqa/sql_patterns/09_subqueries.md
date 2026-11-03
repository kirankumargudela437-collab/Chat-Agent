# Queries inside queries (subqueries)
id: subquery
cues: than the average, above average, more than average, compared to, same as, in the list of, whose, that have, maximum of, greater than all, at least one
use: A scalar subquery returns one value to compare with; IN (subquery) tests membership; a correlated subquery runs per outer row.
avoid: A subquery that returns several rows where one value is expected; a correlated subquery when a JOIN or GROUP BY is simpler.

## Example
Q: Which products cost more than the average product price?
```sql
SELECT name, price FROM products WHERE price > (SELECT AVG(price) FROM products);
```
Why: the inner query returns one number to compare against.

## Example
Q: Which employees earn the highest salary?
```sql
SELECT name, salary FROM employees WHERE salary = (SELECT MAX(salary) FROM employees);
```
Why: unlike ORDER BY ... LIMIT 1, this returns every employee tied for the top salary.

## Example
Q: Customers who ordered in June 2024.
```sql
SELECT name FROM customers
WHERE customer_id IN (SELECT customer_id FROM orders WHERE order_date >= '2024-06-01' AND order_date < '2024-07-01');
```
Why: IN checks membership in the list of ids.

## Example
Q: Employees who earn more than the average of their own department.
```sql
SELECT e.name, e.salary FROM employees e
WHERE e.salary > (SELECT AVG(e2.salary) FROM employees e2 WHERE e2.dept_id = e.dept_id);
```
Why: the subquery is correlated: it is recomputed for each employee's department.

## Example
Q: Products priced above every accessory.
```sql
SELECT name, price FROM products
WHERE price > (SELECT MAX(price) FROM products WHERE category_id = 3);
```
Why: "above every X" means above the maximum of X.
