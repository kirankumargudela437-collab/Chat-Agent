# The row with the highest or lowest value, with its details
id: argmax
cues: the one with the highest, the most, the largest, the cheapest, the most expensive, that has the maximum, with the fewest, the biggest, the top, has the most, with the highest, with the lowest
use: When the question wants the NAME (or other details) of the row with the extreme value, ORDER BY the value LIMIT 1, or compare to MAX/MIN in a subquery when ties must all be returned. For "the group with the most X", aggregate first and then pick.
avoid: SELECT name, MAX(x) without GROUP BY (returns an arbitrary name in SQLite, an error elsewhere); returning the value when the question asks for the name.

## Example
Q: Which product is the cheapest?
```sql
SELECT name FROM products ORDER BY price ASC LIMIT 1;
```
Why: the question asks for the name, not the price.

## Example
Q: Which customer placed the most orders?
```sql
SELECT c.name FROM customers c JOIN orders o ON o.customer_id = c.customer_id
GROUP BY c.customer_id, c.name ORDER BY COUNT(*) DESC LIMIT 1;
```
Why: count per customer, then take the top group.

## Example
Q: Which department has the highest total salary?
```sql
SELECT d.name FROM departments d JOIN employees e ON e.dept_id = d.dept_id
GROUP BY d.dept_id, d.name ORDER BY SUM(e.salary) DESC LIMIT 1;
```
Why: aggregate inside ORDER BY, return only the name.

## Example
Q: Which products have the highest price (all of them if tied)?
```sql
SELECT name, price FROM products WHERE price = (SELECT MAX(price) FROM products);
```
Why: the subquery form returns every tied row.

## Example
Q: The employee with the lowest salary in the Sales department.
```sql
SELECT e.name FROM employees e JOIN departments d ON d.dept_id = e.dept_id
WHERE d.name = 'Sales' ORDER BY e.salary ASC LIMIT 1;
```
Why: filter to the group first, then pick the extreme.
