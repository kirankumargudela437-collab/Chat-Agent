# Keeping rows without a match (LEFT JOIN)
id: left-join
cues: including those with none, even if, all customers, every, zero, 0, no orders, with or without, including, count of each including
use: LEFT JOIN keeps every row of the left table, with NULLs where the right side has no match. Use it when rows with no match must still appear (for example with a count of 0).
avoid: Putting a condition on the right table in WHERE (turns the LEFT JOIN back into an inner join; move it into ON); COUNT(*) instead of COUNT(right.key) (counts 1 for no match).

## Example
Q: How many orders does every customer have, including customers with none?
```sql
SELECT c.name, COUNT(o.order_id) AS orders
FROM customers c LEFT JOIN orders o ON o.customer_id = c.customer_id
GROUP BY c.customer_id, c.name;
```
Why: COUNT(o.order_id) gives 0 for a customer without orders.

## Example
Q: List all departments and their employees, keeping empty departments.
```sql
SELECT d.name AS department, e.name AS employee
FROM departments d LEFT JOIN employees e ON e.dept_id = d.dept_id;
```
Why: Legal has no employees and still appears, with NULL.

## Example
Q: Shipped-order count per customer, keeping customers with no shipped orders.
```sql
SELECT c.name, COUNT(o.order_id) AS shipped_orders
FROM customers c LEFT JOIN orders o ON o.customer_id = c.customer_id AND o.status = 'shipped'
GROUP BY c.customer_id, c.name;
```
Why: the status filter is in ON; in WHERE it would drop the customers with none.

## Example
Q: Units sold per product, 0 for products never sold.
```sql
SELECT p.name, COALESCE(SUM(oi.quantity), 0) AS units
FROM products p LEFT JOIN order_items oi ON oi.product_id = p.product_id
GROUP BY p.product_id, p.name;
```
Why: LEFT JOIN plus COALESCE for the products with no sales.

## Example
Q: Show every employee and the number of orders they handled.
```sql
SELECT e.name, COUNT(o.order_id) AS orders_handled
FROM employees e LEFT JOIN orders o ON o.employee_id = e.emp_id
GROUP BY e.emp_id, e.name;
```
Why: employees who handled no orders get 0.
