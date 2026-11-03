# Combining tables (INNER JOIN, bridge tables)
id: joins
cues: with their, along with, and their, together with, who bought, belongs to, sold by, related, linked, joined, through, bridge, many to many
use: JOIN tables on their key columns (a foreign key to the primary key it references). For many-to-many links, go through the bridge table that holds both keys.
avoid: Joining on columns that only share a name but mean different things; skipping the bridge table and joining two tables that have no direct key; joining tables the question does not need.
manual: JOIN ON inner join join clause foreign key table expressions

## Example
Q: Show each order with the name of its customer.
```sql
SELECT o.order_id, c.name FROM orders o JOIN customers c ON c.customer_id = o.customer_id;
```
Why: orders.customer_id references customers.customer_id.

## Example
Q: Which products were in order 5?
```sql
SELECT p.name FROM order_items oi JOIN products p ON p.product_id = oi.product_id WHERE oi.order_id = 5;
```
Why: order_items is the bridge between orders and products.

## Example
Q: Which customers bought a Laptop Pro?
```sql
SELECT DISTINCT c.name
FROM customers c
JOIN orders o ON o.customer_id = c.customer_id
JOIN order_items oi ON oi.order_id = o.order_id
JOIN products p ON p.product_id = oi.product_id
WHERE p.name = 'Laptop Pro';
```
Why: customer to product needs the whole chain customers - orders - order_items - products; DISTINCT because a customer may buy it twice.

## Example
Q: List employees with their department name.
```sql
SELECT e.name, d.name AS department FROM employees e JOIN departments d ON d.dept_id = e.dept_id;
```
Why: employees.dept_id references departments.dept_id.

## Example
Q: Which employees sold to customers in Pune?
```sql
SELECT DISTINCT e.name
FROM employees e
JOIN orders o ON o.employee_id = e.emp_id
JOIN customers c ON c.customer_id = o.customer_id
WHERE c.city = 'Pune';
```
Why: the order links the employee to the customer; the key names differ (emp_id vs employee_id), so follow the reference, not the name.
