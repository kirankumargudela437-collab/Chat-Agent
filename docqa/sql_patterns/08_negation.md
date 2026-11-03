# Rows that do NOT have something (anti-join)
id: negation
cues: not, never, no, without, none, did not, have not, haven't, didn't, except, other than, excluding, not in, missing from, has no, with no, no students, any, not have, do not, does not, never been
use: Find rows with no matching row elsewhere using NOT EXISTS, NOT IN (on a column without NULLs), EXCEPT, or LEFT JOIN ... WHERE right.key IS NULL.
avoid: Using != on a join ("customers whose order is not X" still returns customers who also have an X order); NOT IN over a subquery that can return NULL (returns nothing).

## Example
Q: Which customers have never placed an order?
```sql
SELECT name FROM customers c
WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id);
```
Why: NOT EXISTS is true when no order points at the customer.

## Example
Q: Which products were never sold?
```sql
SELECT name FROM products WHERE product_id NOT IN (SELECT product_id FROM order_items);
```
Why: NOT IN is safe here because order_items.product_id is never NULL.

## Example
Q: Which employees have not handled any order?
```sql
SELECT e.name FROM employees e LEFT JOIN orders o ON o.employee_id = e.emp_id WHERE o.order_id IS NULL;
```
Why: after a LEFT JOIN, a NULL on the right side means "no match".

## Example
Q: Which customers have never had an order cancelled?
```sql
SELECT name FROM customers
WHERE customer_id NOT IN (SELECT customer_id FROM orders WHERE status = 'cancelled');
```
Why: exclude anyone with a cancelled order; status <> 'cancelled' on a join would still return them through their other orders.

## Example
Q: Cities that have customers but no employees.
```sql
SELECT city FROM customers
EXCEPT
SELECT city FROM employees;
```
Why: EXCEPT removes every value found in the second list (and returns each city once).
