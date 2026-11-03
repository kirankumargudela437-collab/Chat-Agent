# Combining result lists (UNION, INTERSECT, EXCEPT)
id: set-ops
cues: both, either, and also, as well as, in both, in either, combined list, but not, only in, appear in both, union, intersect
use: UNION merges two lists (removing duplicates; UNION ALL keeps them), INTERSECT keeps values in both, EXCEPT keeps values in the first but not the second. Both sides need the same number of columns.
avoid: AND on one row when the question means "in both lists" (one row cannot be two things at once); different column counts on the two sides.

## Example
Q: Cities where we have customers or employees.
```sql
SELECT city FROM customers UNION SELECT city FROM employees;
```
Why: UNION gives each city once.

## Example
Q: Cities that have both customers and employees.
```sql
SELECT city FROM customers INTERSECT SELECT city FROM employees;
```
Why: INTERSECT keeps cities found in both lists.

## Example
Q: Customers who bought both a laptop (category 2) and software (category 5).
```sql
SELECT o.customer_id FROM orders o JOIN order_items oi ON oi.order_id = o.order_id
JOIN products p ON p.product_id = oi.product_id WHERE p.category_id = 2
INTERSECT
SELECT o.customer_id FROM orders o JOIN order_items oi ON oi.order_id = o.order_id
JOIN products p ON p.product_id = oi.product_id WHERE p.category_id = 5;
```
Why: one item row cannot be in two categories; "both" needs two lists.

## Example
Q: Customer ids that have orders but none shipped.
```sql
SELECT customer_id FROM orders
EXCEPT
SELECT customer_id FROM orders WHERE status = 'shipped';
```
Why: everyone with an order, minus everyone with a shipped order.

## Example
Q: One list of all people names: customers and employees, keeping duplicates.
```sql
SELECT name FROM customers UNION ALL SELECT name FROM employees;
```
Why: UNION ALL keeps every row and is faster.
