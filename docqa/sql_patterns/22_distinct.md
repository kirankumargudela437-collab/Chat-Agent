# Unique values and duplicates (DISTINCT)
id: distinct
cues: different, distinct, unique, list of all, which cities, duplicates, repeated, more than once, appear twice, without repeats
use: SELECT DISTINCT returns each combination once. Find duplicates with GROUP BY ... HAVING COUNT(*) > 1.
avoid: DISTINCT to hide a wrong join that multiplies rows; DISTINCT on a column list that includes a unique id (does nothing).
manual: SELECT DISTINCT duplicate rows

## Example
Q: Which countries do our customers come from?
```sql
SELECT DISTINCT country FROM customers;
```
Why: each country once.

## Example
Q: Which cities have more than one customer?
```sql
SELECT city, COUNT(*) AS customers FROM customers GROUP BY city HAVING COUNT(*) > 1;
```
Why: duplicates are groups with more than one row.

## Example
Q: Which distinct products has customer 1 bought?
```sql
SELECT DISTINCT p.name FROM orders o JOIN order_items oi ON oi.order_id = o.order_id
JOIN products p ON p.product_id = oi.product_id WHERE o.customer_id = 1;
```
Why: a product bought in two orders is listed once.

## Example
Q: How many different products have been sold?
```sql
SELECT COUNT(DISTINCT product_id) FROM order_items;
```
Why: COUNT(DISTINCT) counts unique values.

## Example
Q: List each customer-and-employee pair that has done business, once.
```sql
SELECT DISTINCT customer_id, employee_id FROM orders ORDER BY customer_id, employee_id;
```
Why: DISTINCT applies to the whole combination.
