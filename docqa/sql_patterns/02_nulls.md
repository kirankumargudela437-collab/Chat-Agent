# Missing values (NULL)
id: nulls
cues: missing, without, no email, unknown, not provided, empty, null, blank, has no, default to, treat as zero
use: NULL means "unknown". Test it with IS NULL / IS NOT NULL, replace it with COALESCE, and avoid division by zero with NULLIF.
avoid: Writing = NULL or <> NULL (always unknown, returns nothing); forgetting that COUNT(column) skips NULLs while COUNT(*) does not.

## Example
Q: Which customers have no email address?
```sql
SELECT name FROM customers WHERE email IS NULL;
```
Why: = NULL never matches; IS NULL does.

## Example
Q: How many customers gave an email address?
```sql
SELECT COUNT(email) FROM customers;
```
Why: COUNT(column) counts only non-NULL values; COUNT(*) would count every customer.

## Example
Q: Show every employee with their manager id, writing 'none' when there is no manager.
```sql
SELECT name, COALESCE(CAST(manager_id AS TEXT), 'none') AS manager FROM employees;
```
Why: COALESCE returns the first non-NULL argument.

## Example
Q: Revenue per order, counting orders without items as 0.
```sql
SELECT o.order_id, COALESCE(SUM(oi.quantity * oi.unit_price), 0) AS revenue
FROM orders o LEFT JOIN order_items oi ON oi.order_id = o.order_id
GROUP BY o.order_id;
```
Why: SUM over no rows is NULL; COALESCE turns it into 0.

## Example
Q: Average item price per order line, safe when quantity is 0.
```sql
SELECT order_id, product_id, (quantity * unit_price) / NULLIF(quantity, 0) AS price_each FROM order_items;
```
Why: NULLIF(x, 0) gives NULL instead of a division-by-zero error.
