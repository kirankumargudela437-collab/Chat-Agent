# Percentages and ratios
id: percentage
cues: percent, percentage, %, ratio, rate, proportion, share, fraction, how much of, out of, conversion, per cent
use: Percentage = 100.0 * part / whole. Use 100.0 (or CAST ... AS REAL) so integer division does not round to 0, and NULLIF(whole, 0) against division by zero. Count the part with SUM(CASE ...) over the same rows as the whole.
avoid: Integer division (1 / 4 = 0 in SQLite and Postgres); a denominator that counts different rows from the numerator; reading "ratio of A to B" upside down (it is A / B).

## Example
Q: What percentage of orders were cancelled?
```sql
SELECT 100.0 * SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) / COUNT(*) AS pct_cancelled FROM orders;
```
Why: part and whole come from the same rows; 100.0 forces decimal division.

## Example
Q: Percentage of customers who have an email address.
```sql
SELECT 100.0 * COUNT(email) / COUNT(*) AS pct_with_email FROM customers;
```
Why: COUNT(email) skips NULLs, COUNT(*) does not.

## Example
Q: Ratio of shipped to cancelled orders.
```sql
SELECT CAST(SUM(CASE WHEN status = 'shipped' THEN 1 ELSE 0 END) AS REAL)
       / NULLIF(SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END), 0) AS shipped_to_cancelled
FROM orders;
```
Why: "A to B" is A divided by B; NULLIF guards an empty denominator.

## Example
Q: Share of revenue from each country, in percent.
```sql
SELECT c.country, 100.0 * SUM(oi.quantity * oi.unit_price)
       / (SELECT SUM(quantity * unit_price) FROM order_items) AS pct
FROM customers c JOIN orders o ON o.customer_id = c.customer_id
JOIN order_items oi ON oi.order_id = o.order_id
GROUP BY c.country;
```
Why: the whole is the total over all items; the part is per country.

## Example
Q: Average discount in percent, comparing the sold price with the list price.
```sql
SELECT AVG(100.0 * (p.price - oi.unit_price) / p.price) AS avg_discount_pct
FROM order_items oi JOIN products p ON p.product_id = oi.product_id;
```
Why: the discount per line, then averaged.
