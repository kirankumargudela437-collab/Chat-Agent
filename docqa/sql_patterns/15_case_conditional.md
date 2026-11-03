# Conditional values and pivots (CASE WHEN)
id: case
cues: if, when, classify, label, category, bucket, as high or low, pivot, separate columns, count of each status, how many shipped and how many cancelled, flag
use: CASE WHEN ... THEN ... ELSE ... END builds a value from conditions. Inside SUM or COUNT it counts rows per condition, which turns rows into columns (a pivot).
avoid: Running one query per category when one CASE pivot answers all; forgetting ELSE (gives NULL).

## Example
Q: Label each product as cheap (under 50), mid (50 to 500) or expensive.
```sql
SELECT name, price,
       CASE WHEN price < 50 THEN 'cheap' WHEN price <= 500 THEN 'mid' ELSE 'expensive' END AS price_band
FROM products;
```
Why: conditions are checked in order; the first true one wins.

## Example
Q: For each customer, the number of shipped and cancelled orders.
```sql
SELECT customer_id,
       SUM(CASE WHEN status = 'shipped' THEN 1 ELSE 0 END) AS shipped,
       SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) AS cancelled
FROM orders GROUP BY customer_id;
```
Why: one pass, one column per status.

## Example
Q: How many employees are in Pune versus elsewhere?
```sql
SELECT CASE WHEN city = 'Pune' THEN 'Pune' ELSE 'Other' END AS location, COUNT(*) AS employees
FROM employees GROUP BY location;
```
Why: group by the CASE value.

## Example
Q: Revenue split into hardware and software columns.
```sql
SELECT SUM(CASE WHEN p.category_id = 5 THEN oi.quantity * oi.unit_price ELSE 0 END) AS software,
       SUM(CASE WHEN p.category_id <> 5 THEN oi.quantity * oi.unit_price ELSE 0 END) AS hardware
FROM order_items oi JOIN products p ON p.product_id = oi.product_id;
```
Why: conditional sums give one column per group.

## Example
Q: Flag employees with a salary above 100000.
```sql
SELECT name, salary, CASE WHEN salary > 100000 THEN 1 ELSE 0 END AS high_earner FROM employees;
```
Why: a 0/1 flag is easy to sum later.
