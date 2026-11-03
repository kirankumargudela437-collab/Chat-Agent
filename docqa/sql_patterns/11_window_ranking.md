# Ranking within groups (ROW_NUMBER, RANK, DENSE_RANK)
id: window-rank
cues: rank, ranking, top per, best in each, highest in each, top 2 in every, first per, nth, second highest, position, latest per, most recent for each, second, third, top ... each, top ... per, top ... every, highest ... each, highest ... per, latest ... each, best ... each, most ... each, lowest ... each
use: ROW_NUMBER() / RANK() / DENSE_RANK() OVER (PARTITION BY group ORDER BY measure) numbers rows inside each group; filter the number in an outer query.
avoid: Filtering a window function in WHERE of the same query (wrap it in a CTE first); ROW_NUMBER when ties should share a place (use RANK or DENSE_RANK).

## Example
Q: The highest-paid employee in each department.
```sql
WITH ranked AS (
  SELECT name, dept_id, salary, ROW_NUMBER() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rn
  FROM employees
)
SELECT name, dept_id, salary FROM ranked WHERE rn = 1;
```
Why: number per department by salary, keep number 1.

## Example
Q: Rank products by price, ties sharing the same rank.
```sql
SELECT name, price, RANK() OVER (ORDER BY price DESC) AS price_rank FROM products;
```
Why: RANK gives equal prices the same rank and skips the next numbers.

## Example
Q: The most recent order of every customer.
```sql
WITH ranked AS (
  SELECT customer_id, order_id, order_date,
         ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) AS rn
  FROM orders
)
SELECT customer_id, order_id, order_date FROM ranked WHERE rn = 1;
```
Why: "latest per group" is a ranking by date DESC per group.

## Example
Q: What is the second highest salary?
```sql
WITH ranked AS (SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS r FROM employees)
SELECT DISTINCT salary FROM ranked WHERE r = 2;
```
Why: DENSE_RANK makes "second" mean the second distinct value.

## Example
Q: Top two products by units sold in each category.
```sql
WITH sold AS (
  SELECT p.category_id, p.name, SUM(oi.quantity) AS units
  FROM products p JOIN order_items oi ON oi.product_id = p.product_id
  GROUP BY p.product_id, p.category_id, p.name
), ranked AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY category_id ORDER BY units DESC) AS rn FROM sold
)
SELECT category_id, name, units FROM ranked WHERE rn <= 2;
```
Why: aggregate first, then rank the aggregates.
