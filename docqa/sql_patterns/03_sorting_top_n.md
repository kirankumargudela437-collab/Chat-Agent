# Sorting and top N
id: top-n
cues: top, first, highest, lowest, most, least, largest, smallest, best, worst, latest, earliest, most recent, oldest, newest, order by, sorted, rank, youngest, oldest, top 3, top 5, fewest
use: ORDER BY the measure (DESC for highest / latest / most, ASC for lowest / earliest / least) and LIMIT to the number asked for.
avoid: Sorting the wrong direction ("most recent" is DESC on the date; "youngest" is DESC on birth date); returning all rows when the question asks for one; forgetting LIMIT.

## Example
Q: What are the three most expensive products?
```sql
SELECT name, price FROM products ORDER BY price DESC LIMIT 3;
```
Why: highest first, then keep three.

## Example
Q: Which order was placed most recently?
```sql
SELECT order_id, order_date FROM orders ORDER BY order_date DESC LIMIT 1;
```
Why: most recent = largest date = DESC.

## Example
Q: Who is the employee hired earliest?
```sql
SELECT name FROM employees ORDER BY hire_date ASC LIMIT 1;
```
Why: earliest = smallest date = ASC.

## Example
Q: List customers alphabetically.
```sql
SELECT name FROM customers ORDER BY name;
```
Why: ASC is the default.

## Example
Q: Show the 5 largest order lines by value, largest first, and break ties by order id.
```sql
SELECT order_id, product_id, quantity * unit_price AS line_value
FROM order_items ORDER BY line_value DESC, order_id ASC LIMIT 5;
```
Why: a second sort key makes ties deterministic.
