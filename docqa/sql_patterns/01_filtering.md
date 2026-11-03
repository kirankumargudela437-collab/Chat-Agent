# Filtering rows (WHERE)
id: filtering
cues: equal to, more than, less than, greater than, at least, at most, between, either, or, located in, and also, with status
use: Keep only the rows that meet conditions. Combine conditions with AND / OR, and put parentheses around every OR group that sits next to an AND.
avoid: Mixing AND and OR without parentheses; adding conditions the question did not ask for; using = with a value spelled differently from the stored one.
manual: WHERE clause boolean operators AND OR operator precedence IN BETWEEN comparison operators

## Example
Q: Which customers are in London?
```sql
SELECT name FROM customers WHERE city = 'London';
```
Why: one equality filter on the column that holds the city.

## Example
Q: List shipped orders from customers 1 or 2.
```sql
SELECT order_id FROM orders WHERE status = 'shipped' AND (customer_id = 1 OR customer_id = 2);
```
Why: the OR group is in parentheses; without them, every order of customer 2 would match, shipped or not.

## Example
Q: Which products cost between 20 and 100?
```sql
SELECT name, price FROM products WHERE price BETWEEN 20 AND 100;
```
Why: BETWEEN includes both ends (20 and 100).

## Example
Q: Show employees who work in Pune or Berlin.
```sql
SELECT name, city FROM employees WHERE city IN ('Pune', 'Berlin');
```
Why: IN is shorter and safer than a chain of ORs.

## Example
Q: Orders placed in the first quarter of 2024 that were not cancelled.
```sql
SELECT order_id, order_date FROM orders
WHERE order_date >= '2024-01-01' AND order_date < '2024-04-01' AND status <> 'cancelled';
```
Why: a half-open date range (>= start, < next start) never misses the last day.
