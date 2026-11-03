# Dates and times
id: dates
cues: year, month, day, date, when, since, before, after, between, last, during, in 2024, per month, monthly, weekly, age, how long, days, years, recent, quarter, youngest, oldest, born, birthday
use: Filter with date ranges; group by a truncated date; compute differences. SQLite: strftime('%Y', d), strftime('%Y-%m', d), julianday(a) - julianday(b). Postgres: EXTRACT(YEAR FROM d), date_trunc('month', d), a - b. MySQL: YEAR(d), DATE_FORMAT(d, '%Y-%m'), DATEDIFF(a, b). Oracle: EXTRACT(YEAR FROM d), TO_CHAR(d, 'YYYY-MM'), TRUNC(d, 'MM'), a - b in days. SQL Server: YEAR(d), FORMAT(d, 'yyyy-MM'), DATEDIFF(day, b, a).
avoid: Comparing a date column to a bare year number; using the wrong dialect's functions; "youngest" means the latest birth date (MAX), not the smallest.
manual: date and time functions strftime date julianday extract date_trunc interval

## Example
Q: How many orders were placed in 2024?
```sql
SELECT COUNT(*) FROM orders WHERE strftime('%Y', order_date) = '2024';
```
Why: strftime returns text, so compare with '2024' (in Postgres: EXTRACT(YEAR FROM order_date) = 2024).

## Example
Q: Number of orders per month.
```sql
SELECT strftime('%Y-%m', order_date) AS month, COUNT(*) AS orders FROM orders GROUP BY month ORDER BY month;
```
Why: group by the year-month string (Postgres: date_trunc('month', order_date)).

## Example
Q: Who is the most recently hired employee?
```sql
SELECT name, hire_date FROM employees ORDER BY hire_date DESC LIMIT 1;
```
Why: most recent = largest date. In the same way, the youngest person has the largest birth date.

## Example
Q: How many years has each employee worked here, as of 2025-01-01?
```sql
SELECT name, CAST((julianday('2025-01-01') - julianday(hire_date)) / 365.25 AS INTEGER) AS years
FROM employees;
```
Why: the difference in days divided by 365.25, cut to whole years.

## Example
Q: Customers who signed up in the second half of 2023.
```sql
SELECT name, signup_date FROM customers WHERE signup_date >= '2023-07-01' AND signup_date < '2024-01-01';
```
Why: a half-open range is correct for any dialect.
