# Text matching and text functions
id: strings
cues: contains, containing, starts with, start with, ends with, end with, like, includes the word, case insensitive, first letter, length, characters, concatenate, combined with, full name, domain, substring, letter
use: LIKE with % (any text) and _ (one character) for patterns; LOWER() for case-insensitive matching; || to join text (CONCAT in MySQL); LENGTH and SUBSTR to cut text.
avoid: = when the question says "contains"; LIKE when an exact value is known (slower, can over-match); assuming the stored case and spacing (check the values).

## Example
Q: Products whose name contains "Cable".
```sql
SELECT name FROM products WHERE name LIKE '%Cable%';
```
Why: % on both sides means "anywhere in the text".

## Example
Q: Customers whose name starts with a C, ignoring case.
```sql
SELECT name FROM customers WHERE LOWER(name) LIKE 'c%';
```
Why: LOWER makes the match case-insensitive in every dialect (Postgres also has ILIKE).

## Example
Q: Show each employee as "name (city)".
```sql
SELECT name || ' (' || city || ')' AS label FROM employees;
```
Why: || concatenates in SQLite and Postgres; MySQL uses CONCAT(name, ' (', city, ')').

## Example
Q: Email domains of customers.
```sql
SELECT DISTINCT SUBSTR(email, INSTR(email, '@') + 1) AS domain FROM customers WHERE email IS NOT NULL;
```
Why: everything after the @ (Postgres: split_part(email, '@', 2)).

## Example
Q: Products with names longer than 8 characters.
```sql
SELECT name, LENGTH(name) AS chars FROM products WHERE LENGTH(name) > 8;
```
Why: LENGTH counts characters.
