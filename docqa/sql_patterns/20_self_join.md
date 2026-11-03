# A table joined to itself (self-join)
id: self-join
cues: manager, reports to, boss, parent, same, pairs, two different, both in the same, colleague, another, each other, compared with other rows of the same table
use: Join a table to itself with two aliases when a row refers to another row of the same table (employee to manager) or when comparing pairs of rows.
avoid: Forgetting the second alias; pairs counted twice (use a.id < b.id).

## Example
Q: List each employee with their manager's name.
```sql
SELECT e.name AS employee, m.name AS manager FROM employees e LEFT JOIN employees m ON m.emp_id = e.manager_id;
```
Why: manager_id points to another row of employees; LEFT JOIN keeps the top boss.

## Example
Q: Which employees earn more than their manager?
```sql
SELECT e.name FROM employees e JOIN employees m ON m.emp_id = e.manager_id WHERE e.salary > m.salary;
```
Why: compare two rows of the same table.

## Example
Q: How many direct reports does each manager have?
```sql
SELECT m.name, COUNT(e.emp_id) AS reports FROM employees m JOIN employees e ON e.manager_id = m.emp_id
GROUP BY m.emp_id, m.name;
```
Why: group by the manager side of the self-join.

## Example
Q: Pairs of employees who live in the same city.
```sql
SELECT a.name, b.name, a.city FROM employees a JOIN employees b ON a.city = b.city AND a.emp_id < b.emp_id;
```
Why: a.emp_id < b.emp_id lists each pair once and never pairs someone with themselves.

## Example
Q: Categories with their parent category name.
```sql
SELECT c.name AS category, p.name AS parent FROM categories c LEFT JOIN categories p ON p.category_id = c.parent_id;
```
Why: parent_id references the same table.
