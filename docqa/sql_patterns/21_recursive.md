# Hierarchies and chains (recursive CTE)
id: recursive
cues: all levels, hierarchy, tree, under, below, descendants, ancestors, chain of command, indirect, subcategories, everyone who reports to, path
use: WITH RECURSIVE starts from the top rows and repeatedly joins the next level, to walk a parent/child hierarchy of any depth.
avoid: A fixed number of self-joins (misses deeper levels); a recursion without a stopping join (loops).

## Example
Q: Everyone who reports to Asha, directly or indirectly.
```sql
WITH RECURSIVE team AS (
  SELECT emp_id, name FROM employees WHERE manager_id = (SELECT emp_id FROM employees WHERE name = 'Asha')
  UNION ALL
  SELECT e.emp_id, e.name FROM employees e JOIN team t ON e.manager_id = t.emp_id
)
SELECT name FROM team;
```
Why: start with direct reports, then keep adding their reports.

## Example
Q: All subcategories of Hardware, at any depth.
```sql
WITH RECURSIVE sub AS (
  SELECT category_id, name FROM categories WHERE parent_id = 1
  UNION ALL
  SELECT c.category_id, c.name FROM categories c JOIN sub s ON c.parent_id = s.category_id
)
SELECT name FROM sub;
```
Why: Cables sits under Accessories under Hardware, two levels down.

## Example
Q: Each employee's level in the hierarchy (the top boss is level 1).
```sql
WITH RECURSIVE lvl AS (
  SELECT emp_id, name, 1 AS level FROM employees WHERE manager_id IS NULL
  UNION ALL
  SELECT e.emp_id, e.name, l.level + 1 FROM employees e JOIN lvl l ON e.manager_id = l.emp_id
)
SELECT name, level FROM lvl ORDER BY level, name;
```
Why: carry a counter down the recursion.

## Example
Q: The management chain above Fatima.
```sql
WITH RECURSIVE chain AS (
  SELECT emp_id, name, manager_id FROM employees WHERE name = 'Fatima'
  UNION ALL
  SELECT m.emp_id, m.name, m.manager_id FROM employees m JOIN chain c ON m.emp_id = c.manager_id
)
SELECT name FROM chain WHERE name <> 'Fatima';
```
Why: walk upwards by joining on the manager.

## Example
Q: Revenue per top-level category, including all subcategories.
```sql
WITH RECURSIVE tree AS (
  SELECT category_id, category_id AS root_id FROM categories WHERE parent_id IS NULL
  UNION ALL
  SELECT c.category_id, t.root_id FROM categories c JOIN tree t ON c.parent_id = t.category_id
)
SELECT r.name, SUM(oi.quantity * oi.unit_price) AS revenue
FROM tree t JOIN categories r ON r.category_id = t.root_id
JOIN products p ON p.category_id = t.category_id
JOIN order_items oi ON oi.product_id = p.product_id
GROUP BY r.category_id, r.name;
```
Why: map every category to its root, then aggregate by root.
