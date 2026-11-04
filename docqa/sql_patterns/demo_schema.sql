-- Demo schema used by every pattern example. It is a small shop: the examples teach a
-- technique, and the agent adapts the technique to the real schema, never these names.
CREATE TABLE departments (dept_id INTEGER PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE employees (
  emp_id INTEGER PRIMARY KEY, name TEXT NOT NULL, dept_id INTEGER REFERENCES departments(dept_id),
  manager_id INTEGER REFERENCES employees(emp_id), hire_date TEXT, salary REAL, city TEXT
);
CREATE TABLE customers (
  customer_id INTEGER PRIMARY KEY, name TEXT NOT NULL, city TEXT, country TEXT,
  signup_date TEXT, email TEXT
);
CREATE TABLE categories (category_id INTEGER PRIMARY KEY, name TEXT NOT NULL, parent_id INTEGER REFERENCES categories(category_id));
CREATE TABLE products (
  product_id INTEGER PRIMARY KEY, name TEXT NOT NULL, category_id INTEGER REFERENCES categories(category_id),
  price REAL, discontinued INTEGER DEFAULT 0
);
CREATE TABLE orders (
  order_id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(customer_id),
  employee_id INTEGER REFERENCES employees(emp_id), order_date TEXT, status TEXT
);
-- bridge table: an order has many products, a product is in many orders
CREATE TABLE order_items (
  order_id INTEGER REFERENCES orders(order_id), product_id INTEGER REFERENCES products(product_id),
  quantity INTEGER, unit_price REAL, PRIMARY KEY (order_id, product_id)
);

INSERT INTO departments VALUES (1,'Sales'),(2,'Support'),(3,'Engineering'),(4,'Legal');
INSERT INTO employees VALUES
 (1,'Asha',3,NULL,'2019-03-01',150000,'Pune'),
 (2,'Ben',1,1,'2020-06-15',90000,'London'),
 (3,'Chen',1,2,'2021-01-10',70000,'London'),
 (4,'Divya',1,2,'2022-09-01',72000,'Pune'),
 (5,'Eli',2,1,'2020-11-20',60000,'Berlin'),
 (6,'Fatima',2,5,'2023-02-01',52000,'Berlin'),
 (7,'Gopal',3,1,'2018-07-07',120000,'Pune'),
 (8,'Hana',3,7,'2024-04-01',125000,'Tokyo');
INSERT INTO customers VALUES
 (1,'Acme Ltd','London','UK','2022-01-05','ops@acme.example'),
 (2,'Birla Traders','Pune','India','2022-03-12',NULL),
 (3,'Contoso','Berlin','Germany','2022-07-30','buy@contoso.example'),
 (4,'Delta Foods','Pune','India','2023-01-18','hi@delta.example'),
 (5,'Evergreen','Tokyo','Japan','2023-05-02',NULL),
 (6,'Fjord AS','Oslo','Norway','2023-08-21','post@fjord.example'),
 (7,'Gamma Corp','London','UK','2024-02-14','g@gamma.example');
INSERT INTO categories VALUES (1,'Hardware',NULL),(2,'Laptops',1),(3,'Accessories',1),(4,'Cables',3),(5,'Software',NULL);
INSERT INTO products VALUES
 (1,'Laptop Pro',2,1500,0),(2,'Laptop Air',2,1100,0),(3,'Mouse',3,25,0),(4,'Keyboard',3,45,0),
 (5,'USB-C Cable',4,12,0),(6,'HDMI Cable',4,15,1),(7,'Office Suite',5,200,0),(8,'Antivirus',5,60,0),(9,'Webcam',3,80,0);
INSERT INTO orders VALUES
 (1,1,2,'2024-01-10','shipped'),(2,2,3,'2024-01-22','shipped'),(3,1,2,'2024-02-03','shipped'),
 (4,3,4,'2024-02-17','cancelled'),(5,4,3,'2024-03-05','shipped'),(6,2,4,'2024-03-28','shipped'),
 (7,5,2,'2024-04-09','pending'),(8,1,3,'2024-04-30','shipped'),(9,3,4,'2024-05-15','shipped'),
 (10,4,2,'2024-06-01','shipped'),(11,6,3,'2024-06-20','cancelled'),(12,2,2,'2024-07-04','shipped');
INSERT INTO order_items VALUES
 (1,1,2,1500),(1,3,2,25),(2,2,1,1100),(2,7,1,200),(3,4,3,45),(3,5,5,12),
 (4,1,1,1450),(5,2,2,1050),(5,8,2,60),(6,3,10,22),(6,5,10,12),(7,7,3,200),
 (8,1,1,1500),(8,6,4,15),(9,8,5,60),(10,2,1,1100),(10,4,1,45),(11,3,1,25),(12,7,2,190),(12,8,1,60);
