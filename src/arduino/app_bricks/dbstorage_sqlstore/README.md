# Database - SQL Brick

This brick helps you manage SQLite databases easily by providing a simple interface for creating tables, inserting data, and handling database connections.

## Overview

The Database - SQL brick allows you to:

- Use a simple API for SQLite database operations
- Create tables with custom schemas
- Insert, update, and delete records
- Query data with flexible filters
- Manage connections automatically
- Handle errors for common database issues

It provides thread-safe database operations using SQLite as the underlying database engine. It supports named access to columns for easy data handling. The brick automatically manages database file storage in a dedicated directory structure and handles the connection lifecycle.

## Features

- Thread-safe database operations for multi-threaded applications
- Automatic table creation with type inference from data
- Flexible data querying with `WHERE`, `ORDER BY`, and `LIMIT` clauses
- Schema management with column addition and removal capabilities
- Raw SQL execution for advanced operations
- Named column access using `sqlite3.Row` factory

## Code example and usage

Instantiate a new class to open (or create a new database):

```python
from arduino.app_bricks.dbstorage_sqlstore import SQLStore

db = SQLStore("example.db")
# ... Do work

# Close database
db.stop()
```

To create a new table:

```python
# Create a table
columns = {
    "id": "INTEGER PRIMARY KEY",
    "name": "TEXT",
    "age": "INTEGER"
}
db.create_table("users", columns)
```

Insert new data in a table:

```python
# Insert data
data = {
    "name": "Alice",
    "age": 30
}
db.store("users", data)
```

Query, update and delete records:

```python
# Read rows (columns, condition, order_by and limit are optional)
rows = db.read("users", condition="age > 18", order_by="name ASC", limit=10)

# Update rows matching a condition
db.update("users", {"age": 31}, condition="name = 'Alice'")

# Delete rows matching a condition (without a condition ALL records are deleted)
db.delete("users", condition="name = 'Alice'")

# Raw SQL for advanced operations
result = db.execute_sql("SELECT COUNT(*) FROM users")
```

## Understanding Database Operations

The SQLStore automatically creates the database file inside the `/app/data` directory of your application. The brick supports automatic type inference when creating tables, mapping Python types (*int*, *float*, *str*, *bytes*) to corresponding SQLite column types (*INTEGER*, *REAL*, *TEXT*, *BLOB*).

The `store()` method can automatically create tables if they don't exist by analyzing the data types of the provided values. This makes it easy to get started without defining schemas upfront, while still allowing explicit table creation for more control over column definitions and constraints.
