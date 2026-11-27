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
- Transactions to save several changes together
- Optional faster writes for apps that save often
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
user_id = db.store("users", data)  # the id the database assigned to the new record
```

To save a value that may already be there, such as a setting stored under its name, tell `store()` what to do with a record that has the same key:

```python
db.create_table("settings", {"key": "TEXT PRIMARY KEY", "value": "TEXT"})
db.store("settings", {"key": "theme", "value": "dark"}, on_conflict="replace")   # insert, or replace the old value
db.store("settings", {"key": "theme", "value": "light"}, on_conflict="ignore")   # insert only if missing
```

Query, update and delete records:

```python
# Read rows (columns, condition, order_by and limit are optional)
rows = db.read("users", condition="age > 18", order_by="name ASC", limit=10)

# Update rows matching a condition: the result is the number of rows updated
updated = db.update("users", {"age": 31}, condition="name = 'Alice'")
if updated == 0:
    print("No user named Alice")

# Delete rows matching a condition (without a condition ALL records are deleted)
deleted = db.delete("users", condition="name = 'Alice'")

# Count or summarize rows: a column can be an SQL expression, named with AS
stats = db.read("users", columns=["COUNT(*) AS total", "MAX(age) AS oldest"])[0]
print(stats["total"], stats["oldest"])

# Raw SQL for advanced operations
result = db.execute_sql("SELECT COUNT(*) FROM users")
```

When a condition contains a value that comes from outside your code, such as text typed by a user or a field of a web request, write `?` in the condition and pass the value in `args`. The value is then never read as SQL, so it cannot change the query:

```python
name = "O'Brien"  # e.g. received from a web page
rows = db.read("users", condition="name = ?", args=(name,))
db.update("users", {"age": 42}, condition="name = ?", args=(name,))
db.delete("users", condition="name = ? AND age < ?", args=(name, 18))
```

To make several changes that belong together, such as renumbering a list of records, group them in a transaction. They are saved together when the block ends, or not at all if the block raises an exception, and saving them together is faster than one by one:

```python
with db.transaction():
    for position, user_id in enumerate(ordered_ids, start=1):
        db.update("users", {"position": position}, condition="id = ?", args=(user_id,))
```

## Faster writes for apps that save often

By default every change is flushed to disk before the call returns. It survives even a power loss right after it, but each change waits for the storage of the board, a few milliseconds or more. An app that saves at every user action, such as an editor that stores each stroke, can ask for faster writes:

```python
db = SQLStore("example.db", power_loss_safe=False)
```

What you give up: the last changes made before a **power loss or a system crash** can be lost. The database is never corrupted, and nothing is lost when only the app stops or crashes. Keep the default when a record must be on disk as soon as it is stored, for example a measurement you cannot take again.

With fast writes the brick keeps a second file next to the database, `example.db-wal`, where the changes are appended. You can go back to the default at any start of the app: the database is safe again, and keeps its data.

## Understanding Database Operations

The SQLStore automatically creates the database file inside the `/app/data` directory of your application. The brick supports automatic type inference when creating tables, mapping Python types (*int*, *float*, *str*, *bytes*) to corresponding SQLite column types (*INTEGER*, *REAL*, *TEXT*, *BLOB*).

The `store()` method can automatically create tables if they don't exist by analyzing the data types of the provided values. This makes it easy to get started without defining schemas upfront, while still allowing explicit table creation for more control over column definitions and constraints.
