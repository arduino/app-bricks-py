# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Store and read data using SQLStore"
from arduino.app_bricks.dbstorage_sqlstore import SQLStore

db = SQLStore("example.db")

# Create a table
columns = {"id": "INTEGER PRIMARY KEY", "name": "TEXT", "age": "INTEGER"}
db.create_table("users", columns)

# Insert data
data = {"name": "Alice", "age": 30}
db.store("users", data)

# Read data
result = db.read("users")
print(result)

# Drop the table
db.drop_table("users")
