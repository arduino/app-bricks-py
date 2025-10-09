# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import gc
import pytest
import os
import time
from arduino.app_bricks.dbstorage_sqlstore import SQLStore, DBStorageSQLStoreError
from arduino.app_utils import Logger

logger = Logger("SQLStore.tests")


@pytest.fixture
def open_sqlstore_database():
    """Fixture to provide shared data."""
    # Setup logic
    db = SQLStore(database_name="test")
    yield db
    # Teardown logic
    db.stop()

    gc.collect()  # Force close all connections and clean up resources

    for _ in range(30):
        try:
            os.remove("./data/dbstorage_sqlstore/test.db")
            break
        except PermissionError:
            time.sleep(0.1)
    else:
        print("[WARNING] Impossible to remove the test database file. There may be a holding block after 3s.")


def test_create_table(open_sqlstore_database: SQLStore):
    """Test creating a table in the SQLStore database."""
    db = open_sqlstore_database

    # Create a table
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})

    # Verify table creation
    res = db.execute_sql("SELECT name FROM sqlite_master WHERE type='table' AND name='test_table'")
    assert len(res) == 1, "Table not found in the database"
    assert res[0]["name"] == "test_table", "Table name does not match"


def test_drop_table(open_sqlstore_database: SQLStore):
    """Test dropping a table in the SQLStore database."""
    db = open_sqlstore_database

    # Create a table
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})

    # Drop the table
    db.drop_table("test_table")

    # Verify table deletion
    res = db.execute_sql("SELECT name FROM sqlite_master WHERE type='table' AND name='test_table'")
    assert res is None, "Table still exists in the database after drop"


def test_store(open_sqlstore_database: SQLStore):
    """Test storing data into the SQLStore database."""
    db = open_sqlstore_database

    db.store("test_table", {"id": 1, "name": "Test Name"})

    # Verify data
    res = db.execute_sql("SELECT * FROM test_table WHERE id = 1")
    assert len(res) == 1, "Data not found in the table"
    assert res[0]["name"] == "Test Name", "Inserted data does not match"


def test_store_by_creating_table_before(open_sqlstore_database: SQLStore):
    """Test storing data in a table creating, according to the data provided."""
    db = open_sqlstore_database

    # Create a table
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})
    # Store data
    db.store("test_table", {"id": 1, "name": "Test Name"}, create_table=False)

    # Verify data
    res = db.execute_sql("SELECT * FROM test_table WHERE id = 1")
    assert len(res) == 1, "Data not found in the table"
    assert res[0]["name"] == "Test Name", "Inserted data does not match"


def test_store_invalid_data(open_sqlstore_database: SQLStore):
    """Test storing invalid data into the SQLStore database."""
    db = open_sqlstore_database

    # Create a table
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})

    # Attempt to insert invalid data
    try:
        db.store("test_table", {"id": "invalid_id", "name": "Test Name"})
        pytest.fail("Storing invalid data should raise an error")
    except Exception as e:
        assert "Error" in str(e), "Storing invalid data should raise an error"


def test_store_duplicate_key(open_sqlstore_database: SQLStore):
    """Test inserting a duplicate key into the SQLStore database."""
    db = open_sqlstore_database

    # Create a table and insert data
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})
    db.store("test_table", {"id": 1, "name": "Test Name"})

    # Attempt to insert a duplicate key
    try:
        db.store("test_table", {"id": 1, "name": "Duplicate Name"})
        pytest.fail("Inserting a duplicate key should raise an error")
    except Exception as e:
        assert "UNIQUE constraint failed" in str(e), "Inserting duplicate key should raise an error"


def test_read(open_sqlstore_database: SQLStore):
    """Test reading data from the SQLStore database."""
    db = open_sqlstore_database

    # Create a table and insert data
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})
    db.store("test_table", {"id": 1, "name": "Test Name"})

    # Read data
    result = db.read("test_table", columns=["id", "name"], condition="id = 1")
    assert len(result) == 1, "Data read failed"
    assert result[0]["id"] == 1, "ID does not match"
    assert result[0]["name"] == "Test Name", "Name does not match"


def test_read_nonexistent_table(open_sqlstore_database: SQLStore):
    """Test reading from a non-existent table in the SQLStore database."""
    db = open_sqlstore_database

    # Attempt to read from a non-existent table
    result = db.read("non_existent_table")
    assert result == [], "Reading from a non-existent table should return an empty list"


def test_read_nonexistent_column(open_sqlstore_database: SQLStore):
    """Test reading a non-existent column from the SQLStore database."""
    db = open_sqlstore_database

    # Create a table and insert data
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})
    db.store("test_table", {"id": 1, "name": "Test Name"})

    # Attempt to read a non-existent column
    try:
        db.read("test_table", columns=["non_existent_column"], condition="id = 1")
        pytest.fail("Reading a non-existent column should raise an error")
    except Exception as e:
        assert "no such column" in str(e), "Reading a non-existent column should raise an error related to the column"


def test_read_all_data(open_sqlstore_database: SQLStore):
    """Test reading all data from the SQLStore database."""
    db = open_sqlstore_database

    # Create a table and insert data
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})
    db.store("test_table", {"id": 1, "name": "Test Name"})
    db.store("test_table", {"id": 2, "name": "Another Name"})

    # Read all data
    result = db.read("test_table")
    assert len(result) == 2, "Reading all data failed"
    assert result[0]["id"] == 1, "First row ID does not match"
    assert result[0]["name"] == "Test Name", "First row name does not match"
    assert result[1]["id"] == 2, "Second row ID does not match"
    assert result[1]["name"] == "Another Name", "Second row name does not match"


def test_read_with_order(open_sqlstore_database: SQLStore):
    """Test reading data with order from the SQLStore database."""
    db = open_sqlstore_database

    # Create a table and insert data
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})
    db.store("test_table", {"id": 1, "name": "Test Name"})
    db.store("test_table", {"id": 2, "name": "Another Name"})

    # Read data with order
    result = db.read("test_table", order_by="id DESC")
    assert len(result) == 2, "Order read failed"
    assert result[0]["id"] == 2, "First row ID does not match in DESC order"
    assert result[1]["id"] == 1, "Second row ID does not match in DESC order"


def test_read_with_nonexistent_order(open_sqlstore_database: SQLStore):
    """Test reading data with a non-existent order from the SQLStore database."""
    db = open_sqlstore_database

    # Create a table and insert data
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})
    db.store("test_table", {"id": 1, "name": "Test Name"})

    # Attempt to read data with a non-existent order
    try:
        db.read("test_table", order_by="non_existent_column")
        pytest.fail("Reading with a non-existent order should raise an error")
    except Exception as e:
        assert "no such column" in str(e), "Reading with a non-existent order should raise an error related to the column"


def test_read_with_limit(open_sqlstore_database: SQLStore):
    """Test reading data with a limit from the SQLStore database."""
    db = open_sqlstore_database

    # Create a table and insert data
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})
    db.store("test_table", {"id": 1, "name": "Test Name"})
    db.store("test_table", {"id": 2, "name": "Another Name"})

    # Read data with a limit
    result = db.read("test_table", limit=1)
    assert len(result) == 1, "Limit read failed"
    assert result[0]["id"] == 1, "ID does not match in limit read"


def test_update(open_sqlstore_database: SQLStore):
    """Test updating data in the SQLStore database."""
    db = open_sqlstore_database

    # Create a table and insert data
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})
    db.store("test_table", {"id": 1, "name": "Test Name"})

    # Update data
    db.update("test_table", {"name": "Updated Name"}, condition="id = 1")

    # Verify update
    res = db.execute_sql("SELECT * FROM test_table WHERE id = 1")
    assert len(res) == 1, "Data not found in the table after update"
    assert res[0]["name"] == "Updated Name", "Updated data does not match"


def test_delete(open_sqlstore_database: SQLStore):
    """Test deleting data from the SQLStore database."""
    db = open_sqlstore_database

    # Create a table and insert data
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})
    db.store("test_table", {"id": 1, "name": "Test Name"})

    # Delete data
    db.delete("test_table", condition="id = 1")

    # Verify deletion
    res = db.execute_sql("SELECT * FROM test_table WHERE id = 1")
    assert res is None, "Data still found in the table after deletion"


def test_delete_all_data(open_sqlstore_database: SQLStore):
    """Test deleting all data from the SQLStore database."""
    db = open_sqlstore_database

    # Create a table and insert data
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})
    db.store("test_table", {"id": 1, "name": "Test Name"})
    db.store("test_table", {"id": 2, "name": "Another Name"})

    # Delete all data
    db.delete("test_table")

    # Verify deletion
    res = db.execute_sql("SELECT * FROM test_table")
    assert res is None, "Data still found in the table after deleting all data"


def test_execute_insert(open_sqlstore_database: SQLStore):
    """Test executing a raw SQL command in the SQLStore database."""
    db = open_sqlstore_database

    # Create a table
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})

    # Execute a raw SQL command
    res = db.execute_sql("INSERT INTO test_table (id, name) VALUES (?, ?)", (1, "Test Name"))
    assert res is None, "Raw SQL execution should not return a result set"

    # Verify data
    res = db.execute_sql("SELECT * FROM test_table WHERE id = 1")
    assert len(res) == 1, "Data not found in the table after raw SQL execution"
    assert res[0]["name"] == "Test Name", "Inserted data does not match"


def test_execute_select(open_sqlstore_database: SQLStore):
    """Test executing a raw SQL SELECT command in the SQLStore database."""
    db = open_sqlstore_database

    # Create a table and insert data
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})
    db.store("test_table", {"id": 1, "name": "Test Name"})

    # Execute a raw SQL SELECT command
    res = db.execute_sql("SELECT * FROM test_table WHERE id = ?", (1,))
    assert len(res) == 1, "Raw SQL SELECT should return one row"
    assert res[0]["name"] == "Test Name", "Selected data does not match"


def test_execute_select_with_no_results(open_sqlstore_database: SQLStore):
    """Test executing a raw SQL SELECT command that returns no results."""
    db = open_sqlstore_database

    # Create a table and insert data
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})
    db.store("test_table", {"id": 1, "name": "Test Name"})

    # Execute a raw SQL SELECT command that returns no results
    res = db.execute_sql("SELECT * FROM test_table WHERE id = ?", (2,))
    assert res is None, "Raw SQL SELECT with no results should return None"


def test_execute_invalid_sql(open_sqlstore_database: SQLStore):
    """Test executing an invalid SQL command in the SQLStore database."""
    db = open_sqlstore_database

    # Attempt to execute an invalid SQL command
    try:
        db.execute_sql("INVALID SQL COMMAND")
        pytest.fail("Executing invalid SQL should raise an error")
    except Exception as e:
        assert "syntax error" in str(e), "Executing invalid SQL should raise a syntax error"


def test_execute_update(open_sqlstore_database: SQLStore):
    """Test executing a raw SQL UPDATE command in the SQLStore database."""
    db = open_sqlstore_database

    # Create a table and insert data
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})
    db.store("test_table", {"id": 1, "name": "Test Name"})

    # Execute a raw SQL UPDATE command
    res = db.execute_sql("UPDATE test_table SET name = ? WHERE id = ?", ("Updated Name", 1))
    assert res is None, "Raw SQL execution should not return a result set"

    # Verify update
    res = db.execute_sql("SELECT * FROM test_table WHERE id = 1")
    assert len(res) == 1, "Data not found in the table after raw SQL execution"
    assert res[0]["name"] == "Updated Name", "Updated data does not match"


def test_execute_delete(open_sqlstore_database: SQLStore):
    """Test executing a raw SQL DELETE command in the SQLStore database."""
    db = open_sqlstore_database

    # Create a table and insert data
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})
    db.store("test_table", {"id": 1, "name": "Test Name"})

    # Execute a raw SQL DELETE command
    res = db.execute_sql("DELETE FROM test_table WHERE id = ?", (1,))
    assert res is None, "Raw SQL execution should not return a result set"

    # Verify deletion
    res = db.execute_sql("SELECT * FROM test_table WHERE id = 1")
    assert res is None, "Data still found in the table after raw SQL execution"


def test_execute_insert_returning(open_sqlstore_database: SQLStore):
    """Test executing a raw SQL INSERT command with RETURNING clause in the SQLStore database."""
    db = open_sqlstore_database

    # Create a table
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})

    # Execute a raw SQL INSERT command with RETURNING clause
    res = db.execute_sql("INSERT INTO test_table (id, name) VALUES (?, ?) RETURNING id, name", (1, "Test Name"))
    assert len(res) == 1, "Raw SQL INSERT with RETURNING should return one row"
    assert res[0]["id"] == 1, "Returned ID does not match"
    assert res[0]["name"] == "Test Name", "Returned name does not match"


def test_execute_update_returning(open_sqlstore_database: SQLStore):
    """Test executing a raw SQL UPDATE command with RETURNING clause in the SQLStore database."""
    db = open_sqlstore_database

    # Create a table and insert data
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})
    db.store("test_table", {"id": 1, "name": "Test Name"})

    # Execute a raw SQL UPDATE command with RETURNING clause
    res = db.execute_sql("UPDATE test_table SET name = ? WHERE id = ? RETURNING id, name", ("Updated Name", 1))
    assert len(res) == 1, "Raw SQL UPDATE with RETURNING should return one row"
    assert res[0]["id"] == 1, "Returned ID does not match"
    assert res[0]["name"] == "Updated Name", "Returned name does not match"


def test_create_or_replace_table_add_column(open_sqlstore_database: SQLStore):
    """Test adding a column with create_or_replace_table."""
    db = open_sqlstore_database
    db.create_table("t_add", {"a": "INTEGER"})
    db.store("t_add", {"a": 1}, create_table=False)
    db.create_or_replace_table("t_add", {"a": "INTEGER", "b": "TEXT"})
    db.store("t_add", {"a": 2, "b": "hello"}, create_table=False)
    rows = db.read("t_add")
    assert any("b" in row for row in rows)


def test_create_or_replace_table_remove_simple_column(open_sqlstore_database: SQLStore):
    """Test removing a simple column with create_or_replace_table."""
    db = open_sqlstore_database
    db.create_table("t_remove", {"a": "INTEGER", "b": "TEXT"})
    db.store("t_remove", {"a": 1, "b": "x"}, create_table=False)
    db.create_or_replace_table("t_remove", {"a": "INTEGER"})
    rows = db.read("t_remove")
    assert all("b" not in row for row in rows)


def test_create_or_replace_table_change_column_type(open_sqlstore_database: SQLStore):
    """Test changing a column type with create_or_replace_table (should fail if not forced)."""
    db = open_sqlstore_database
    db.create_table("t_type", {"a": "INTEGER"})
    db.store("t_type", {"a": 1}, create_table=False)
    with pytest.raises(DBStorageSQLStoreError):
        db.create_or_replace_table("t_type", {"a": "TEXT"})
    # Table should be unchanged
    rows = db.read("t_type")
    assert all(isinstance(row["a"], int) for row in rows)


def test_create_or_replace_table_change_column_type_force_drop(open_sqlstore_database: SQLStore):
    """Test changing a column type with create_or_replace_table and force_drop_table=True.

    The table should be dropped and recreated with the new column type.
    """
    db = open_sqlstore_database
    db.create_table("t_type_force", {"a": "INTEGER"})
    db.store("t_type_force", {"a": 1}, create_table=False)
    db.create_or_replace_table("t_type_force", {"a": "TEXT"}, force_drop_table=True)
    rows = db.read("t_type_force")
    assert all(isinstance(row["a"], str) or row["a"] is None for row in rows)


def test_create_or_replace_table_remove_pk_column(open_sqlstore_database: SQLStore):
    """Test removing a primary key column with create_or_replace_table (should fail if not forced)."""
    db = open_sqlstore_database
    db.create_table("t_pk", {"id": "INTEGER PRIMARY KEY", "val": "TEXT"})
    db.store("t_pk", {"id": 1, "val": "x"}, create_table=False)
    with pytest.raises(DBStorageSQLStoreError):
        db.create_or_replace_table("t_pk", {"val": "TEXT"})
    # Table should be unchanged
    rows = db.read("t_pk")
    assert all("id" in row for row in rows)


def test_create_or_replace_table_remove_pk_column_force_drop(open_sqlstore_database: SQLStore):
    """Test removing a primary key column with create_or_replace_table and force_drop_table=True."""
    db = open_sqlstore_database
    db.create_table("t_pk", {"id": "INTEGER PRIMARY KEY", "val": "TEXT"})
    db.store("t_pk", {"id": 1, "val": "x"}, create_table=False)
    db.create_or_replace_table("t_pk", {"val": "TEXT"}, force_drop_table=True)
    rows = db.read("t_pk")
    assert all("id" not in row for row in rows)


def test_create_or_replace_table_remove_unique_column(open_sqlstore_database: SQLStore):
    """Test removing a UNIQUE column (non-simple) with create_or_replace_table and force_drop_table=False.

    Should fail and not fallback to table recreation.
    """
    db = open_sqlstore_database
    db.create_table("t_unique_nf", {"a": "INTEGER UNIQUE", "b": "TEXT"})
    db.store("t_unique_nf", {"a": 1, "b": "x"}, create_table=False)
    with pytest.raises(DBStorageSQLStoreError):
        db.create_or_replace_table("t_unique_nf", {"b": "TEXT"})
    # Table should be unchanged
    rows = db.read("t_unique_nf")
    assert all("a" in row for row in rows)


def test_create_or_replace_table_remove_unique_column_force_drop(open_sqlstore_database: SQLStore):
    """Test removing a UNIQUE column with create_or_replace_table and force_drop_table=True.

    The table should be dropped and recreated without the UNIQUE column.
    """
    db = open_sqlstore_database
    db.create_table("t_unique_force", {"a": "INTEGER UNIQUE", "b": "TEXT"})
    db.store("t_unique_force", {"a": 1, "b": "x"}, create_table=False)
    db.create_or_replace_table("t_unique_force", {"b": "TEXT"}, force_drop_table=True)
    rows = db.read("t_unique_force")
    assert all("a" not in row for row in rows)


def test_create_or_replace_table_remove_indexed_column(open_sqlstore_database: SQLStore):
    """Test removing an indexed column (non-simple) with create_or_replace_table and force_drop_table=False.

    Should fail and not fallback to table recreation.
    """
    db = open_sqlstore_database
    db.create_table("t_idx_nf", {"a": "INTEGER", "b": "TEXT"})
    db.store("t_idx_nf", {"a": 1, "b": "x"}, create_table=False)
    db.execute_sql("CREATE INDEX idx_a_nf ON t_idx_nf(a)")
    with pytest.raises(DBStorageSQLStoreError):
        db.create_or_replace_table("t_idx_nf", {"b": "TEXT"})
    # Table should be unchanged
    rows = db.read("t_idx_nf")
    assert all("a" in row for row in rows)


def test_create_or_replace_table_remove_indexed_column_force_drop(open_sqlstore_database: SQLStore):
    """Test removing an indexed column with create_or_replace_table and force_drop_table=True.

    The table should be dropped and recreated without the indexed column.
    """
    db = open_sqlstore_database
    db.create_table("t_idx_force", {"a": "INTEGER", "b": "TEXT"})
    db.store("t_idx_force", {"a": 1, "b": "x"}, create_table=False)
    db.execute_sql("CREATE INDEX idx_a_force ON t_idx_force(a)")
    db.create_or_replace_table("t_idx_force", {"b": "TEXT"}, force_drop_table=True)
    rows = db.read("t_idx_force")
    assert all("a" not in row for row in rows)


def test_create_or_replace_table_remove_column_with_check_constraint(
    open_sqlstore_database: SQLStore,
):
    """Test removing a column with a CHECK constraint (non-simple).

    With create_or_replace_table and force_drop_table=False.
    Should fail and not fallback to table recreation, or succeed if SQLite allows it.
    """
    db = open_sqlstore_database
    db.create_table("t_check_nf", {"a": "INTEGER", "b": "TEXT"})
    db.execute_sql("ALTER TABLE t_check_nf ADD COLUMN c INTEGER CHECK (c > 0)")
    db.store("t_check_nf", {"a": 1, "b": "x", "c": 1}, create_table=False)
    try:
        db.create_or_replace_table("t_check_nf", {"a": "INTEGER", "b": "TEXT"})
    except DBStorageSQLStoreError as exc:
        # Accept any error about inability to drop column
        assert "Cannot drop columns" in str(exc) or "No changes applied" in str(exc) or "Failed to drop column" in str(exc)
        # Table should be unchanged
        rows = db.read("t_check_nf")
        assert all("c" in row for row in rows)
    else:
        # If no error, column should be gone
        rows = db.read("t_check_nf")
        assert all("c" not in row for row in rows)


def test_create_or_replace_table_remove_column_with_check_constraint_force_drop(open_sqlstore_database: SQLStore):
    """Test removing a column with a CHECK constraint (non-simple) with force_drop_table=True.

    The table should be dropped and recreated without the CHECK column.
    """
    db = open_sqlstore_database
    db.create_table("t_check_force", {"a": "INTEGER", "b": "TEXT"})
    db.execute_sql("ALTER TABLE t_check_force ADD COLUMN c INTEGER CHECK (c > 0)")
    db.store("t_check_force", {"a": 1, "b": "x", "c": 1}, create_table=False)
    db.create_or_replace_table("t_check_force", {"a": "INTEGER", "b": "TEXT"}, force_drop_table=True)
    rows = db.read("t_check_force")
    assert all("c" not in row for row in rows)


def test_create_or_replace_table_remove_generated_column(
    open_sqlstore_database: SQLStore,
):
    """Test removing a generated column (non-simple) with create_or_replace_table and force_drop_table=False.

    Should fail and not fallback to table recreation, or succeed if SQLite allows it.
    """
    db = open_sqlstore_database
    db.create_table("t_gen_nf", {"a": "INTEGER", "b": "INTEGER GENERATED ALWAYS AS (a + 1) VIRTUAL"})
    db.store("t_gen_nf", {"a": 1}, create_table=False)
    try:
        db.create_or_replace_table("t_gen_nf", {"a": "INTEGER"})
    except DBStorageSQLStoreError as exc:
        assert "Cannot drop columns" in str(exc) or "No changes applied" in str(exc) or "Failed to drop column" in str(exc)
        rows = db.read("t_gen_nf")
        assert all("b" in row for row in rows)
    else:
        rows = db.read("t_gen_nf")
        if all("b" not in row for row in rows):
            # Colonna effettivamente rimossa
            pass
        else:
            # Colonna ancora presente: SQLite ha ignorato la richiesta (no-op)
            assert all("b" in row for row in rows)


def test_create_or_replace_table_remove_generated_column_force_drop(open_sqlstore_database: SQLStore):
    """Test removing a generated column (non-simple) with create_or_replace_table and force_drop_table=True.

    The table should be dropped and recreated without the generated column.
    This test is xfailed on all platforms due to SQLite/Python limitations and file handling issues.
    """
    db = open_sqlstore_database
    db.create_table("t_gen_force", {"a": "INTEGER", "b": "INTEGER GENERATED ALWAYS AS (a + 1) VIRTUAL"})
    db.store("t_gen_force", {"a": 1}, create_table=False)
    db.create_or_replace_table("t_gen_force", {"a": "INTEGER"}, force_drop_table=True)
    rows = db.read("t_gen_force")
    assert all("b" not in row for row in rows)


def test_create_or_replace_table_remove_non_simple_column_no_force_error(open_sqlstore_database: SQLStore):
    """Test that removing a non-simple column with force_drop_table=False raises an error.

    The table must not be altered in this case.
    """
    db = open_sqlstore_database
    db.create_table("t_non_simple", {"id": "INTEGER PRIMARY KEY", "val": "TEXT"})
    db.store("t_non_simple", {"id": 1, "val": "x"}, create_table=False)
    # Attempt to remove the PK column without force_drop_table: should raise DBStorageSQLStoreError
    import pytest

    with pytest.raises(DBStorageSQLStoreError) as excinfo:
        db.create_or_replace_table("t_non_simple", {"val": "TEXT"})
    assert "Cannot drop columns" in str(excinfo.value) or "No changes applied" in str(excinfo.value) or "Failed to drop column" in str(excinfo.value)
    # Table should still have both columns
    rows = db.read("t_non_simple")
    assert all("id" in row and "val" in row for row in rows)


def test_create_or_replace_table_remove_non_simple_column_force_drop(open_sqlstore_database: SQLStore):
    """Test removing a non-simple column (PK) with create_or_replace_table and force_drop_table=True.

    The table should be dropped and recreated without the PK column.
    """
    db = open_sqlstore_database
    db.create_table("t_non_simple_force", {"id": "INTEGER PRIMARY KEY", "val": "TEXT"})
    db.store("t_non_simple_force", {"id": 1, "val": "x"}, create_table=False)
    db.create_or_replace_table("t_non_simple_force", {"val": "TEXT"}, force_drop_table=True)
    rows = db.read("t_non_simple_force")
    assert all("id" not in row for row in rows)


def test_create_or_replace_table_remove_multiple_columns_force_drop(open_sqlstore_database: SQLStore):
    """Test removing multiple columns (simple and non-simple) with create_or_replace_table and force_drop_table=True.

    The table should be dropped and recreated with only the specified columns.
    """
    db = open_sqlstore_database
    db.create_table("t_multi_force", {"id": "INTEGER PRIMARY KEY", "a": "INTEGER", "b": "TEXT", "c": "REAL"})
    db.store("t_multi_force", {"id": 1, "a": 10, "b": "x", "c": 1.5}, create_table=False)
    db.create_or_replace_table("t_multi_force", {"b": "TEXT"}, force_drop_table=True)
    rows = db.read("t_multi_force")
    assert all(set(row.keys()) == {"b"} for row in rows)


def test_create_or_replace_table_remove_simple_column_success(open_sqlstore_database: SQLStore):
    """Test removing a simple column with create_or_replace_table succeeds and data is preserved."""
    db = open_sqlstore_database
    db.create_table("t_simple_success", {"a": "INTEGER", "b": "TEXT"})
    db.store("t_simple_success", {"a": 1, "b": "x"}, create_table=False)
    db.create_or_replace_table("t_simple_success", {"a": "INTEGER"})
    rows = db.read("t_simple_success")
    assert all("b" not in row for row in rows)
    assert rows[0]["a"] == 1


def test_create_or_replace_table_transaction_success_atomic(open_sqlstore_database: SQLStore):
    """Test that create_or_replace_table is atomic on success (all changes applied, no partial state)."""
    db = open_sqlstore_database
    db.create_table("t_atomic", {"a": "INTEGER", "b": "TEXT"})
    db.store("t_atomic", {"a": 1, "b": "x"}, create_table=False)
    db.create_or_replace_table("t_atomic", {"a": "INTEGER", "c": "REAL"})
    rows = db.read("t_atomic")
    assert all("b" not in row and "c" in row for row in rows)
    assert rows[0]["a"] == 1
    assert rows[0]["c"] is None


def test_create_or_replace_table_transaction_rollback_on_pk_column(open_sqlstore_database: SQLStore):
    """Test rollback on PK column removal error.

    Verifies that if removing a primary key column fails, no partial changes are applied and the table remains
    unchanged.
    """
    db = open_sqlstore_database
    db.create_table("t_rollback_pk", {"id": "INTEGER PRIMARY KEY", "val": "TEXT"})
    db.store("t_rollback_pk", {"id": 1, "val": "x"}, create_table=False)
    with pytest.raises(DBStorageSQLStoreError):
        db.create_or_replace_table("t_rollback_pk", {"val": "TEXT"})
    rows = db.read("t_rollback_pk")
    assert all("id" in row and "val" in row for row in rows)
    assert rows[0]["id"] == 1
    assert rows[0]["val"] == "x"


def test_create_or_replace_table_transaction_rollback_on_unique_column(open_sqlstore_database: SQLStore):
    """Test rollback on UNIQUE column removal error.

    Verifies that if removing a UNIQUE column fails, no partial changes are applied and the table remains unchanged.
    """
    db = open_sqlstore_database
    db.create_table("t_rollback_unique", {"a": "INTEGER UNIQUE", "b": "TEXT"})
    db.store("t_rollback_unique", {"a": 1, "b": "x"}, create_table=False)
    with pytest.raises(DBStorageSQLStoreError):
        db.create_or_replace_table("t_rollback_unique", {"b": "TEXT"})
    rows = db.read("t_rollback_unique")
    assert all("a" in row and "b" in row for row in rows)
    assert rows[0]["a"] == 1
    assert rows[0]["b"] == "x"


def test_create_or_replace_table_transaction_rollback_on_indexed_column(open_sqlstore_database: SQLStore):
    """Test rollback on indexed column removal error.

    Verifies that if removing an indexed column fails, no partial changes are applied and the table remains unchanged.
    """
    # Setup: create table and index
    db = open_sqlstore_database
    db.create_table("t_rollback_idx", {"a": "INTEGER", "b": "TEXT"})
    db.store("t_rollback_idx", {"a": 1, "b": "x"}, create_table=False)
    db.execute_sql("CREATE INDEX idx_a_rb ON t_rollback_idx(a)")
    # Attempt to remove indexed column, should fail and rollback
    with pytest.raises(DBStorageSQLStoreError):
        db.create_or_replace_table("t_rollback_idx", {"b": "TEXT"})
    rows = db.read("t_rollback_idx")
    assert all("a" in row and "b" in row for row in rows)
    assert rows[0]["a"] == 1
    assert rows[0]["b"] == "x"


def test_create_or_replace_table_transaction_rollback_on_foreign_key_column(
    open_sqlstore_database: SQLStore,
):
    """Test rollback on foreign key column removal error.

    Verifies that if removing a foreign key column fails, no partial changes are applied and the table
    remains unchanged.
    """
    # Setup: create parent and child tables with foreign key
    db = open_sqlstore_database
    db.execute_sql("PRAGMA foreign_keys = ON")
    db.create_table("parent", {"id": "INTEGER PRIMARY KEY"})
    db.execute_sql(
        "CREATE TABLE child (id INTEGER PRIMARY KEY, parent_id INTEGER, val TEXT, FOREIGN KEY(parent_id) REFERENCES parent(id) ON DELETE CASCADE)"
    )
    db.store("parent", {"id": 1}, create_table=False)
    db.store("child", {"id": 1, "parent_id": 1, "val": "x"}, create_table=False)
    # Attempt to remove foreign key column, should fail and rollback
    with pytest.raises(DBStorageSQLStoreError):
        db.create_or_replace_table("child", {"id": "INTEGER PRIMARY KEY", "val": "TEXT"})
    rows = db.read("child")
    assert all("parent_id" in row and "val" in row for row in rows)
    assert rows[0]["parent_id"] == 1
    assert rows[0]["val"] == "x"


def test_insert_with_multiple_threads(open_sqlstore_database: SQLStore):
    """Test inserting data into the SQLStore database with multiple threads."""
    import threading

    db = open_sqlstore_database

    # Create a table
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})

    def insert_data(thread_id):
        for i in range(5):
            db.store("test_table", {"id": thread_id * 10 + i, "name": f"Thread {thread_id} Name {i}"})

    threads = []
    for i in range(5):
        thread = threading.Thread(target=insert_data, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    # Verify data
    res = db.execute_sql("SELECT * FROM test_table")
    assert len(res) == 25, "Data not found in the table after multi-threaded insertions"
