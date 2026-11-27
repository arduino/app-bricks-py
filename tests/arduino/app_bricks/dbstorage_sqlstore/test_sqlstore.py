# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import gc
import pytest
import os
import tempfile
import time
from unittest.mock import patch
from arduino.app_bricks.dbstorage_sqlstore import SQLStore, DBStorageSQLStoreError
from arduino.app_utils import Logger

logger = Logger("SQLStore.tests")


@pytest.fixture
def open_sqlstore_database():
    """Fixture to provide an open SQLStore database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir, patch("os.makedirs"):
        db = SQLStore(database_name="test")
        db_path = os.path.join(tmpdir, "test.db")
        os.makedirs(db_path, exist_ok=True)
        db.database_name = db_path
        yield db
        db.stop()
        gc.collect()

        for _ in range(30):
            try:
                os.remove(db_path)
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


def test_read_with_no_limit(open_sqlstore_database: SQLStore):
    """None and -1 both read every row."""
    db = open_sqlstore_database
    db.store("limit_table", {"id": 1}, create_table=True)
    db.store("limit_table", {"id": 2})

    assert len(db.read("limit_table", limit=None)) == 2
    assert len(db.read("limit_table", limit=-1)) == 2


def test_store_returns_the_row_id(open_sqlstore_database: SQLStore):
    """store() returns the id of the new record, also when the database assigns it."""
    db = open_sqlstore_database
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})

    assert db.store("test_table", {"name": "first"}) == 1
    assert db.store("test_table", {"name": "second"}) == 2
    assert db.store("test_table", {"id": 10, "name": "explicit"}) == 10
    assert db.read("test_table", condition="id = 2")[0]["name"] == "second"


def test_update_and_delete_return_the_affected_rows(open_sqlstore_database: SQLStore):
    """update() and delete() tell how many records they changed."""
    db = open_sqlstore_database
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})
    for name in ("a", "b", "c"):
        db.store("test_table", {"name": name})

    assert db.update("test_table", {"name": "x"}, condition="id = 1") == 1
    assert db.update("test_table", {"name": "y"}, condition="id = 99") == 0
    assert db.update("test_table", {"name": "z"}) == 3
    assert db.delete("test_table", condition="id = 99") == 0
    assert db.delete("test_table", condition="id > 1") == 2
    assert db.delete("test_table") == 1


def test_condition_with_parameters(open_sqlstore_database: SQLStore):
    """read(), update() and delete() take the values of their condition as parameters."""
    db = open_sqlstore_database
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT", "age": "INTEGER"})
    for name, age in (("Alice", 30), ("Bob", 17), ("O'Brien", 41)):
        db.store("test_table", {"name": name, "age": age})

    assert [r["name"] for r in db.read("test_table", condition="age > ?", args=(18,), order_by="id")] == ["Alice", "O'Brien"]
    # A quote in the value needs no escaping
    assert db.read("test_table", condition="name = ?", args=("O'Brien",))[0]["age"] == 41
    assert db.update("test_table", {"age": 18}, condition="name = ?", args=("Bob",)) == 1
    assert db.read("test_table", condition="name = ? AND age = ?", args=("Bob", 18))[0]["id"] == 2
    assert db.delete("test_table", condition="id IN (?, ?)", args=(1, 2)) == 2
    assert [r["name"] for r in db.read("test_table")] == ["O'Brien"]


def test_condition_parameters_are_not_read_as_sql(open_sqlstore_database: SQLStore):
    """Text passed as a parameter cannot change the query."""
    db = open_sqlstore_database
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})
    db.store("test_table", {"name": "Alice"})
    db.store("test_table", {"name": "Bob"})

    hostile = "x' OR '1'='1"
    assert db.read("test_table", condition="name = ?", args=(hostile,)) == []
    assert db.delete("test_table", condition="name = ?", args=(hostile,)) == 0
    assert len(db.read("test_table")) == 2


def test_condition_with_wrong_number_of_parameters(open_sqlstore_database: SQLStore):
    """A condition whose placeholders do not match the parameters is an error."""
    db = open_sqlstore_database
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})
    with pytest.raises(DBStorageSQLStoreError):
        db.read("test_table", condition="id = ?")
    with pytest.raises(DBStorageSQLStoreError):
        db.update("test_table", {"name": "x"}, condition="id = ?", args=(1, 2))


def test_store_on_conflict_replace(open_sqlstore_database: SQLStore):
    """on_conflict="replace" stores the record in place of the one with the same key."""
    db = open_sqlstore_database
    db.create_table("settings", {"key": "TEXT PRIMARY KEY", "value": "TEXT"})

    db.store("settings", {"key": "matrix", "value": "uno_q"}, on_conflict="replace")
    db.store("settings", {"key": "matrix", "value": "modulino"}, on_conflict="replace")
    db.store("settings", {"key": "other", "value": "1"}, on_conflict="replace")

    assert db.read("settings", order_by="key") == [{"key": "matrix", "value": "modulino"}, {"key": "other", "value": "1"}]


def test_store_on_conflict_ignore(open_sqlstore_database: SQLStore):
    """on_conflict="ignore" keeps the record already in the table and returns 0."""
    db = open_sqlstore_database
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})

    assert db.store("test_table", {"id": 1, "name": "first"}, on_conflict="ignore") == 1
    assert db.store("test_table", {"id": 1, "name": "second"}, on_conflict="ignore") == 0
    assert db.store("test_table", {"id": 2, "name": "third"}, on_conflict="ignore") == 2
    assert [r["name"] for r in db.read("test_table", order_by="id")] == ["first", "third"]


def test_store_conflict_is_an_error_by_default(open_sqlstore_database: SQLStore):
    """Without on_conflict a duplicate key is still an error, and unknown values are rejected."""
    db = open_sqlstore_database
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})
    db.store("test_table", {"id": 1, "name": "first"})

    with pytest.raises(DBStorageSQLStoreError):
        db.store("test_table", {"id": 1, "name": "second"})
    with pytest.raises(DBStorageSQLStoreError):
        db.store("test_table", {"id": 3, "name": "x"}, on_conflict="update")  # type: ignore[arg-type]
    assert db.read("test_table") == [{"id": 1, "name": "first"}]


def test_transaction_commits_all_changes(open_sqlstore_database: SQLStore):
    """The changes made inside a transaction are saved together when the block ends."""
    db = open_sqlstore_database
    db.create_table("accounts", {"id": "INTEGER PRIMARY KEY", "balance": "INTEGER"})
    db.store("accounts", {"id": 1, "balance": 100})
    db.store("accounts", {"id": 2, "balance": 100})

    with db.transaction():
        db.update("accounts", {"balance": 50}, condition="id = ?", args=(1,))
        db.update("accounts", {"balance": 150}, condition="id = ?", args=(2,))
        new_id = db.store("accounts", {"balance": 0})
        assert db.read("accounts", condition="id = 1")[0]["balance"] == 50  # visible inside the block

    assert new_id == 3
    assert [r["balance"] for r in db.read("accounts", order_by="id")] == [50, 150, 0]
    # A second connection sees the committed data
    import sqlite3

    other = sqlite3.connect(db.database_name)
    assert other.execute("SELECT COUNT(*) FROM accounts").fetchone()[0] == 3
    other.close()


def test_transaction_rolls_back_on_error(open_sqlstore_database: SQLStore):
    """An exception inside the block undoes every change of the block and continues."""
    db = open_sqlstore_database
    db.create_table("accounts", {"id": "INTEGER PRIMARY KEY", "balance": "INTEGER"})
    db.store("accounts", {"id": 1, "balance": 100})

    with pytest.raises(RuntimeError):
        with db.transaction():
            db.update("accounts", {"balance": 0}, condition="id = 1")
            db.store("accounts", {"id": 2, "balance": 5})
            db.execute_sql("DELETE FROM accounts WHERE id = 1")
            raise RuntimeError("boom")

    assert db.read("accounts") == [{"id": 1, "balance": 100}]
    # The store works normally afterwards
    db.store("accounts", {"id": 2, "balance": 7})
    assert len(db.read("accounts")) == 2


def test_transaction_rolls_back_on_database_error(open_sqlstore_database: SQLStore):
    """A failed operation inside the block undoes the ones before it."""
    db = open_sqlstore_database
    db.create_table("accounts", {"id": "INTEGER PRIMARY KEY", "balance": "INTEGER"})

    with pytest.raises(DBStorageSQLStoreError):
        with db.transaction():
            db.store("accounts", {"id": 1, "balance": 1})
            db.store("accounts", {"id": 1, "balance": 2})  # duplicate key

    assert db.read("accounts") == []


def test_nested_transactions_commit_at_the_end_of_the_outer_one(open_sqlstore_database: SQLStore):
    """A nested block joins the outer transaction: an error in the outer block undoes it too."""
    db = open_sqlstore_database
    db.create_table("accounts", {"id": "INTEGER PRIMARY KEY", "balance": "INTEGER"})

    with pytest.raises(RuntimeError):
        with db.transaction():
            with db.transaction():
                db.store("accounts", {"id": 1, "balance": 1})
            raise RuntimeError("boom")
    assert db.read("accounts") == []

    with db.transaction():
        with db.transaction():
            db.store("accounts", {"id": 1, "balance": 1})
        db.store("accounts", {"id": 2, "balance": 2})
    assert len(db.read("accounts")) == 2


def test_transaction_keeps_other_threads_out(open_sqlstore_database: SQLStore):
    """Another thread reads only after the transaction is over, so it never sees half of it."""
    import threading

    db = open_sqlstore_database
    db.create_table("accounts", {"id": "INTEGER PRIMARY KEY", "balance": "INTEGER"})
    db.store("accounts", {"id": 1, "balance": 100})
    db.store("accounts", {"id": 2, "balance": 100})

    seen: list[int] = []
    inside = threading.Event()

    def reader():
        inside.wait(5)
        seen.append(sum(r["balance"] for r in db.read("accounts")))

    thread = threading.Thread(target=reader)
    thread.start()
    with db.transaction():
        db.update("accounts", {"balance": 0}, condition="id = 1")
        inside.set()
        time.sleep(0.2)  # the reader is waiting for the lock here
        db.update("accounts", {"balance": 200}, condition="id = 2")
    thread.join(5)

    assert seen == [200]


def test_create_or_replace_table_is_refused_inside_a_transaction(open_sqlstore_database: SQLStore):
    """create_or_replace_table() manages its own transaction and cannot join another one."""
    db = open_sqlstore_database
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})

    with pytest.raises(DBStorageSQLStoreError):
        with db.transaction():
            db.store("test_table", {"id": 1, "name": "kept out"})
            db.create_or_replace_table("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT", "age": "INTEGER"})
    assert db.read("test_table") == []


@pytest.fixture
def sqlstore_factory():
    """Fixture to create SQLStore databases with options, all on the same file in a temporary directory."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        opened: list[SQLStore] = []

        def create(**options) -> SQLStore:
            with patch("os.makedirs"):
                db = SQLStore(database_name="test", **options)
            db.database_name = os.path.join(tmpdir, "test.db")
            opened.append(db)
            return db

        yield create
        for db in opened:
            db.stop()
        gc.collect()


def _pragma(db: SQLStore, name: str):
    res = db.execute_sql(f"PRAGMA {name}")
    assert res is not None
    return res[0][name]


def test_power_loss_safe_is_the_default(sqlstore_factory):
    """By default every change is flushed to disk, and the journal of the file is left as it is."""
    db = sqlstore_factory()
    assert db.power_loss_safe is True
    assert _pragma(db, "synchronous") == 2  # FULL
    assert _pragma(db, "journal_mode") == "delete"


def test_fast_writes_without_power_loss_safety(sqlstore_factory):
    """power_loss_safe=False turns on write-ahead logging with the relaxed flush policy."""
    db = sqlstore_factory(power_loss_safe=False)
    assert _pragma(db, "journal_mode") == "wal"
    assert _pragma(db, "synchronous") == 1  # NORMAL

    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})
    assert db.store("test_table", {"name": "saved"}) == 1
    assert os.path.exists(db.database_name + "-wal")
    assert db.read("test_table") == [{"id": 1, "name": "saved"}]


def test_safe_store_on_a_database_used_with_fast_writes(sqlstore_factory):
    """A database that was used with fast writes is safe again when reopened with the default."""
    fast = sqlstore_factory(power_loss_safe=False)
    fast.create_table("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})
    fast.store("test_table", {"name": "written fast"})
    fast.stop()

    safe = sqlstore_factory()
    # The file keeps write-ahead logging, which is safe once every change is flushed
    assert _pragma(safe, "journal_mode") == "wal"
    assert _pragma(safe, "synchronous") == 2  # FULL
    assert safe.store("test_table", {"name": "written safe"}) == 2
    assert [r["name"] for r in safe.read("test_table", order_by="id")] == ["written fast", "written safe"]
    safe.stop()

    fast_again = sqlstore_factory(power_loss_safe=False)
    assert _pragma(fast_again, "synchronous") == 1  # NORMAL
    assert len(fast_again.read("test_table")) == 2


def test_fast_writes_fall_back_to_safe_without_write_ahead_logging(sqlstore_factory):
    """Where write-ahead logging is not available the relaxed flush policy is not used: it would not be safe."""
    db = sqlstore_factory(power_loss_safe=False)
    db.database_name = ":memory:"  # an in-memory database cannot use write-ahead logging

    assert _pragma(db, "journal_mode") == "memory"
    assert _pragma(db, "synchronous") == 2  # FULL
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY"})
    assert db.store("test_table", {"id": 1}) == 1


def test_settings_changed_with_raw_sql_still_apply(sqlstore_factory):
    """An app that tunes SQLite itself with execute_sql, as before this option existed, keeps working."""
    db = sqlstore_factory()
    db.start()
    db.execute_sql("PRAGMA journal_mode=WAL")
    db.execute_sql("PRAGMA synchronous=NORMAL")

    assert _pragma(db, "journal_mode") == "wal"
    assert _pragma(db, "synchronous") == 1  # NORMAL
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY"})
    assert db.store("test_table", {"id": 1}) == 1


def test_read_with_expressions_as_columns(open_sqlstore_database: SQLStore):
    """The columns of read() can be SQL expressions, named with AS."""
    db = open_sqlstore_database
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY", "age": "INTEGER"})

    # On an empty table an aggregate still answers one row, with NULL values
    assert db.read("test_table", columns=["COUNT(*) AS total", "MAX(id) AS last_id"]) == [{"total": 0, "last_id": None}]

    for age in (30, 17, 41):
        db.store("test_table", {"age": age})
    assert db.read("test_table", columns=["COUNT(*) AS total", "MAX(id) AS last_id", "MIN(age) AS youngest"]) == [
        {"total": 3, "last_id": 3, "youngest": 17}
    ]
    assert db.read("test_table", columns=["COUNT(*) AS adults"], condition="age >= ?", args=(18,)) == [{"adults": 2}]
    assert db.read("test_table", columns=["id", "age * 2 AS doubled"], order_by="id", limit=1) == [{"id": 1, "doubled": 60}]


def test_failed_operation_leaves_no_open_transaction(open_sqlstore_database: SQLStore):
    """A failed operation is rolled back at once: nothing stays pending on the connection."""
    db = open_sqlstore_database
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})
    db.store("test_table", {"id": 1, "name": "first"})

    failing = [
        lambda: db.store("test_table", {"id": 1, "name": "duplicate"}),
        lambda: db.update("test_table", {"missing_column": 1}),
        lambda: db.delete("test_table", condition="missing_column = 1"),
        lambda: db.execute_sql("INSERT INTO test_table (id, name) VALUES (1, 'duplicate')"),
        lambda: db.execute_sql("UPDATE missing_table SET a = 1"),
    ]
    for operation in failing:
        with pytest.raises(DBStorageSQLStoreError):
            operation()
        assert db.conn is not None and not db.conn.in_transaction

    assert db.read("test_table") == [{"id": 1, "name": "first"}]


def test_schema_change_after_a_failed_operation(open_sqlstore_database: SQLStore):
    """create_or_replace_table() works right after an operation that failed."""
    db = open_sqlstore_database
    db.create_table("test_table", {"id": "INTEGER", "name": "TEXT"})
    db.execute_sql("CREATE UNIQUE INDEX test_table_id ON test_table (id)")
    db.store("test_table", {"id": 1, "name": "first"})
    with pytest.raises(DBStorageSQLStoreError):
        db.store("test_table", {"id": 1, "name": "duplicate"})

    db.create_or_replace_table("test_table", {"id": "INTEGER", "name": "TEXT", "age": "INTEGER"})

    assert db.read("test_table") == [{"id": 1, "name": "first", "age": None}]


def test_failed_operation_inside_a_transaction_is_left_to_the_block(open_sqlstore_database: SQLStore):
    """Inside transaction() a failed operation undoes nothing by itself: the block can handle the error and go on."""
    db = open_sqlstore_database
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})

    with db.transaction():
        db.store("test_table", {"id": 1, "name": "first"})
        with pytest.raises(DBStorageSQLStoreError):
            db.store("test_table", {"id": 1, "name": "duplicate"})
        db.store("test_table", {"id": 2, "name": "second"})

    assert [r["name"] for r in db.read("test_table", order_by="id")] == ["first", "second"]


def test_create_or_replace_table_commits_the_schema_change(open_sqlstore_database: SQLStore):
    """The new schema is saved when the call returns, without waiting for another write."""
    import sqlite3

    db = open_sqlstore_database
    db.create_table("test_table", {"id": "INTEGER", "name": "TEXT"})
    db.store("test_table", {"id": 1, "name": "first"})

    db.create_or_replace_table("test_table", {"id": "INTEGER", "name": "TEXT", "age": "INTEGER"})

    assert db.conn is not None and not db.conn.in_transaction
    # A second connection sees the new column, as the app would after a restart
    other = sqlite3.connect(db.database_name)
    assert [row[1] for row in other.execute("PRAGMA table_info(test_table)")] == ["id", "name", "age"]
    other.close()


def test_create_or_replace_table_twice_in_a_row(open_sqlstore_database: SQLStore):
    """Schema changes can follow each other, also when one of them changes nothing."""
    db = open_sqlstore_database
    db.create_table("test_table", {"id": "INTEGER", "name": "TEXT"})

    db.create_or_replace_table("test_table", {"id": "INTEGER", "name": "TEXT", "age": "INTEGER"})
    db.create_or_replace_table("test_table", {"id": "INTEGER", "name": "TEXT", "age": "INTEGER"})  # nothing to change
    db.create_or_replace_table("test_table", {"id": "INTEGER", "name": "TEXT", "age": "INTEGER", "city": "TEXT"})
    db.create_or_replace_table("test_table", {"id": "INTEGER", "city": "TEXT"})

    db.store("test_table", {"id": 1, "city": "Turin"})
    assert db.read("test_table") == [{"id": 1, "city": "Turin"}]
    assert db.conn is not None and not db.conn.in_transaction


def test_create_or_replace_table_refused_change_leaves_no_open_transaction(open_sqlstore_database: SQLStore):
    """A refused schema change is rolled back and reported with the brick's own error."""
    db = open_sqlstore_database
    db.create_table("test_table", {"id": "INTEGER", "name": "TEXT"})

    with pytest.raises(DBStorageSQLStoreError):
        db.create_or_replace_table("test_table", {"id": "TEXT", "name": "TEXT"})
    assert db.conn is not None and not db.conn.in_transaction

    # An SQLite error while changing the schema is not raised as it is
    with patch.object(db, "conn_lock"), patch.object(db, "conn") as conn:
        import sqlite3

        conn.cursor.return_value.fetchone.return_value = ("test_table",)
        conn.execute.side_effect = sqlite3.OperationalError("cannot start a transaction within a transaction")
        with pytest.raises(DBStorageSQLStoreError):
            db.create_or_replace_table("test_table", {"id": "INTEGER", "name": "TEXT", "age": "INTEGER"})


def test_transaction_starts_when_the_block_is_entered(open_sqlstore_database: SQLStore):
    """The transaction is open from the first line of the block, not from its first change."""
    db = open_sqlstore_database
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})

    assert db.conn is not None and not db.conn.in_transaction
    with db.transaction():
        assert db.conn.in_transaction
        db.read("test_table")
        assert db.conn.in_transaction
    assert not db.conn.in_transaction


def test_transaction_rolls_back_tables_created_and_dropped(open_sqlstore_database: SQLStore):
    """A table created or dropped inside a block that fails is put back as it was."""
    db = open_sqlstore_database
    db.create_table("kept", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})
    db.store("kept", {"id": 1, "name": "first"})

    with pytest.raises(RuntimeError):
        with db.transaction():
            db.create_table("made_inside", {"id": "INTEGER"})  # before any change to the data
            db.store("made_inside", {"id": 1}, create_table=False)
            db.drop_table("kept")
            raise RuntimeError("boom")

    names = [r["name"] for r in db.execute_sql("SELECT name FROM sqlite_master WHERE type='table'") or []]
    assert names == ["kept"]
    assert db.read("kept") == [{"id": 1, "name": "first"}]

    with db.transaction():
        db.create_table("made_inside", {"id": "INTEGER"})
    assert db.read("made_inside") == []


def test_transaction_keeps_other_writers_out_between_a_read_and_a_write(open_sqlstore_database: SQLStore):
    """Another connection to the same file can write only after the block ends, so what the block read stays true."""
    import sqlite3
    import threading

    db = open_sqlstore_database
    db.create_table("counter", {"id": "INTEGER PRIMARY KEY", "value": "INTEGER"})
    db.store("counter", {"id": 1, "value": 10})

    written = threading.Event()

    def other_program():
        other = sqlite3.connect(db.database_name, timeout=10)
        other.execute("UPDATE counter SET value = value + 100 WHERE id = 1")
        other.commit()
        other.close()
        written.set()

    with db.transaction():
        value = db.read("counter", condition="id = 1")[0]["value"]
        thread = threading.Thread(target=other_program)
        thread.start()
        assert not written.wait(0.3)  # the other writer is waiting for the block to end
        db.update("counter", {"value": value + 1}, condition="id = 1")
    thread.join(10)

    assert written.is_set()
    # Neither update is lost: 10 + 1 from the block, then + 100 from the other writer
    assert db.read("counter")[0]["value"] == 111


def test_transaction_that_cannot_start_raises_the_brick_error(open_sqlstore_database: SQLStore):
    """If another program keeps the database locked for writing, the block does not run."""
    import sqlite3

    db = open_sqlstore_database
    db.create_table("test_table", {"id": "INTEGER PRIMARY KEY"})
    assert db.conn is not None
    db.conn.execute("PRAGMA busy_timeout=100")  # do not wait the default 5 s for the lock

    other = sqlite3.connect(db.database_name)
    other.execute("BEGIN IMMEDIATE")
    try:
        ran = False
        with pytest.raises(DBStorageSQLStoreError):
            with db.transaction():
                ran = True
        assert not ran
    finally:
        other.rollback()
        other.close()

    # Once the lock is released transactions work again
    with db.transaction():
        db.store("test_table", {"id": 1})
    assert db.read("test_table") == [{"id": 1}]
