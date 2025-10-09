# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import sqlite3
import os
import threading
from typing import Optional, Any

from arduino.app_utils import brick, Logger

logger = Logger("SQLStore")


class DBStorageSQLStoreError(Exception):
    """Exception raised for SQLite database operations errors.

    This exception is raised when database operations fail, such as connection
    errors, SQL syntax errors, constraint violations, or table access issues.
    """

    pass


@brick
class SQLStore:
    """SQLStore client for storing and retrieving data in a SQLite database.

    This class provides methods to create tables, insert, read, update, and delete records,
    and execute raw SQL commands. It uses SQLite as the underlying database engine and
    supports named access to columns using sqlite3.Row as the row factory.
    It is designed to be thread-safe and can be used in multi-threaded applications.
    """

    def __init__(self, database_name: str = "arduino.db"):
        """Initialize the SQLStore client with automatic directory setup.

        Creates the database file in the data/dbstorage_sqlstore/ directory within
        the application root. If the filename doesn't end with `.db`, the extension
        is automatically added.

        Args:
            database_name (str, optional): Name of the SQLite database file.
                Defaults to "arduino.db".
        """
        app_root_dir = os.getenv("APP_HOME", ".")
        data_dir = os.path.join(app_root_dir, "data", "dbstorage_sqlstore")
        os.makedirs(data_dir, exist_ok=True)
        self.database_name = f"{data_dir}/{database_name}"
        if not self.database_name.endswith(".db"):
            self.database_name = f"{self.database_name}.db"
        self.conn = None
        self.conn_lock = threading.RLock()

    def _connect(self):
        """Establish a thread-safe connection to the SQLite database.

        Sets up the connection with named column access using sqlite3.Row factory
        and configures thread-safety with check_same_thread=False.

        Raises:
            DBStorageSQLStoreError: If there is an error connecting to the database.

        Note:
            This is an internal method. Use start() instead for public API.
        """
        with self.conn_lock:
            if self.conn:
                return

        try:
            with self.conn_lock:
                self.conn = sqlite3.connect(self.database_name, check_same_thread=False)
                self.conn.row_factory = sqlite3.Row  # Enable named access to columns
            logger.info(f"Connected to SQLite database: {self.database_name}")
        except sqlite3.Error as e:
            raise DBStorageSQLStoreError(f"Failed to connect to SQLite database: {e}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get the SQLite connection object. If the connection does not exist, it will create one.

        Returns:
            sqlite3.Connection: The active SQLite connection object with thread-safety enabled.

        Raises:
            DBStorageSQLStoreError: If there is an error establishing the connection.

        Note:
            This is an internal method used by other SQLStore operations.
        """
        with self.conn_lock:
            if not self.conn:
                self._connect()
            return self.conn

    def start(self):
        """Open the SQLite database connection.

        This method establishes the database connection and should be called before
        performing any database operations. The connection is thread-safe and enables
        named column access using sqlite3.Row factory.

        Raises:
            DBStorageSQLStoreError: If there is an error starting the SQLite connection.
        """
        with self.conn_lock:
            if not self.conn:
                self._connect()

    def stop(self):
        """Close the SQLite database connection.

        Raises:
            DBStorageSQLStoreError: If there is an error stopping the SQLite connection.
        """
        with self.conn_lock:
            if self.conn:
                try:
                    self.conn.close()
                except sqlite3.Error as e:
                    raise DBStorageSQLStoreError(f"Error stopping SQLite connection: {e}")

    def create_table(self, table: str, columns: dict[str, str]):
        """Create a table in the SQLite database if it does not already exist.

        Args:
            table (str): Name of the table to create.
            columns (dict[str, str]): Dictionary mapping column names to SQL types.
                Common types: "INTEGER", "REAL", "TEXT", "BLOB", "INTEGER PRIMARY KEY".

        Raises:
            DBStorageSQLStoreError: If there is an error creating the table.
        """
        with self.conn_lock:
            if not self.conn:
                self._connect()

        columns_definition = ", ".join([f"{col} {dtype}" for col, dtype in columns.items()])
        sql = f"CREATE TABLE IF NOT EXISTS {table} ({columns_definition})"

        try:
            with self.conn_lock:
                cursor = self._get_connection().cursor()
                cursor.execute(sql)
                self.conn.commit()
            logger.debug(f"Created table {table} with columns {columns}")
        except sqlite3.Error as e:
            raise DBStorageSQLStoreError(f"Error creating table {table}: {e}")

    def drop_table(self, table: str):
        """Remove a table and all its data from the database. This permanently deletes the table and all its data.

        Args:
            table (str): Name of the table to drop.

        Raises:
            DBStorageSQLStoreError: If there is an error dropping the table.
        """
        with self.conn_lock:
            if not self.conn:
                self._connect()

        sql = f"DROP TABLE IF EXISTS {table}"

        try:
            with self.conn_lock:
                cursor = self._get_connection().cursor()
                cursor.execute(sql)
                self.conn.commit()
            logger.debug(f"Dropped table {table}")
        except sqlite3.Error as e:
            raise DBStorageSQLStoreError(f"Error dropping table {table}: {e}")

    def store(self, table: str, data: dict[str, Any], create_table: bool = True):
        """Store data in the specified table with automatic table creation. By default, it creates the table if it doesn't exist.

        Args:
            table (str): Name of the table to store the record in.
            data (dict[str, Any]): Dictionary of column names and their values.
                Supported types: int (INTEGER), float (REAL), str (TEXT), bytes (BLOB).
            create_table (bool, optional): If True, create the table if it doesn't exist
                using automatic type inference. Defaults to True.

        Raises:
            DBStorageSQLStoreError: If there is an error inserting data or creating the table.
        """
        with self.conn_lock:
            if not self.conn:
                self._connect()

        if create_table:
            columns = {}
            for k, v in data.items():
                if isinstance(v, int):
                    columns[k] = "INTEGER"
                elif isinstance(v, float):
                    columns[k] = "REAL"
                elif isinstance(v, str):
                    columns[k] = "TEXT"
                elif isinstance(v, bytes):
                    columns[k] = "BLOB"
                else:
                    raise DBStorageSQLStoreError(f"Unsupported data type for column {k}: {type(v)}")
            self.create_table(table, columns)

        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"

        try:
            with self.conn_lock:
                cursor = self._get_connection().cursor()
                cursor.execute(sql, tuple(data.values()))
                self.conn.commit()
            logger.debug(f"Inserted data into {table}: {data}")
        except sqlite3.Error as e:
            raise DBStorageSQLStoreError(f"Error inserting data into {table}: {e}")

    def read(
        self,
        table: str,
        columns: Optional[list] = None,
        condition: Optional[str] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = -1,
    ) -> list[dict[str, Any]]:
        """Get data from the specified table with flexible filtering options. If the table does not exist, it returns an empty list.

        Args:
            table (str): Name of the table to read from.
            columns (Optional[list], optional): List of column names to select.
                If None, selects all columns. Defaults to None.
            condition (Optional[str], optional): WHERE clause for filtering results
                (e.g., "age > 18"). Defaults to None.
            order_by (Optional[str], optional): ORDER BY clause for sorting results
                (e.g., "name ASC"). Defaults to None.
            limit (Optional[int], optional): Maximum number of rows to return.
                Use -1 for no limit. Defaults to -1.

        Returns:
            list[dict[str, Any]]: List of dictionaries representing the rows, where each
                dictionary maps column names to their values. Empty list if table doesn't exist.

        Raises:
            DBStorageSQLStoreError: If there is an error reading data from the table.
        """
        with self.conn_lock:
            if not self.conn:
                self._connect()

        columns = ", ".join(columns) if columns else "*"
        sql = f"SELECT {columns} FROM {table}"
        if condition:
            sql += f" WHERE {condition}"
        if order_by:
            sql += f" ORDER BY {order_by}"
        if limit > 0:
            sql += f" LIMIT {limit}"

        try:
            with self.conn_lock:
                cursor = self._get_connection().cursor()
                cursor.execute(sql)
                col_names = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
            return [dict(zip(col_names, row)) for row in rows]
        except sqlite3.Error as e:
            if "no such table" in str(e):
                return []
            raise DBStorageSQLStoreError(f"Error reading data from {table}: {e}")

    def update(self, table: str, data: dict[str, Any], condition: Optional[str] = ""):
        """Update data or records in the specified table.

        Args:
            table (str): Name of the table to update.
            data (dict[str, Any]): Dictionary of column names and their new values.
            condition (Optional[str], optional): WHERE clause for filtering which records
                to update (e.g., "id = 1"). If empty, updates all records. Defaults to "".

        Raises:
            DBStorageSQLStoreError: If there is error updating data in the table.
        """
        with self.conn_lock:
            if not self.conn:
                self._connect()

        set_clause = ", ".join([f"{col} = ?" for col in data.keys()])
        sql = f"UPDATE {table} SET {set_clause}"
        if condition:
            sql += f" WHERE {condition}"

        try:
            with self.conn_lock:
                cursor = self._get_connection().cursor()
                cursor.execute(sql, tuple(data.values()))
                self.conn.commit()
            logger.debug(f"Updated data in {table}: {data} where {condition}")
        except sqlite3.Error as e:
            raise DBStorageSQLStoreError(f"Error updating data in {table}: {e}")

    def delete(self, table: str, condition: Optional[str] = ""):
        """Delete data from the specified table. If no condition is provided, this will delete ALL records from the table.

        Args:
            table (str): Name of the table to delete from.
            condition (Optional[str], optional): WHERE clause for filtering which records
                to delete (e.g., "age < 18"). If empty, deletes all records. Defaults to "".

        Raises:
            DBStorageSQLStoreError: If there is an error deleting data from the table.
        """
        with self.conn_lock:
            if not self.conn:
                self._connect()

        sql = f"DELETE FROM {table}"
        if condition:
            sql += f" WHERE {condition}"

        try:
            with self.conn_lock:
                cursor = self._get_connection().cursor()
                cursor.execute(sql)
                self.conn.commit()
            logger.debug(f"Deleted data from {table} where {condition}")
        except sqlite3.Error as e:
            raise DBStorageSQLStoreError(f"Error deleting data from {table}: {e}")

    def execute_sql(self, sql: str, args: Optional[tuple] = None) -> list[dict[str, Any]] | None:
        """Execute a raw SQL command.

        Args:
            sql (str): The SQL command to execute.
            args (Optional[tuple]): Optional parameters for the SQL command.

        Returns:
           list[dict[str, Any]] | None: A list of dictionaries representing the rows returned by the SQL command,
              or None if the command does not return any rows.

        Raises:
            DBStorageSQLStoreError: If there is an error executing the SQL command.
        """
        with self.conn_lock:
            if not self.conn:
                self._connect()

        try:
            with self.conn_lock:
                cursor = self._get_connection().cursor()
                cursor.execute(sql, args or ())
                logger.debug(f"Executing SQL command: {sql} with args {args}")

                first_row = cursor.fetchone()
                if first_row is None:
                    self.conn.commit()
                    return None
                else:
                    rows = [first_row] + cursor.fetchall()
                    self.conn.commit()
                    col_names = [description[0] for description in cursor.description]
            return [dict(zip(col_names, row)) for row in rows]
        except sqlite3.Error as e:
            raise DBStorageSQLStoreError(f"Error executing SQL command: {e}")

    def create_or_replace_table(self, table: str, columns: dict[str, str], force_drop_table: bool = False):
        """Create or update a table in the SQLite database to match the provided schema.

        All schema changes (adding/removing/changing columns) are performed within a single transaction.
        If any error occurs during the operation due to SQLite limitations or constraints, the transaction is rolled back
        , and the table remains unchanged. If force_drop_table is True, after rollback, the table is dropped and recreated.

        If the table exists, it will add missing columns and remove extra columns unless they are not-simple columns.
        (e.g., primary key, unique, indexed, or used in constraints/triggers/views).

        If a column's type has changed or if a column is not simple, it will raise an error unless
        force_drop_table is True, in which case the table is dropped and recreated with the new schema, losing all
        existing data in the table.

        Args:
            table (str): Name of the table to create or update.
            columns (dict[str, str]): Dictionary of column names and their SQL types.
            force_drop_table (bool): If True, always drop and recreate the table if schema change fails.

        Raises:
            DBStorageSQLStoreError: If there is an error creating or updating the table. All changes are rolled back.
        """
        with self.conn_lock:
            if not self.conn:
                self._connect()

            cursor = self._get_connection().cursor()
            # Check for persistent tables
            cursor.execute("""SELECT name FROM sqlite_master WHERE type='table' AND name=?""", tuple([table]))
            exists = cursor.fetchone() is not None

        if not exists:
            logger.debug(f"Table {table} does not exist, creating with columns: {columns}")
            self.create_table(table, columns)
            return

        try:
            logger.debug(f"Attempting schema alignment for table {table} (force_drop_table={force_drop_table})")
            with self.conn_lock:
                self.conn.execute("BEGIN")
                # Table exists: get current schema
                cursor.execute(f"PRAGMA table_xinfo({table})")
                existing_cols = {row["name"]: row["type"] for row in cursor.fetchall()}  # {col_name: col_type}
            # Find columns to add and remove
            to_add = {col: dtype for col, dtype in columns.items() if col not in existing_cols}
            to_remove = [col for col in existing_cols if col not in columns]
            type_changed = [col for col in columns if col in existing_cols and columns[col].upper() != existing_cols[col].upper()]
            logger.debug(f"Columns to add: {to_add}, to remove: {to_remove}, type changed: {type_changed}")
            dropped = []
            failed_drop = []
            # Try to drop columns, handle errors as non-simple
            for col in to_remove:
                try:
                    with self.conn_lock:
                        cursor.execute(f"ALTER TABLE {table} DROP COLUMN {col}")
                    dropped.append(col)
                    logger.debug(f"Dropped column {col} from {table}")
                except sqlite3.Error as e:
                    logger.error(f"Failed to drop column {col} from {table}: {e}")
                    failed_drop.append(col)
            if failed_drop or type_changed:
                msg = (
                    f"Cannot drop columns {failed_drop} or change column types {type_changed} in table {table} "
                    f"without force_drop_table=True. No changes applied. "
                    f"Columns may be protected by constraints, indexes, triggers, or special definitions."
                )
                logger.error(msg)
                with self.conn_lock:
                    self.conn.rollback()
                raise DBStorageSQLStoreError(msg)

            # Add columns if needed
            for col, dtype in to_add.items():
                try:
                    alter_sql = f"ALTER TABLE {table} ADD COLUMN {col} {dtype}"
                    with self.conn_lock:
                        cursor.execute(alter_sql)
                    logger.debug(f"Added column {col} {dtype} to {table}")
                except sqlite3.Error as e:
                    with self.conn_lock:
                        self.conn.rollback()
                    logger.error(f"Error adding column {col} to {table}: {e}")
                    raise DBStorageSQLStoreError(f"Error adding column {col} to {table}: {e}")
            logger.debug(f"Table {table} schema aligned. Added columns: {list(to_add.keys())}, dropped: {dropped}")
        except Exception as exc:
            logger.error(f"Schema change failed for table {table}: {exc}")
            logger.debug("Rolling back transaction")
            with self.conn_lock:
                self.conn.rollback()
            if force_drop_table:
                logger.warning(f"Dropping and recreating table {table} due to force_drop_table=True")
                self.drop_table(table)
                self.create_table(table, columns)
                logger.debug(f"Table {table} dropped and recreated with new schema: {columns}")
            else:
                raise
