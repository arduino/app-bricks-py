# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import sqlite3
import os
import threading
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Literal

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

    def __init__(self, database_name: str = "arduino.db", power_loss_safe: bool = True) -> None:
        """Initialize the SQLStore client with automatic directory setup.

        Creates the database file in the `/app/data`. If the filename doesn't end with `.db`, the extension
        is automatically added.

        Args:
            database_name (str, optional): Name of the SQLite database file.
                Defaults to "arduino.db".
            power_loss_safe (bool, optional): Whether a change must be safe on disk as soon as it is
                made. With True, the default, every change is flushed to disk before the call
                returns: it survives even a power loss right after it, but each change waits for
                the storage of the board, a few milliseconds or more. With False changes are much
                faster, which suits an app that saves often (e.g. at every user action): the last
                changes made before a power loss or a system crash can be lost, but the database is
                never corrupted, and nothing is lost when only the app stops or crashes.
        """
        self.power_loss_safe = bool(power_loss_safe)
        data_dir = "/app/data"
        os.makedirs(data_dir, exist_ok=True)
        self.database_name = f"{data_dir}/{database_name}"
        if not self.database_name.endswith(".db"):
            self.database_name = f"{self.database_name}.db"
        self.conn: sqlite3.Connection | None = None
        self.conn_lock = threading.RLock()
        self._transaction_depth = 0  # > 0 inside transaction(): changed only with conn_lock held

    def _connect(self) -> None:
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
                self._apply_power_loss_safety(self.conn)
            logger.info(f"Connected to SQLite database: {self.database_name}")
        except sqlite3.Error as e:
            raise DBStorageSQLStoreError(f"Failed to connect to SQLite database: {e}")

    def _apply_power_loss_safety(self, conn: sqlite3.Connection) -> None:
        """Set how changes reach the disk on a new connection, as asked with power_loss_safe.

        Fast writes need two SQLite settings together: write-ahead logging, which is saved in the
        database file, and a relaxed flush policy, which belongs to the connection. The relaxed
        policy is safe only with write-ahead logging: if that cannot be turned on, changes keep
        being flushed one by one. A safe store only needs the flush at every change, whatever the
        journal of the file: a database that was used with fast writes stays in write-ahead
        logging, which is just as safe once every change is flushed.
        """
        if self.power_loss_safe:
            conn.execute("PRAGMA synchronous=FULL")
            return
        try:
            # The pragma answers with the mode in use: some filesystems cannot do write-ahead logging
            mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        except sqlite3.Error as e:
            mode = f"unchanged ({e})"
        if mode == "wal":
            conn.execute("PRAGMA synchronous=NORMAL")
        else:
            conn.execute("PRAGMA synchronous=FULL")
            logger.warning(f"Fast writes not available for {self.database_name} (journal mode: {mode}): changes are flushed one by one")

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
            if self.conn is None:
                raise DBStorageSQLStoreError("SQLite connection is not available")
            return self.conn

    def _commit(self, conn: sqlite3.Connection) -> None:
        """Commit the changes of an operation, unless a transaction is open: it commits at its end."""
        if self._transaction_depth == 0:
            conn.commit()

    def _rollback_failed(self) -> None:
        """Undo what a failed operation left pending, so that it cannot be committed by a later one.

        SQLite opens a transaction at the first change and keeps it open until a commit or a
        rollback: without this, the transaction of a failed operation would stay open. Inside
        transaction() nothing is undone here: the block decides at its end.
        """
        with self.conn_lock:
            if self.conn is not None and self._transaction_depth == 0:
                try:
                    self.conn.rollback()
                except sqlite3.Error as e:
                    # The caller is already reporting the error of the operation: do not hide it
                    logger.debug(f"Rollback of a failed operation not done: {e}")

    @contextmanager
    def transaction(self) -> Generator[None]:
        """Group several operations so that they are saved together, or not at all.

        Use it as a context manager, `with db.transaction():`. The changes made inside the block are committed when the
        block ends; if the block raises an exception they are all undone and the exception
        continues. Tables created or dropped inside the block are undone too.

        While the block runs nothing else can change the database: the other threads wait before
        reading or writing it, and so does another program that writes to the same file. What the
        block reads stays true until it ends, so it can read a value and write one that depends
        on it. Keep the block short.

        Saving the changes together is also faster than saving them one by one, since the
        database is written to disk once.

        Transactions can be nested: the changes are committed when the outermost block ends.
        `create_or_replace_table()` cannot be used inside a transaction.

        Raises:
            DBStorageSQLStoreError: If the transaction cannot be started, e.g. because another
                program keeps the database locked, or there is an error committing the changes.
        """
        with self.conn_lock:
            conn = self._get_connection()
            if self._transaction_depth == 0:
                # Start the transaction now, taking the right to write. Without it SQLite would
                # start one only at the first change: a table created before it could not be
                # undone, and another program could write between a read and that change.
                try:
                    conn.execute("BEGIN IMMEDIATE")
                except sqlite3.Error as e:
                    raise DBStorageSQLStoreError(f"Error starting the transaction: {e}")
            self._transaction_depth += 1
            try:
                yield
            except BaseException:
                if self._transaction_depth == 1:
                    conn.rollback()
                raise
            else:
                if self._transaction_depth == 1:
                    try:
                        conn.commit()
                    except sqlite3.Error as e:
                        conn.rollback()
                        raise DBStorageSQLStoreError(f"Error committing the transaction: {e}")
            finally:
                self._transaction_depth -= 1

    def start(self) -> None:
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

    def stop(self) -> None:
        """Close the SQLite database connection.

        The connection and the files it keeps open are released. The store can be used again
        afterwards: start(), or any operation, opens a new connection to the same database.
        Calling stop() on a store that is not open does nothing.

        Raises:
            DBStorageSQLStoreError: If there is an error stopping the SQLite connection, or stop()
                is called inside a transaction.
        """
        with self.conn_lock:
            if self._transaction_depth > 0:
                # The rest of the block would run on a new connection, outside the transaction
                raise DBStorageSQLStoreError("stop() cannot be used inside a transaction")
            if self.conn:
                try:
                    self.conn.close()
                except sqlite3.Error as e:
                    raise DBStorageSQLStoreError(f"Error stopping SQLite connection: {e}")
                finally:
                    # Forget the connection also when closing it failed: a closed one cannot be used again
                    self.conn = None

    def create_table(self, table: str, columns: dict[str, str]) -> None:
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
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute(sql)
                self._commit(conn)
            logger.debug(f"Created table {table} with columns {columns}")
        except sqlite3.Error as e:
            self._rollback_failed()
            raise DBStorageSQLStoreError(f"Error creating table {table}: {e}")

    def drop_table(self, table: str) -> None:
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
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute(sql)
                self._commit(conn)
            logger.debug(f"Dropped table {table}")
        except sqlite3.Error as e:
            self._rollback_failed()
            raise DBStorageSQLStoreError(f"Error dropping table {table}: {e}")

    def store(
        self,
        table: str,
        data: dict[str, Any],
        create_table: bool = True,
        on_conflict: Literal["replace", "ignore"] | None = None,
    ) -> int:
        """Store data in the specified table with automatic table creation. By default, it creates the table if it doesn't exist.

        Args:
            table (str): Name of the table to store the record in.
            data (dict[str, Any]): Dictionary of column names and their values.
                Supported types: int (INTEGER), float (REAL), str (TEXT), bytes (BLOB).
            create_table (bool, optional): If True, create the table if it doesn't exist
                using automatic type inference. Defaults to True.
            on_conflict (Optional[str], optional): What to do when the record has the same
                primary key, or the same value in a `UNIQUE` column, as a record already in the
                table. "replace" stores the new record in place of the old one, "ignore" keeps the
                old one. With None, the default, the conflict is an error.

        Returns:
            int: The row id of the inserted record. In a table with an `INTEGER PRIMARY KEY` column it
                is the value of that column, so a record stored without it gets its id back.
                0 if the record was ignored because of a conflict.

        Raises:
            DBStorageSQLStoreError: If there is an error inserting data or creating the table.
        """
        with self.conn_lock:
            if not self.conn:
                self._connect()

        if create_table:
            column_types: dict[str, str] = {}
            for k, v in data.items():
                if isinstance(v, int):
                    column_types[k] = "INTEGER"
                elif isinstance(v, float):
                    column_types[k] = "REAL"
                elif isinstance(v, str):
                    column_types[k] = "TEXT"
                elif isinstance(v, bytes):
                    column_types[k] = "BLOB"
                else:
                    raise DBStorageSQLStoreError(f"Unsupported data type for column {k}: {type(v)}")
            self.create_table(table, column_types)

        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        if on_conflict not in (None, "replace", "ignore"):
            raise DBStorageSQLStoreError(f"Unsupported on_conflict value: {on_conflict!r}, use 'replace' or 'ignore'")
        insert = f"INSERT OR {on_conflict.upper()}" if on_conflict else "INSERT"
        sql = f"{insert} INTO {table} ({columns}) VALUES ({placeholders})"

        try:
            with self.conn_lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute(sql, tuple(data.values()))
                # An ignored record inserts no row: lastrowid would still be the one of an earlier insert
                row_id = (cursor.lastrowid or 0) if cursor.rowcount > 0 else 0
                self._commit(conn)
            logger.debug(f"Inserted data into {table}: {data}")
            return row_id
        except sqlite3.Error as e:
            self._rollback_failed()
            raise DBStorageSQLStoreError(f"Error inserting data into {table}: {e}")

    def read(
        self,
        table: str,
        columns: list[str] | None = None,
        condition: str | None = None,
        order_by: str | None = None,
        limit: int | None = -1,
        args: tuple[Any, ...] | None = None,
    ) -> list[dict[str, Any]]:
        """Get data from the specified table with flexible filtering options. If the table does not exist, it returns an empty list.

        Args:
            table (str): Name of the table to read from.
            columns (Optional[list[str]], optional): List of column names to select.
                An item can also be an SQL expression, such as "COUNT(*) AS total" or
                "MAX(id) AS last_id": name it with AS to read it from the result.
                If None, selects all columns. Defaults to None.
            condition (Optional[str], optional): WHERE clause for filtering results
                (e.g., "age > 18"). Use `?` for the values and pass them in `args`
                (e.g., "age > ?"). Defaults to None.
            order_by (Optional[str], optional): ORDER BY clause for sorting results
                (e.g., "name ASC"). Defaults to None.
            limit (Optional[int], optional): Maximum number of rows to return.
                Use -1 or None for no limit. Defaults to -1.
            args (Optional[tuple], optional): Values for the `?` placeholders of the condition,
                in order. Values passed this way are never read as SQL, so text coming from a
                user or a request cannot change the query. Defaults to None.

        Returns:
            list[dict[str, Any]]: List of dictionaries representing the rows, where each
                dictionary maps column names to their values. Empty list if table doesn't exist.

        Raises:
            DBStorageSQLStoreError: If there is an error reading data from the table.
        """
        with self.conn_lock:
            if not self.conn:
                self._connect()

        selected = ", ".join(columns) if columns else "*"
        sql = f"SELECT {selected} FROM {table}"
        if condition:
            sql += f" WHERE {condition}"
        if order_by:
            sql += f" ORDER BY {order_by}"
        if limit is not None and limit > 0:
            sql += f" LIMIT {limit}"

        try:
            with self.conn_lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute(sql, args or ())
                col_names = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
            return [dict(zip(col_names, row)) for row in rows]
        except sqlite3.Error as e:
            if "no such table" in str(e):
                return []
            raise DBStorageSQLStoreError(f"Error reading data from {table}: {e}")

    def update(self, table: str, data: dict[str, Any], condition: str | None = "", args: tuple[Any, ...] | None = None) -> int:
        """Update data or records in the specified table.

        Args:
            table (str): Name of the table to update.
            data (dict[str, Any]): Dictionary of column names and their new values.
            condition (Optional[str], optional): WHERE clause for filtering which records
                to update (e.g., "id = 1", or "id = ?" with the value in `args`).
                If empty, updates all records. Defaults to "".
            args (Optional[tuple], optional): Values for the `?` placeholders of the condition,
                in order. Defaults to None.

        Returns:
            int: The number of records updated, 0 if none matched the condition.

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
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute(sql, tuple(data.values()) + tuple(args or ()))
                updated = cursor.rowcount
                self._commit(conn)
            logger.debug(f"Updated data in {table}: {data} where {condition}")
            return updated
        except sqlite3.Error as e:
            self._rollback_failed()
            raise DBStorageSQLStoreError(f"Error updating data in {table}: {e}")

    def delete(self, table: str, condition: str | None = "", args: tuple[Any, ...] | None = None) -> int:
        """Delete data from the specified table. If no condition is provided, this will delete ALL records from the table.

        Args:
            table (str): Name of the table to delete from.
            condition (Optional[str], optional): WHERE clause for filtering which records
                to delete (e.g., "age < 18", or "age < ?" with the value in `args`).
                If empty, deletes all records. Defaults to "".
            args (Optional[tuple], optional): Values for the `?` placeholders of the condition,
                in order. Defaults to None.

        Returns:
            int: The number of records deleted, 0 if none matched the condition.

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
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute(sql, args or ())
                deleted = cursor.rowcount
                self._commit(conn)
            logger.debug(f"Deleted data from {table} where {condition}")
            return deleted
        except sqlite3.Error as e:
            self._rollback_failed()
            raise DBStorageSQLStoreError(f"Error deleting data from {table}: {e}")

    def execute_sql(self, sql: str, args: tuple[Any, ...] | None = None) -> list[dict[str, Any]] | None:
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
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute(sql, args or ())
                logger.debug(f"Executing SQL command: {sql} with args {args}")

                first_row = cursor.fetchone()
                if first_row is None:
                    self._commit(conn)
                    return None
                else:
                    rows = [first_row] + cursor.fetchall()
                    self._commit(conn)
                    col_names = [description[0] for description in cursor.description]
            return [dict(zip(col_names, row)) for row in rows]
        except sqlite3.Error as e:
            self._rollback_failed()
            raise DBStorageSQLStoreError(f"Error executing SQL command: {e}")

    def create_or_replace_table(self, table: str, columns: dict[str, str], force_drop_table: bool = False) -> None:
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
            if self._transaction_depth > 0:
                # It opens its own transaction and may roll it back, which would undo the caller's changes too
                raise DBStorageSQLStoreError("create_or_replace_table() cannot be used inside a transaction")
            if not self.conn:
                self._connect()

            conn = self._get_connection()

            cursor = conn.cursor()
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
                conn.execute("BEGIN")
                # Table exists: get current schema
                cursor.execute(f"PRAGMA table_xinfo({table})")
                existing_cols = {row["name"]: row["type"] for row in cursor.fetchall()}  # {col_name: col_type}
            # Find columns to add and remove
            to_add = {col: dtype for col, dtype in columns.items() if col not in existing_cols}
            to_remove = [col for col in existing_cols if col not in columns]
            type_changed = [col for col in columns if col in existing_cols and columns[col].upper() != existing_cols[col].upper()]
            logger.debug(f"Columns to add: {to_add}, to remove: {to_remove}, type changed: {type_changed}")
            dropped: list[str] = []
            failed_drop: list[str] = []
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
                    conn.rollback()
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
                        conn.rollback()
                    logger.error(f"Error adding column {col} to {table}: {e}")
                    raise DBStorageSQLStoreError(f"Error adding column {col} to {table}: {e}")
            # Save the new schema: until the commit it is only pending on this connection
            with self.conn_lock:
                conn.commit()
            logger.debug(f"Table {table} schema aligned. Added columns: {list(to_add.keys())}, dropped: {dropped}")
        except Exception as exc:
            logger.error(f"Schema change failed for table {table}: {exc}")
            logger.debug("Rolling back transaction")
            with self.conn_lock:
                conn.rollback()
            if force_drop_table:
                logger.warning(f"Dropping and recreating table {table} due to force_drop_table=True")
                self.drop_table(table)
                self.create_table(table, columns)
                logger.debug(f"Table {table} dropped and recreated with new schema: {columns}")
            elif isinstance(exc, sqlite3.Error):
                raise DBStorageSQLStoreError(f"Error updating the schema of table {table}: {exc}") from exc
            else:
                raise
