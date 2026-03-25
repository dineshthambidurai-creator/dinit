"""
turso_db.py — Turso (libsql_client) database helper
Fixed: Row → dict conversion using column names from result.columns
"""

from libsql_client import create_client_sync
import os

_client = None


def get_db():
    global _client

    if _client is None:
        url   = os.getenv("TURSO_DB_URL")
        token = os.getenv("TURSO_AUTH_TOKEN")

        if not url or not token:
            raise Exception(
                "Turso ENV not set. "
                "Please set TURSO_DB_URL and TURSO_AUTH_TOKEN environment variables."
            )

        _client = create_client_sync(url=url, auth_token=token)

    return _client


def _rows_to_dicts(result):
    """
    Convert libsql_client ResultSet rows → list[dict].

    libsql_client returns:
      result.columns  — tuple/list of column name strings
      result.rows     — list of Row objects (each is a sequence of values)

    dict(row) FAILS because Row is not a key-value sequence.
    We must zip(columns, row) ourselves.
    """
    if result is None:
        return []

    columns = getattr(result, "columns", None)
    rows    = getattr(result, "rows", None)

    if not rows:
        return []

    if not columns:
        # Fallback: return list of tuples if no column info
        return [tuple(r) for r in rows]

    # ✅ Correct conversion
    return [dict(zip(columns, row)) for row in rows]


def execute_query(query, params=None):
    """
    Execute a SELECT query and return list of dicts.
    Returns [] on any error (error is printed).
    """
    try:
        db     = get_db()
        result = db.execute(query, params or [])
        return _rows_to_dicts(result)

    except Exception as e:
        print("DB ERROR:", e)
        return []


def execute_write(query, params=None):
    """
    Execute an INSERT/UPDATE/DELETE.
    Returns the result object (for last_insert_rowid etc.), or None on error.
    """
    try:
        db     = get_db()
        result = db.execute(query, params or [])
        return result

    except Exception as e:
        print("DB ERROR:", e)
        return None


def execute_batch(queries):
    """
    Execute a list of (query, params) tuples in one batch.
    queries: list of (sql_string, params_list) tuples
    """
    try:
        db = get_db()
        db.batch(queries)
    except Exception as e:
        print("DB BATCH ERROR:", e)
