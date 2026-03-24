from libsql_client import create_client_sync
import os

_client = None

def get_db():
    global _client

    if _client is None:
        url = os.getenv("TURSO_DB_URL")
        token = os.getenv("TURSO_AUTH_TOKEN")

        if not url or not token:
            raise Exception("Turso ENV not set")

        _client = create_client_sync(
            url=url,
            auth_token=token
        )

    return _client


# 🔥 UNIVERSAL SAFE QUERY
def execute_query(query, params=None):
    try:
        db = get_db()

        result = db.execute(query, params or [])

        # ✅ HANDLE ALL CASES
        if result is None:
            return []

        if hasattr(result, "rows") and result.rows:
            return [dict(r) for r in result.rows]

        return []

    except Exception as e:
        print("DB ERROR:", e)
        return []
