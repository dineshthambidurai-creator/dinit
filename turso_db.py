from libsql_client import create_client
import os

_client = None

def get_db():
    global _client

    if _client is None:
        _client = create_client(
            url=os.getenv("TURSO_DB_URL"),
            auth_token=os.getenv("TURSO_AUTH_TOKEN")
        )

    return _client
