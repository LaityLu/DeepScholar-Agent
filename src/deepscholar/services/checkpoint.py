import sqlite3
from pathlib import Path
from langgraph.checkpoint.sqlite import (
    SqliteSaver,
)


def create_sqlite_checkpointer(
    db_path: str,
) -> SqliteSaver:

    path = Path(db_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        db_path,
        check_same_thread=False,
    )

    return SqliteSaver(
        connection
    )