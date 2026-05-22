from pathlib import Path
import sqlite3


class Database:
    def __init__(self, path: str = "database/face_security.db"):
        self.path = Path(path)
        self.connection = sqlite3.connect(self.path)
        self._initialize()

    def _initialize(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY,
                name TEXT,
                embedding BLOB
            )
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()
