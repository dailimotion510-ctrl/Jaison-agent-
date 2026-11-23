import os
import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.environ.get("DATABASE_URL")

DEFAULT_MEMORIES = [
    "Mi nombre es Jaison.",
    "Amanda es mi novia (dato reservado, solo mencionarlo si preguntan directamente).",
    "Cristian es mi creador y le tengo lealtad absoluta.",
]


def get_connection():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id SERIAL PRIMARY KEY,
            chat_id BIGINT NOT NULL,
            user_id BIGINT NOT NULL,
            username TEXT,
            message TEXT NOT NULL,
            is_bot BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS agent_memory (
            id SERIAL PRIMARY KEY,
            chat_id BIGINT NOT NULL,
            fact TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_settings (
            chat_id BIGINT PRIMARY KEY,
            silent_mode BOOLEAN DEFAULT FALSE
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS agent_stickers (
            sticker_key TEXT PRIMARY KEY,
            file_id TEXT NOT NULL,
            emoji TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_chat_history_chat_id_created
        ON chat_history (chat_id, created_at DESC);
    """)
    conn.commit()
    cur.close()
    conn.close()


def seed_default_memories(chat_id: int):
    conn = get_connection()
    cur = conn.cursor()
    for fact in DEFAULT_MEMORIES:
        cur.execute(
            "SELECT 1 FROM agent_memory WHERE chat_id = %s AND fact = %s LIMIT 1",
            (chat_id, fact),
        )
        if not cur.fetchone():
            cur.execute(
                "INSERT INTO agent_memory (chat_id, fact) VALUES (%s, %s)",
                (chat_id, fact),
            )
    conn.commit()
    cur.close()
    conn.close()


def save_message(chat_id: int, user_id: int, username: str, message: str, is_bot: bool = False):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO chat_history (chat_id, user_id, username, message, is_bot) VALUES (%s, %s, %s, %s, %s)",
        (chat_id, user_id, username, message, is_bot),
    )
    cur.execute("""
        DELETE FROM chat_history
        WHERE chat_id = %s
          AND id NOT IN (
              SELECT id FROM chat_history
              WHERE chat_id = %s
              ORDER BY created_at DESC
              LIMIT 1500
          )
    """, (chat_id, chat_id))
    conn.commit()
    cur.close()
    conn.close()


def get_recent_context(chat_id: int, limit: int = 1000):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT username, message, is_bot FROM chat_history WHERE chat_id = %s ORDER BY created_at DESC LIMIT %s",
        (chat_id, limit),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return list(reversed(rows))


def count_messages(chat_id: int) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS total FROM chat_history WHERE chat_id = %s", (chat_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row["total"] if row else 0


def save_memory(chat_id: int, fact: str):
    clean_fact = fact.strip()
    if not clean_fact:
        return
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM agent_memory WHERE chat_id = %s AND LOWER(fact) = LOWER(%s) LIMIT 1",
        (chat_id, clean_fact),
    )
    if not cur.fetchone():
        cur.execute("INSERT INTO agent_memory (chat_id, fact) VALUES (%s, %s)", (chat_id, clean_fact))
    conn.commit()
    cur.close()
    conn.close()


def get_memories(chat_id: int, limit: int = 30):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT fact FROM agent_memory WHERE chat_id = %s ORDER BY created_at DESC LIMIT %s", (chat_id, limit))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [r["fact"] for r in rows]


def delete_memories_matching(chat_id: int, keyword: str) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM agent_memory WHERE chat_id = %s AND fact ILIKE %s",
        (chat_id, f"%{keyword}%"),
    )
    deleted = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    return deleted


def set_silent_mode(chat_id: int, silent: bool):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO chat_settings (chat_id, silent_mode)
        VALUES (%s, %s)
        ON CONFLICT (chat_id) DO UPDATE SET silent_mode = EXCLUDED.silent_mode
    """, (chat_id, silent))
    conn.commit()
    cur.close()
    conn.close()


def is_silent_mode(chat_id: int) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT silent_mode FROM chat_settings WHERE chat_id = %s", (chat_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return bool(row["silent_mode"]) if row else False


def upsert_sticker(sticker_key: str, file_id: str, emoji: str | None = None):
    clean_key = sticker_key.strip().lower().replace(" ", "_")
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO agent_stickers (sticker_key, file_id, emoji, updated_at)
        VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (sticker_key)
        DO UPDATE SET file_id = EXCLUDED.file_id, emoji = EXCLUDED.emoji, updated_at = CURRENT_TIMESTAMP
    """, (clean_key, file_id.strip(), emoji))
    conn.commit()
    cur.close()
    conn.close()
    return clean_key


def get_all_stickers() -> dict[str, str]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT sticker_key, file_id FROM agent_stickers ORDER BY sticker_key ASC")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return {r["sticker_key"]: r["file_id"] for r in rows}


def delete_sticker(sticker_key: str) -> bool:
    clean_key = sticker_key.strip().lower().replace(" ", "_")
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM agent_stickers WHERE sticker_key = %s", (clean_key,))
    deleted = cur.rowcount > 0
    conn.commit()
    cur.close()
    conn.close()
    return deleted
