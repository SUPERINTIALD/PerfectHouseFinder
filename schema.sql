CREATE TABLE users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    password TEXT
);
CREATE TABLE session_tokens (
    session_token TEXT PRIMARY KEY,
    user_id INTEGER
);
CREATE TABLE threads (
    thread_id INTEGER PRIMARY KEY AUTOINCREMENT,
    author TEXT,
    title TEXT,
    body TEXT
);
CREATE TABLE replies (
    reply_id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id INTEGER,
    author TEXT,
    body TEXT
);