DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS images;
DROP TABLE IF EXISTS labels;
DROP TABLE IF EXISTS trains;
DROP TABLE IF EXISTS models;

CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password BLOB NOT NULL
);

CREATE TABLE images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    data_id TEXT,
    filename TEXT,
    image_data BLOB
);

CREATE TABLE labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    data_id TEXT,
    image_id INTEGER,
    label INTEGER,
    FOREIGN KEY (image_id) REFERENCES images (id)
);

CREATE TABLE trains (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trained_model_file_path TEXT,
    loss REAL,
    accuracy REAL,
    optimizer_type TEXT,
    lr REAL,
    momentum REAL,
    epochs INTEGER,
    train_time_secs REAL,
    train_time_mins REAL
);

CREATE TABLE models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    model_path TEXT
);