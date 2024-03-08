DROP TABLE IF EXISTS images;
DROP TABLE IF EXISTS labels;
DROP TABLE IF EXISTS models;

CREATE TABLE images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    image_data BLOB
);

CREATE TABLE labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id INTEGER,
    label INTEGER,
    FOREIGN KEY (image_id) REFERENCES images (id)
);

CREATE TABLE models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT,
    model_file_path TEXT
);
