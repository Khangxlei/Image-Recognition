DROP TABLE IF EXISTS images;
DROP TABLE IF EXISTS labels;

CREATE TABLE images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT
);

CREATE TABLE labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_name TEXT,
    label TEXT,
    FOREIGN KEY (image_name) REFERENCES images (filename)
);
