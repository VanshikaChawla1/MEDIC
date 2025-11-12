# data/init_db.py
import sqlite3
import os
import json

DB_PATH = "data/medic.db"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INTEGER,
    score REAL,
    triage TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS resources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT UNIQUE,
    available INTEGER,
    shift TEXT
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS bookings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER,
    resources_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

# Seed default resources if not present
defaults = [
    ("icu_beds", 10, "day"),
    ("oxygen_cylinders", 50, "day"),
    ("doctors", 10, "day"),
]

for rtype, avail, shift in defaults:
    c.execute("INSERT OR IGNORE INTO resources (type, available, shift) VALUES (?, ?, ?)", (rtype, avail, shift))

conn.commit()
conn.close()
print("Database initialized ✅ (data/medic.db). Default resources seeded.")
