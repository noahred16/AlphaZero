import sqlite3
import numpy as np
import torch
import json

# Connect to a database (creates it if it doesn't exist)

# When you call sqlite3.connect('rl_experience.db'), SQLite checks if a file named rl_experience.db exists in your current working directory.
# If the file doesn't exist, SQLite automatically creates it as a new, empty database file.
# If the file already exists, SQLite simply opens a connection to it.
conn = sqlite3.connect('rl_experience.db')
cursor = conn.cursor()

# Create tables for your RL data
cursor.execute('''
CREATE TABLE IF NOT EXISTS episodes (
    episode_id INTEGER PRIMARY KEY,
    total_reward REAL,
    steps INTEGER,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
)
''')

conn.commit()

# conn.close()

print("Created database and tables if they did not exist.")

# show tables
# Created database and tables if they did not exist.
# Traceback (most recent call last):
#   File "/home/noahred16/github/AlphaZero/db_setup.py", line 27, in <module>
#     cursor.execute("SHOW TABLES")
# sqlite3.OperationalError: near "SHOW": syntax error
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables in the database:")
for table in tables:
    print(table[0])
# Close the connection
conn.close()
