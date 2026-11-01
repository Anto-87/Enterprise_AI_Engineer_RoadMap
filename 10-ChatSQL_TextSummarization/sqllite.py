import sqlite3

## Connect to sqliete database
connection = sqlite3.connect('student.db')

# Create a cursor
cursor = connection.cursor()

# create the table 
table_info = """
CREATE TABLE student (
    name TEXT,
    class TEXT,
    marks INTEGER
);
"""
cursor.execute(table_info)

## Insert some more records
cursor.execute("INSERT INTO student VALUES ('Anto', 'A', 34)")
cursor.execute("INSERT INTO student VALUES ('Chirs', 'A', 44)")
cursor.execute("INSERT INTO student VALUES ('John', 'B', 56)")
cursor.execute("INSERT INTO student VALUES ('Jane', 'A', 79)")
cursor.execute("INSERT INTO student VALUES ('Jill', 'C', 88)")
cursor.execute("INSERT INTO student VALUES ('Joe', 'A', 23)")

## Display all the records 

cursor.execute("SELECT * FROM student")
records = cursor.fetchall()
print(records)

connection.commit()
connection.close()