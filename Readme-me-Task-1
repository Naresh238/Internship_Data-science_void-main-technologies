import statistics
import numpy as np
import pandas as pd
students = []
class Student:
    def __init__(self, sid, name, m1, m2, m3):
        self.sid = sid
        self.name = name
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
def average(self):
        return (self.m1 + self.m2 + self.m3) / 3
def add_student(sid, name, m1, m2, m3):
    try:
        if m1 < 0 or m1 > 100:
            raise ValueError("Invalid mark")
        if m2 < 0 or m2 > 100:
            raise ValueError("Invalid mark")
        if m3 < 0 or m3 > 100:
            raise ValueError("Invalid mark")
        student = Student(sid, name, m1, m2, m3)
        students.append(student)
   except ValueError as e:
        print(e)
def display_students():
    print("STUDENT DETAILS")
    print("--------------------------")

    for s in students:
        print("ID:", s.sid)
        print("Name:", s.name)
        print("Mark 1:", s.m1)
        print("Mark 2:", s.m2)
        print("Mark 3:", s.m3)
        print("Average:", round(s.average(), 2))
        print("--------------------------")
def analysis():
    averages = []
    for s in students:
        averages.append(s.average())
    print("STATISTICAL ANALYSIS")
    print("--------------------------")
    print("Mean:", round(statistics.mean(averages), 2))
    print("Median:", round(statistics.median(averages), 2))
    print("Highest:", round(max(averages), 2))
    print("Lowest:", round(min(averages), 2))
    numbers = np.array(averages)
    print("NumPy Mean:", round(np.mean(numbers), 2))
    print("NumPy Maximum:", round(np.max(numbers), 2))
    print("NumPy Minimum:", round(np.min(numbers), 2))
def save_data():
    try:
        data = []

        for s in students:
            data.append([
                s.sid,
                s.name,
                s.m1,
                s.m2,
                s.m3,
                round(s.average(), 2)
            ])

        df = pd.DataFrame(
            data,
            columns=[
                "ID",
                "Name",
                "Mark1",
                "Mark2",
                "Mark3",
                "Average"
            ]
        )
       df.to_csv("students.csv", index=False)
        print("Data saved to students.csv")
    except Exception as e:
        print("Error:", e)
add_student("S001", "Naresh", 85, 90, 88)
add_student("S002", "Ravi", 75, 80, 70)
add_student("S003", "Priya", 92, 95, 90)
add_student("S004", "Anil", 65, 70, 68)
add_student("S005", "Sneha", 80, 82, 85)
display_students()
analysis()
save_data()
print("Program completed successfully")
