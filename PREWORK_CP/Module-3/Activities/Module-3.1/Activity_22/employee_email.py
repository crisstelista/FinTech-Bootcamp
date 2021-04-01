import csv
import os

filepath = "employees.csv"
csvpath = "output/result.csv"

new_employee_data = []

# Read data into dictionary and create a new email field
with open(filepath, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        first_name = row["first_name"]
        last_name = row["last_name"]
        email = f"{first_name}.{last_name}@example.com"
        new_employee_data.append(
            {
                "first_name": row["first_name"],
                "last_name": row["last_name"],
                "ssn": row["ssn"],
                "email": email
            }
        )


filename = os.path.split(filepath)

with open(csvpath, "w") as csvfile:
    fieldnames = ["last_name", "first_name", "ssn", "email"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(new_employee_data)
