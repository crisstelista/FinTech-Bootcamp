# -*- coding: utf-8 -*-
"""Student Do: Sales Analysis.

This script will use the Pathlib library to set the file path,
use the csv library to read in the file, and iterate over each
row of the file to calculate customer sales averages.
"""

# @TODO: Import the pathlib and csv library
from pathlib import Path
import csv


# @TODO: Set the file path
input_file = Path("sales.csv")

# Initialize list of records
records = []
analysis={}
count = 0
revenue=0

# @TODO: Open the csv file as an object
with open(input_file, 'r') as file:
    csv_reader = csv.reader(file, delimiter=',')
    header = next(csv_reader)
    print(f"{header} < ---- Header")

    # @TODO: Append the column 'Average' to the header
    header.append('average')
    # @TODO: Append the header to the list of records
    records.append(header)
    # @TODO: Read each row of data after the header
    for row in csv_reader:

        # @TODO: Print the row
        print(row)
        # @TODO:
        # Set the 'name', 'count', 'revenue' variables for better
        # readability, convert strings to ints for numerical calculations
        name = row[0]
        revenue = int(row[2])
        count = int(row[1])

        # @TODO: Calculate the average (round to the nearest 2 decimal places)
        average = revenue/count
        print(average)

        if name not in analysis.keys():
            analysis['name'] = {"count":count, "revenue" : revenue}
        else:
            analysis['name']['count'] += count
            analysis['name']['revenue'] += revenue

        # @TODO: Append the average to the row
        row.append(average)
        # @TODO: Append the row to the list of records

# @TODO: Set the path for the output.csv
output_path = "CSV Output.csv"

# @TODO:
# Open the output path as a file and pass into the 'csv.writer()' function
# Set the delimiter/separater as a ','
with open(output_path, "w") as csv_file:
    csvwriter = csv.writer(csv_file, delimiter=',')
    csvwriter.writerow(header)

    for key in analysis:
        print(analysis)
        csvwriter.writerow(
             name
        )
