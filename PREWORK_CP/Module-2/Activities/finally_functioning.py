"""
Determine the Compound Annual Growth Rate for an investment
"""

# Declare a variable beginning_balance as a float
beginning_balance = 29000.00

# Declare a variable ending_balance as float
ending_balance = 45000.10

# Declare a variable years as an float
years = 1.0

# Define a function called calculate_compound_growth_rate with three arguments: beginning_balance, ending_balance, years. Function should output growth_rate.
def calculate_compound_growth_rate(beginning_balance, ending_balance, years):
    CAGR = pow((ending_balance/beginning_balance), 1/years) - 1
    return round(CAGR,2)

# Call calculate_compound_growth_rate using beginning_balance, ending_balance, and years. Capture as year_one_growth.
year_one_growth = calculate_compound_growth_rate(beginning_balance, ending_balance, years)
print('year_one_growth = ', year_one_growth)
# Update beginning_balance and ending balance for year two, and then execute calculate_compound_growth_rate
beginning_balance = 45000.00
ending_balance = 47000.00
year_two_growth = calculate_compound_growth_rate(beginning_balance, ending_balance, years)
print('year_two_growth = ', year_two_growth)
# Call calculate_compound_growth_rate using beginning_balance, ending_balance, and years. Capture as year_two_growth.
beginning_balance = 47000.00
ending_balance = 48930.00
year_three_growth = calculate_compound_growth_rate(beginning_balance, ending_balance, years)
print('year_three_growth = ', year_three_growth)

# Use Python round() function to round year_one_growth, year_two_growth, and year_three_growth. Capture these as new variables.

# Print year_one_growth, year_two_growth, year_three_growth as percents using string formatting
print(f'year_one_growth = ', "{:.2f}%".format(year_one_growth*100))
print(f'year_two_growth = ', "{:.2f}%".format(year_two_growth*100))
print(f'year_three_growth = ', "{:.2f}%".format(year_three_growth*100))
# Challenge

# Create a global, empty list
growth_rates=[]

# Define a function called
def calculate_compound_growth_rate_list(beginning_balance, ending_balance, years):
    growth_rates.append(calculate_compound_growth_rate(beginning_balance, ending_balance, years))
    return growth_rates


# Call calculate_compound_growth_rate_list and populate growth_rates with 2016 values (beginning_balance and ending_balance)
beginning_balance = 29000.00
ending_balance = 45000.10
calculate_compound_growth_rate_list(beginning_balance, ending_balance, years)

# Call calculate_compound_growth_rate_list and populate growth_rates with 2017 values (beginning_balance and ending_balance)
beginning_balance = 45000.00
ending_balance = 47000.00
calculate_compound_growth_rate_list(beginning_balance, ending_balance, years)

# Call calculate_compound_growth_rate_list and populate growth_rates with 2018 values (beginning_balance and ending_balance)
beginning_balance = 47000.00
ending_balance = 48930.00
calculate_compound_growth_rate_list(beginning_balance, ending_balance, years)

# Print growth_rates list
print("Growth rates: ", growth_rates)
