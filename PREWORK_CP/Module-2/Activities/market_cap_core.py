# -*- coding: utf-8 -*-
"""
Student Activity: Market Capitalization.

This script showcases the use of Python Dicts to determine the
bank names associated with the corresponding market cap ranges.
"""

# Banks and Market Caps
#-----------------------
# JP Morgan Chase: 327
# Bank of America: 302
# Citigroup: 173
# Wells Fargo: 273
# Goldman Sachs: 87
# Morgan Stanley: 72
# U.S. Bancorp: 83
# TD Bank: 108
# PNC Financial Services: 67
# Capital One: 47
# FNB Corporation: 4
# First Hawaiian Bank: 3
# Ally Financial: 12
# Wachovia: 145
# Republic Bancorp: .97

# @TODO: Initialize a dictionary of banks and market caps (in billions)
banks = {'JP Morgan Chase': 327, 'Bank of America': 302, 'Citigroup': 173, 'Wells Fargo': 273,
         'Goldman Sachs': 87, 'Morgan Stanley': 72, 'U.S. Bancorp': 83, 'TD Bank': 108,
         'PNC Financial Services': 67, 'Capital One': 47, 'FNB Corporation': 4, 'First Hawaiian Bank': 3,
         'Ally Financial': 12, 'Wachovia': 145, 'Republic Bancorp': 0.97
         }

# @TODO: Change the market cap for 'Citigroup'
banks['Citigroup'] = 170

# @TODO: Add a new bank and market cap pair
banks['American Express'] = 33

# @TODO: Remove a bank from the dictionary
del banks['Wachovia']

# @TODO: Print the modified dictionary
print(banks)

# @TODO: Challenge
total_market_capitalizaion = sum(banks.values())
total_banks = len(banks)
averge_market_capitalization = round(total_market_capitalizaion/total_banks,2)

def GetKey(val):
   for key, value in banks.items():
      if val == value:
         return key
      else:
          return "key doesn't exist"

largest_bank = GetKey(max(banks.values()))
smallest_bank = GetKey(min(banks.values()))

print('total market capitalization', total_market_capitalizaion)
print('total banks', total_banks)
print('average market capitalization', averge_market_capitalization)
print('largest bank', largest_bank)
print('smallest bank', smallest_bank)

mega_cap=[]
large_cap=[]
mid_cap=[]
small_cap=[]
for key, value in banks.items():
    if value >= 300:
        mega_cap.append(key)
    elif value >= 10 and value < 300:
        large_cap.append(key)
    elif value >= 2 and value < 10:
        mid_cap.append(key)
    elif value >= .300 and value < 2:
        small_cap.append(key)
    else:
        continue

print('mega cap', mega_cap)
print('large cap', large_cap)
print('mid cap', mid_cap)
print('small cap', small_cap)

# dir(banks)
# help(banks)