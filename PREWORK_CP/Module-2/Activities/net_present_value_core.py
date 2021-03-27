# -*- coding: utf-8 -*-
"""Student Activity: Financial Analysis using NPV.

This script will choose the optimal project scenario to
undertake based on max NPV values.
"""

# @TODO: Import the NumPy Financial (numpy_financial) library
import numpy_financial as npf

# Discount Rate
discount_rate = .1

# Initial Investment, Cash Flow 1, Cash Flow 2, Cash Flow 3, Cash Flow 4
cash_flows_conservative = [-1000, 400, 400, 400, 400]
cash_flows_neutral = [-1500, 600, 600, 600, 600]
cash_flows_aggressive = [-2250, 800, 800, 800, 800]

# @TODO: Initialize dictionary to hold NPV return values
npv_dict = {'conservative': npf.npv(discount_rate, cash_flows_conservative),
            'neutral' : npf.npv(discount_rate, cash_flows_neutral),
            'aggresive': npf.npv(discount_rate, cash_flows_aggressive)
            }
# @TODO: Calculate the NPV for each scenario
for key in npv_dict:
    print(key, npv_dict[key])

# @TODO: Manually Choose the project with the highest NPV value
highest_NPV = max(npv_dict)
highest_NPV_value = max(npv_dict.values())
print('the project scenario with the max NPV value is: ', highest_NPV, 'with a NPV of ' , round(highest_NPV_value,2))