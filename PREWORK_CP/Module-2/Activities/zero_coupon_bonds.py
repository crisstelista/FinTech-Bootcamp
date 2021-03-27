# -*- coding: utf-8 -*-
"""
Zero-Coupon Bond Valuation.

This script will calculate the present value of zero-coupon bonds, compare the present value to the price of the bond, and determine the corresponding action (buy, not buy, neutral).
"""

# @TODO: Create a function to calculate present value

def present_value(future_value, discount_rate, compounding_periods, years):
    pv = future_value/((1+(discount_rate/compounding_periods))**(compounding_periods*years))
    return pv

# Intialize the zero-coupon bond parameters, assume compounding period is equal to 1
price = 700
future_value = 1000
discount_rate = .1
compounding_periods = 1
years = 5

# @TODO: Call the calculate_present_value() function and assign to a variables
bond_value = present_value(future_value, discount_rate, compounding_periods, years)
print('bond value: ', round(bond_value,2))

# @TODO: Determine if the bond is worth it
if (bond_value > price):
    print('purchase the bond. Price is ', price, ' and bond value is ', round(bond_value,2))
elif (bond_value < price):
    print(' do not purchase the bond. Price is ', price, ' and bond value is ', round(bond_value,2))
else:
    print('it is up to you if you buy it or not. Price is ', price, ' and bond value is ', round(bond_value,2))