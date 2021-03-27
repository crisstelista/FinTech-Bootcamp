# -*- coding: utf-8 -*-
"""
Student Do: Trading Log.

This script demonstrates how to perform basic analysis of trading profits/losses
over the course of a month (20 business days).
"""

# @TODO: Initialize the metric variables
count = 0
total = 0
average = 0
minimum = 0
maximum = 0

# @TODO: Initialize lists to hold profitable and unprofitable day profits/losses
profitable_days=[]
unprofitable_days=[]

# List of trading profits/losses
trading_pnl = [ -224,  352, 252, 354, -544,
                -650,   56, 123, -43,  254,
                 325, -123,  47, 321,  123,
                 133, -151, 613, 232, -311 ]

# @TODO: Iterate over each element of the list
for i in trading_pnl:
    #@TODO: Cumulatively sum up the total and count
    total += i
    count += 1

    # @TODO: Write logic to determine minimum and maximum values
    if minimum == 0:
        minimum = i
    elif i<minimum:
        minimum = i

    if i>maximum:
        maximum = i


    # @TODO: Write logic to determine profitable vs. unprofitable days
    if(i<0):
        unprofitable_days.append(i)
    elif(i>0):
        profitable_days.append(i)



# @TODO: Calculate the average
average = round(total/count,2)

# @TODO: Calculate count metrics
count

# @TODO: Calculate percentage metrics
percentage_profitable_days = 100 * len(profitable_days)/len(trading_pnl)
percentage_unprofitable_days = 100 * len(unprofitable_days)/len(trading_pnl)


# @TODO: Print out the summary statistics
print('count', count)
print('total', total)
print('average', average)
print('minimum', minimum)
print('maximum', maximum)
print('percentage of unprofitable_days', percentage_unprofitable_days)
print('percentage of profitable_days', percentage_profitable_days)