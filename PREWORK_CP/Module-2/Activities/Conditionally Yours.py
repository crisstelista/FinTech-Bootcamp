"""
Conditionally Yours

Pseudocode:

calculate percent increase as ((current price - original price)/original price) * 100

if percent increase is positive (greater than threshold to sell)
    then it means that the price has increased
else if percent increase is negative (less than threshold to buy)
    then it means that the price has decreased
else, meaning the percent increase is 0, it means that the price did not increase or decrease at all.

"""

original_price = 360.35
current_price = 293.33
threshold_to_buy = -10
threshold_to_sell = 20
balance = 1000
recommendation = 'buy'

increase = current_price - original_price
percent_increase = increase/original_price * 100


print("original price: ", original_price)
print("current price: ", current_price)
print("increase: ", increase)
print("percent increase: ", percent_increase)

if (percent_increase>threshold_to_sell):
    recommendation = 'sell'
elif (percent_increase<threshold_to_buy):
    recommendation = 'buy'
else:
    recommendation = 'hold'

print("Recommendation: " + recommendation)