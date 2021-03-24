original_price = 198.87
current_price = 254.32

increase = current_price - original_price
percent_increase = increase/original_price * 100

print("original price: ", original_price)
print("current price: ", current_price)
print("increase: ", increase)
print("percent increase: ", percent_increase)

message = f"The price has been increased by: {round(percent_increase,2)}%"
message_2 = f"Apples's stock price increased by {percent_increase:.2f}%"

print("Apples's stock price increased by", "{:.2f}%".format(percent_increase))
print(message)
print(message_2)
