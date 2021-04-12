input_list = [12, 33, 41, 2, 61, 32, 75, 43, 67]
max_result = None
min_result = None

for element in input_list:
    if(max_result == None or element>max_result):
        max_result = element
    elif(min_result == None or element<min_result):
        min_result = element
    else:
        continue

print(max_result)
print(min_result)

#second solution
sorted_list = input_list.sort()
min_value = input_list[0]
max_value = input_list[len(input_list)-1]
print(min_value)
print(max_value)

