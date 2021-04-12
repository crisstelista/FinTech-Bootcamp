list_num=[-100, 2, 42, 100]
def max_product(input_list):
    first_max_value = max(input_list)
    input_list.remove(first_max_value)
    second_max_value = max(input_list)

    if(not len(input_list)<2):
        first_min_value = min(input_list)
        input_list.remove(first_min_value)
        second_min_value = min(input_list)

        if(first_max_value*second_max_value <= first_min_value * second_min_value):
            return first_min_value * second_min_value
        else:
            return first_max_value * second_max_value
    else:
        return first_max_value * second_max_value

result = max_product(list_num)
print(result)
print('--------------')

list_num=[-100, -2, -42, -100]
result = max_product(list_num)
print(result)