def most_frequent(List):
    counter = 0
    # element = List[0]
    for i in List:
        curr_frequency = List.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            element = i
    return element

num_list = [1, 2, 2, 3]
print(most_frequent(num_list))

num_list = [1, '-2', '-2', 3, 3, '-2', -3]
print(most_frequent(num_list))
