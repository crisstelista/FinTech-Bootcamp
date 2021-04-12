list_first = [1, 2, -3]
list_second = [-3, 2, 1]

def compare_lists(list1, list2):
    if(len(list1) != len(list2)):
        print('The content of the lists do not match, considering their different length')
    else:
        for element in list1:
            if(element in list2):
                print('The element ' + str(element) + ' is belonging to the second list.')
            else:
                print('The element ' + str(element) + ' is missing from the second list.')

compare_lists(list_first, list_second)
