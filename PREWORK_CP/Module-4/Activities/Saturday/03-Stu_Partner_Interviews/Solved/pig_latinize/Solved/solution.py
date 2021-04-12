#you move the first letter to the end of the word
#then ad ay. So, "fish" in Pig Latin is "ish-f-ay."

def pig_latinize(text):
    text = text.split()
    result=''

    for element in text:
        first_letter = element[0]
        result += element[1:] + str(first_letter) + "ay "
    print(result)

pig_latinize("fish")
pig_latinize("your car is very nice")