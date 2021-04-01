# Define a function `calculate` that takes in two numbers and adds, subtracts, multiplies, or divides the two numbers.
def switch_function(argument, num_1, num_2):
    switcher = {
        "Add": num_1 + num_2,
        "Subtract": num_1 - num_2,
        "Multiply": num_1 * num_2,
        "Divide": num_1 / num_2,
    }
    print(switcher.get(argument, "You did not choose a valid choice."))


def calculate(num_1, num_2):
    # Create a variable `result` and set it to 0.
    result = 0

    # Prompt the user "What do you want to do: Add, Subtract, Multiply or Divide?" and assign the answer to a variable `choice`.
    choice = input("What do you want to do: Add, Subtract, Multiply or Divide? ")

    # Create an if-else statement to perform the proper calculation with the two parameters based on the user's `choice`.
    switch_function(choice, num_1, num_2)
    # Return the calculated `result` variable.
    return result

# Call the `calculate` function and print the results.
calculate(25, 5)


