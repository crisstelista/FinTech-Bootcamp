# -*- coding: utf-8 -*-
"""
Student Do: Grocery List.

This script showcases basic operations of Python Lists to help Sally
organize her grocery shopping list.
"""

# @TODO: Create a list of groceries
recipe=['Water', 'Butter', 'Eggs', 'Apples', 'Cinnamon', 'Sugar', 'Milk']



# @TODO: Find the first two items on the list
print("first ingredient is: ", recipe[0])
print("second ingredient is: ", recipe[1])
print('====================================')
# print(recipe[:2])


# @TODO: Find the last five items on the list
print(recipe[-5:])
print('====================================')


# @TODO: Find every other item on the list, starting from the second item
print(recipe[1::2])
print('====================================')



# @TODO: Add an element to the end of the list
recipe = recipe + ['Flour']
# recipe = recipe.append(['Flour'])
print(recipe)
print('====================================')


# @TODO: Changes a specified element within the list at the given index
for i in range(len(recipe)):
    if(recipe[i]=="Apples"):
        recipe[i] = "Gala Apples"
    else:
        continue
print(recipe)
print('====================================')


# @TODO: Calculate how many items you have in the list
print(len(recipe))
# print(recipe[::2]) shows everything except index 2
print('====================================')

# @TODO: Challange
print("where Gala apples is on the list: ", recipe.index('Gala Apples'))
recipe.remove('Sugar')
recipe.pop(recipe.index("Water"))
recipe.pop()
print(recipe)
