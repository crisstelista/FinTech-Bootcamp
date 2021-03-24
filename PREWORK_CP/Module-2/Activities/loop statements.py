string = 'Give me a'
cheer = "Python"
string2 = "What does that spell?!"
string3 = "Woohoo! Go " + cheer + '!' 
string4 = ("*\\0/*")

for letter in cheer:
    letter = letter +'!'
    print(string, letter, end='\n')
    print(letter)
print(string2)
print(string3)
print(string4*3)
print('\n')

for index, letter in enumerate(cheer):
    letter = letter +'!'
    print(string, letter, end='\n')
    print(letter)
print(string2)
print(string3)
print(string4*3)
print('\n')

index = 0
while index < len(cheer):
    print(string, cheer[index], '!', end='\n')
    print(cheer[index], '!')
    index += 1
    # index = input("To run again. Enter '0'")
print(string2)
print(string3)
print(string4*3)