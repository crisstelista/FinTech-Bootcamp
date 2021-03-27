greeting =''
customers = [
    { "first_name": "Tom", "last_name": "Bell", "revenue": 0 },
    { "first_name": "Maggie", "last_name": "Johnson", "revenue": 1032 },
    { "first_name": "John", "last_name": "Spectre", "revenue": 2543 },
    { "first_name": "Susy", "last_name": "Simmons", "revenue": 5322 }
]

def create_greeting(first_name, last_name, revenue):
    if(revenue >= 3001):
        greeting = last_name + ', ' + first_name + ' - Platinum member'
    elif (revenue >= 2001 and revenue < 3000):
        greeting = last_name + ', ' + first_name + ' - Gold member'
    elif (revenue >= 1001 and revenue < 2000):
        greeting = last_name + ', ' + first_name + ' - Silver member'
    elif (revenue >= 0 and revenue < 1000):
        greeting = last_name + ', ' + first_name + ' - Bronze member'

    return greeting

for person in customers:
    greeting = create_greeting(person['first_name'], person['last_name'], person['revenue'])
    print(greeting)
