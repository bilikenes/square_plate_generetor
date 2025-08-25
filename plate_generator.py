import random
import string

def generate_turkish_plate():
    letters_list = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","R","S","T","U","V","Y","Z"]
    first_numbers = f"{random.randint(1,81):02d}"
    letters = ''.join(random.choices(letters_list, k=random.randint(1,3)))

    if(len(letters) == 1):
        last_numbers = f"{random.randint(1,9999):04d}"
    elif(len(letters) == 2):
        last_numbers = f"{random.randint(1,999):03d}"
    else:
        last_numbers = f"{random.randint(1,999):03d}"
    
    part1 = f"{first_numbers}{letters}"
    part2 = f"{last_numbers}"
    
    return part1 + part2

with open(r"C:\Users\PC\Desktop\square_plates\plates.txt", "a") as f:
    for _ in range(100000):
        f.write(generate_turkish_plate() + "\n")
