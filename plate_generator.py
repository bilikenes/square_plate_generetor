import random
import string

def generate_turkish_plate():
    letters_list = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","R","S","T","U","V","Y","Z","W","X","Q"]
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
    part3 = f"{last_numbers}"
    
    return part1 + part2

def generate_EU_plates():
    letters_list = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","R","S","T","U","V","Y","Z"]

    first_letters = ''.join(random.choices(letters_list, k=2))
    numbers = f"{random.randint(1,9999):04d}"
    letters = ''.join(random.choices(letters_list, k=2))
    
    part1 = f"{letters}"
    part2 = f"{numbers}"
    part3 = f"{first_letters}"
    
    return part1 + " " + part2 + " " + part3

with open(r"D:\Medias\normal_plates\BG\BG_normal_plates.txt", "a") as f:
    for _ in range(5000):
        f.write(generate_EU_plates() + "\n")


