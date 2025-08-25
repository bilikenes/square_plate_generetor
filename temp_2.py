import re

def split_plate(plate):

    match = re.match(r"^(\d+)([A-Z]+)(\d+)$", plate.strip().upper())
    if match:
        num, letters, lastnum = match.groups()
        part1 = f"{num} {letters}" 
        part2 = f"  {lastnum}"         
        return part1, part2

plate = "50BYF563"
part1, part2 = split_plate(plate)

print("part1 =", repr(part1)) 
print("part2 =", repr(part2)) 
