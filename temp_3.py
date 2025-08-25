from PIL import Image, ImageDraw, ImageFont
import re
import os

def split_plate(plate):
    match = re.match(r"^(\d+)([A-Z]+)(\d+)$", plate.strip().upper())
    if match:
        num, letters, lastnum = match.groups()
        if(len(letters) == 1):
            part1 = f"{num}    {letters} "
            part2 = f"{lastnum}"
        elif(len(letters) == 2):
            part1 = f"{num} {letters} "
            part2 = f"{lastnum}"
        else:
            part1 = f"{num} {letters}"
            part2 = f"{lastnum}"

        
        return part1, part2
    else:
        return None, None

font = ImageFont.truetype('resources/din1451alt.ttf', size=310)
color = 'rgb(0, 0, 0)'

output_dir = "plates/generated_plates_4"
os.makedirs(output_dir, exist_ok=True)

with open("plates/plates_5.txt", "r", encoding="utf-8") as f:
    plates = f.read().splitlines()

for idx, plate in enumerate(plates, start=1):

    part1, part2 = split_plate(plate)
    image = Image.open('resources/CZ-number-plate-2004-US.png')
    draw = ImageDraw.Draw(image)

    bbox1 = draw.textbbox((0, 0), part1, font=font)
    text_width1 = bbox1[2] - bbox1[0]

    max_x1 = 1100 
    x1 = max_x1 - text_width1
    y1 = 0

    draw.text((x1, y1), part1, fill=color, font=font)

    bbox2 = draw.textbbox((0, 0), part2, font=font)
    text_width2 = bbox2[2] - bbox2[0]

    plate_center_x = 700 
    x2 = plate_center_x - text_width2 // 2
    y2 = 350

    draw.text((x2, y2), part2, fill=color, font=font)

    file_name = os.path.join(output_dir, f"{plate}.png")
    image.save(file_name)
    

    print(" ok : " + file_name)

print("ok!")