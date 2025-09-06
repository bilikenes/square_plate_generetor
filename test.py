from PIL import Image, ImageDraw, ImageFont
import os

with open(r"D:\Medias\normal_plates\BG\BG_normal_plates.txt", "r", encoding="utf-8") as f:
    plate_list = [line.strip() for line in f.readlines() if line.strip()]

font = ImageFont.truetype("resources/din1451alt.ttf", size=310)
color = "rgb(0, 0, 0)"
count = 0

for plate_text in plate_list:

    img = Image.open("resources/TR-Plate.png")
    draw = ImageDraw.Draw(img)
    
    text_bbox = draw.textbbox((0,0), plate_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    img_width, img_height = img.size

    offset_x = 300
    pos_x = offset_x + (img_width - offset_x - text_width) // 2 - 65
    pos_y = 20

    draw.text((pos_x, pos_y), plate_text, fill=color, font=font)

    output_dir = r"D:\Medias\normal_plates\BG\images"
    os.makedirs(output_dir, exist_ok=True)

    out_name = os.path.join(output_dir, f"{plate_text}.png")
    img.save(out_name)
    count += 1
    print(f"{count} : {plate_text}.png")
