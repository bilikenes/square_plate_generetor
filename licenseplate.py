from PIL import Image, ImageDraw, ImageFont
import string
import random
random.seed(4141)

def rnd_str(l):
	return ''.join(random.choices(string.ascii_uppercase + string.digits, k=l))

font = ImageFont.truetype('resources/din1451alt.ttf', size=310)
color = 'rgb(0, 0, 0)' # black

for i in range(1,10):
	part1 = "34  KLs"
	part2 = "  654"

	image = Image.open('resources/CZ-number-plate-2004-US.png')
	draw = ImageDraw.Draw(image)

	draw.text((300, 0), part1, fill=color, font=font)
	draw.text((250, 350), part2, fill=color, font=font)
	file_name = ""
	image.save('test2.png')