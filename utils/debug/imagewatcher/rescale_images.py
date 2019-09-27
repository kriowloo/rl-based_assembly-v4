import glob
import sys
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

def watch(score, input_path, scale, output_path, text_width_offset = None, font_size = 100):
    font = ImageFont.truetype("DejaVuSansMono-Bold.ttf", font_size)
    img = Image.open(input_path)
    width, height = img.size
    width *= scale
    height *= scale
    img = img.resize((width, height), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    text = "%.2f" % round(score,2)
    if text_width_offset is None:
        text_width_offset = font.getsize(text)[0]
    draw.text((width - text_width_offset, 0),text,font=font)
    img.save(output_path)

if __name__ == "__main__":
    if len(sys.argv) != 6 and len(sys.argv) != 1:
        print("Usage: python3 " + sys.argv[0] + " <font_size> <scale> <prefix> <extension> <order>")
        print("Example: python3 " + sys.argv[0] + " 100 20 rescaled png 1,2,3")
        print("         - order: 1=episode, 2=action in episode, 3=pm")
        sys.exit(1)
    font_size = 100
    scale = 20
    prefix = "new"
    extension = "png"
    order = [0,1,2]
    if len(sys.argv) == 6:
        font_size = int(sys.argv[1])
        scale = int(sys.argv[2])
        prefix = sys.argv[3]
        extension = sys.argv[4]
        order = [int(x)-1 for x in sys.argv[5].split(",")]
    for file in glob.glob('*.' + extension):
        info = file.split("_")[1:4]
        output_path = prefix
        for x in order:
            output_path += "_" + info[x]
        output_path += "." + extension
        score = int(info[2]) / 100.0
        watch(score, file, scale, output_path, font_size)
