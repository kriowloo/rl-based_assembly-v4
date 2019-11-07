from PIL import Image, ImageDraw, ImageFont
import numpy as np
import csv

def getArray(image):
    im = Image.open(image).convert('RGBA')
    return np.array(im)

def newGroup(img, y, x, groupId, group):
    data = []
    data.append((y,x))
    while len(data) > 0:
        y, x = data[0]
        del data[0]
        if x < 0 or y < 0 or y >= img.shape[0] or x >= img.shape[1] or img[y][x][3] != 255:
            continue
        img[y][x][3] = groupId
        # R: 93 143
        # G: 147 180
        # B: 197 215
        if img[y][x][0] >= 70 and img[y][x][0] <= 160 and img[y][x][1] >= 120 and img[y][x][1] <= 200 and img[y][x][2] >= 170 and img[y][x][2] <= 230:
            group.append((y, x))
            for i in range(-1, 2, 1):
                for j in range(-1, 2, 1):
                    data.append((y-i, x-j))

def getBlobs():
    img = getArray('background.png')
    h = img.shape[0]
    w = img.shape[1]
    groupId = 0
    groups = []
    for x in range(w):
        for y in range(h):
            if img[y][x][3] != 255:
                continue
            
            myGroup = []
            newGroup(img, y, x, groupId, myGroup)
            if len(myGroup) > 1000:
                groupId += 1
                groups.append(myGroup)
    return groups

def paintImage(list_probabilities, text, prefix, img, groups):
    order = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7]
    fnt = ImageFont.truetype('DejaVuSansMono.ttf', 40)
    for k in range(len(list_probabilities)):
        probabilities = list_probabilities[k]
        for i in range(len(groups)):
            group = groups[order[i]]
            prob = probabilities[i-1]
            alpha = int(prob * 255)
            color = [0,0,0,255] if i == 0 else ([0,255,0,255] if prob == 1.0 else ([255,0,0,255] if prob == 0 else [0,0,255,alpha]))
            for y, x in group:
                for j in range(len(color)):
                    img[y][x][j] = color[j]
        im = Image.fromarray(img)
        ImageDraw.Draw(im).text((200,60), text, font=fnt, fill=(0,0,0,255))
        im.save(prefix + "_saida" + ("%05d" % k) + ".png")

def loadProbs(filename):
    actions = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            action = row[0]
            im_input = row[1]
            prob = float(row[2])
            if action != 'None' and im_input == '0':
                list_probs = []
                actions.append((action, list_probs))
            if im_input == '0':
                cur_probs = []
                list_probs.append(cur_probs)
            cur_probs.append(prob)
    return actions
            
 
#list1 = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
#list2 = list1[:]
#list2.sort()
#paintImage([list1, list2], "1")
i = 1
img = getArray('background.png')
groups = getBlobs()
for action, list_probs in loadProbs("saida.txt"):
    paintImage(list_probs, str(action), "%05d" % i, img, groups)
    i += 1
