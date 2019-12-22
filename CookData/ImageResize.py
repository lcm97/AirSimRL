from glob import glob
from PIL import Image
import os

img_path = glob("C:/dataset/AirSimData/320/images_raw/*.png")
path_save = "C:/dataset/AirSimData/32/images/"
a = range(0,len(img_path))
print(len(img_path))
i = 0
for file in img_path:
    name = os.path.join(path_save, "%d.png"%a[i])
    print(file)
    im = Image.open(file)
    print(im)
    im.thumbnail((32,32))
    print(im.format, im.size, im.mode)
    im.save(name,'PNG')
    i+=1