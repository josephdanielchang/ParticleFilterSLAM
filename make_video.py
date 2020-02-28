import cv2
import numpy as np
import glob

img_array = []

## Specify dataset number and map_type [output_map, output_wall, output_texture]
for filename in glob.glob('results_images/Data_0/output_map/*.png'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

## Specify video name
out = cv2.VideoWriter('Data_0_output_map.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
