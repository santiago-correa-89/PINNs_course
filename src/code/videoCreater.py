# importing libraries
import cv2 
import numpy
from PIL import Image 
  
img=[]

for i in range(199):
    img.append(cv2.imread(r"src/data/fig/Presion/P_" + str(i) + ".png"))

height,width,layers=img[1].shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video=cv2.VideoWriter(r"src/data/fig/Presion/Presion.avi", fourcc, 10, (width,height))

for j in range(199):
    video.write(img[j])

video.release()