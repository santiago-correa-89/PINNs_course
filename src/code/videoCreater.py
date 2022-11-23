# importing libraries
import cv2 
import numpy
from PIL import Image 
  
# Checking the current directory path
img=[]

for i in range(199):
    img.append(cv2.imread(r'/src/data/fig/velocidadU/U_' + str(i) + '.png'))

height,width,layers=img[1].shape

video=cv2.VideoWriter(r"/src/data/fig/velocidadU/video.avi", -1, 1, (width,height))

for j in range(0,5):
    video.write(img[j])

cv2.destroyAllWindows()
video.release()