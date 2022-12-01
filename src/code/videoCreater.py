# importing libraries
import cv2 
import numpy
from PIL import Image 
 
def videoCreater(folderIn, folderOut, T): 
    img=[]

    for i in range(T):
        img.append(cv2.imread(folderIn + str(i) + ".png"))

    height,width,layers=img[1].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(folderOut, fourcc, 10, (width,height))

    for j in range(T):
        video.write(img[j])

    video.release()