import os
import shutil
import numpy
import cv2
def init():
    os.mkdir("TrainData")
    for i in range(10):
        os.mkdir("TrainData\\"+str(i))
        os.mkdir("TrainData\\" + str(i)+"\\T")
        os.mkdir("TrainData\\" + str(i)+"\\F")
def divide():
    for i in range(1,2163):
        if os.path.isfile("Faces\\{}.jpg".format(str(i))):
            shutil.copyfile("Faces\\{}.jpg".format(str(i)),"TrainData\\{}\\T\\{}.jpg".format(str(i%10),str(i//10)))
    for i in range(2163,4001):
        if os.path.isfile("Faces\\{}.jpg".format(str(i))):
            shutil.copyfile("Faces\\{}.jpg".format(str(i)),"TrainData\\{}\\F\\{}.jpg".format(str(i%10),str(i//10)))

def sift():
    for file in os.listdir("Faces"):
        img=cv2.imread("Faces\\"+file)
        if img.shape[0]<50 or img.shape[1]<50:
            print("a")
            shutil.copy("Faces\\"+file,"err\\small\\"+file)
            os.remove("Faces\\"+file)



