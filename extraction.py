import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Finding the center of mass of a silhouette
def CenterOfMass(img):
    height, width = img.shape
    CoMR = []
    CoMC = []
    WoR = []
    WoC = []
    for i in range(height):
        Nx = []
        for j in range(width):
            if img[i,j] == 255:
                Nx.append(j)
        if len(Nx) != 0:
            CoMR.append(sum(Nx))
            WoR.append(len(Nx))
    CoMX = sum(CoMR) // sum(WoR)
    for j in range(width):
        Ny = []
        for i in range(height):
            if img[i,j] == 255:
                Ny.append(i)
        if len(Ny) != 0:
            CoMC.append(sum(Ny))
            WoC.append(len(Ny))
    CoMY = sum(CoMC) // sum(WoC)
    return CoMX, CoMY

#Finding the corner coordinates of the rectangle surrounding the silhouette
def Border(img):
    height, width = img.shape
    index = []
    for i in range(height):
        for j in range(width):
            if img[i,j] != 0:
                index.append(j)
                break
    x = min(index)
    index = []
    for i in range(height):
        for j in reversed(range(width)):
            if img[i,j] != 0:
                index.append(j)
                break
    X = max(index)
    index = []
    for j in range(width):
        for i in range(height):
            if img[i,j] != 0:
                index.append(i)
                break
    y = min(index)
    index = []
    for j in range(width):
        for i in reversed(range(height)):
            if img[i,j] != 0:
                index.append(i)
                break
    Y = max(index)
    return x,X,y,Y

#Cropping the silhouette based on the results from Border function
def Crop(img,x,X,y,Y):
    Crop = img[y:Y, x:X]
    return Crop

#Aligning the silhouettes based on their center of mass and averaging them
def Add(old,new,center,black,i):
    center = list(center)
    height, width = old.shape
    x = (width // 2) - center[0]
    y = (height // 2) - center[1]
    height, width = new.shape
    X, Y = (x + width), (y + height)
    a = 1.0/(i + 1)
    b = 1.0 - a
    new = cv2.addWeighted(new, a, old[y:Y, x:X], b, 0.0)
    old = cv2.addWeighted(black, a, old, b, 0.0)
    old[y:Y, x:X] = new
    return old

#Importing the silhouettes, and getting their dimensions
path = glob.glob('GaitDatasetB-silh/001/001/bg-01/090/001-bg-01-090-*.png')
im = cv2.imread(path[0])
y,x,c = im.shape

#Combining the functions
black = np.zeros((y,x), np.uint8)
GEI = black
i = 0
for path in glob.glob('GaitDatasetB-silh/001/001/bg-01/090/001-bg-01-090-*.png'):
    image = cv2.imread(path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    x,X,y,Y = Border(image)
    cropped = Crop(image,x,X,y,Y)
    center = CenterOfMass(cropped)
    GEI = Add(GEI,cropped,center,black,i)
    i += 1

#Visualising the result
x,X,y,Y = Border(GEI)
GEI = Crop(GEI,x,X,y,Y)
cv2.imwrite('Gait_Energy_Image.png', GEI)
GEI = cv2.imread('Gait_Energy_Image.png')
plt.imshow(GEI)
plt.show()
