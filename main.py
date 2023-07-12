import math
import cv2 as cv
import numpy as np


# Return RGB Array of the position
def getRgbArray(posX, posY, image):
    if posY >= height or posX >= width or posY < 0 or posX < 0:
        return [0, 0, 0]
    else:
        return image[posY][posX]


def getLenthLs(size):
    lenthLs = []
    for i in range(-size,size+1):
        lenthLs.append(i)
    return lenthLs


def gaussian_2d(x, y, sigma):
    up = -(x**2 + y**2)/(2*sigma**2)
    weight = 1/ (2*math.pi*sigma**2)
    return weight*np.exp(up)


def gausWindow(windowSize, sigma):
    distance = windowSize // 2

    gausLs = []
    for i in range(0,windowSize):
        tempLs = []
        for j in range(0,windowSize):
            tempLs.append(0)
        gausLs.append(tempLs)

    for ySize in getLenthLs(distance):
        for xSize in getLenthLs(distance):
            gausLs[xSize+distance][ySize+distance] = gaussian_2d(xSize, ySize, sigma)
    return gausLs


def getNeighLs(posX, posY, windowSize, image):
    neighLs = []
    distance = windowSize // 2
    for ySize in getLenthLs(distance):
        for xSize in getLenthLs(distance):
            rgbArray = getRgbArray((posX + ySize), (posY + xSize), image)
            weight = gausLs[xSize+distance][ySize+distance]

            optixArray = list(map(lambda x: float(x) * float(weight), rgbArray))
            neighLs.append(optixArray)
    return neighLs


def calNewRGB(posX, posY, windowSize, image):
    neighLs = getNeighLs(posX, posY, windowSize, image)
    newR = 0
    newG = 0
    newB = 0
    for i in range(0,len(neighLs)):
        newR += neighLs[i][0]
        newG += neighLs[i][1]
        newB += neighLs[i][2]
    newRgbArray = [newR, newG, newB]
    return newRgbArray


img_bgr = cv.imread('pic7.png')
img = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

height = len(img)
width = len(img[0])
print(f"The Raw picture is Height: {height}, Width: {width}.")

# Attention: The windowsSize should be around the [6*sigma]*[6*sigma]
windowSize = 13
gausLs = gausWindow(windowSize, 2)

newPicLs = []
for yHeight in range(0,height):
    newPicHeightLs = []
    for xWidth in range(0, width):
        newPicHeightLs.append(calNewRGB(xWidth, yHeight, windowSize, img))
    newPicLs.append(newPicHeightLs)

newImage = cv.cvtColor(np.uint8(newPicLs), cv.COLOR_RGB2BGR)
cv.imwrite('blur_image.png', newImage)
print(f"Your picture is blurred.")