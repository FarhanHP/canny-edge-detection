import numpy as np
import math
import matplotlib.pyplot as plt

class GrayscaleImage:
    def __init__(self, path):
        temp = plt.imread(path)
        self.pixels = temp[:, :, 0]*0.11 + temp[:, :, 1]*0.59 + temp[:, :, 2]*0.3
        self.theta = np.zeros((self.pixels.shape))
        self.gradientMagnitude = np.zeros((self.pixels.shape))
        self.preservedGradient = np.zeros((self.pixels.shape))
        self.edge = np.zeros((self.pixels.shape))

    def gaussianBlur(self):
        def convolutionOperation(arr1, arr2):
            total = 0
            for i in range(len(arr1)):
                for j in range(len(arr1[0])):
                    total += arr1[i, j] * arr2[i, j]
            return total

        gaussianKernel = np.zeros((5, 5))
        gaussianKernel[0] = [2, 4, 5, 4, 2]
        gaussianKernel[1] = [4, 9, 12, 9, 4]
        gaussianKernel[2] = [5, 12, 15, 12, 5]
        gaussianKernel[3] = [4, 9, 12, 9, 4]
        gaussianKernel[4] = [2, 4, 5, 4, 2]
        gaussianKernel *= 1/159

        img = np.pad(self.pixels, ((2, 2), (2, 2)), "constant", constant_values = ((0, 0), (0, 0)))
        blurredImg = np.zeros((self.pixels.shape))

        for i in range(len(blurredImg)):
            for j in range(len(blurredImg[0])):
                blurredImg[i, j] = convolutionOperation(gaussianKernel, img[i : i+5, j : j+5])

        self.pixels = blurredImg

    def findGradient(self):
        def convolutionOperation(arr1, arr2):
            total = 0
            for i in range(len(arr1)):
                for j in range(len(arr1[0])):
                    total += arr1[i, j] * arr2[i, j]
            return total

        def nearest45rounding(a):
            b = a % 45
            if (45 - b > b):
                return a - b
            return a - b + 45 

        kgx = np.zeros((3, 3))
        kgx[0] = [-1, 0, 1]
        kgx[1] = [-2, 0, 2]
        kgx[2] = [-1, 0, 1]

        kgy = np.zeros((3, 3))
        kgy[0] = [1, 2, 1]
        kgy[1] = [0, 0, 0]
        kgy[2] = [-1, -2, -1]

        img = np.pad(self.pixels, ((1, 1), (1, 1)), "constant", constant_values = ((0, 0), (0, 0)))

        for i in range(len(self.gradientMagnitude)):
            for j in range(len(self.gradientMagnitude[0])):
                gX = convolutionOperation(kgx, img[i : i+3, j : j+3])
                gY = convolutionOperation(kgy, img[i : i+3, j : j+3])
                self.gradientMagnitude[i, j] = (gX**2 + gY**2)**0.5
                self.theta[i, j] = nearest45rounding(math.atan(gY/gX)*180/math.pi) % 180

    def nonMaximumSupression(self):
        def getNeigbouringPixels(y, x):
            neighbor = list() #0 upper left, 1 up, 2 upper right, 3 left, 4 right, 5 lower left, 6 down, 7 lower right
            try :
                neighbor.append(self.gradientMagnitude[y-1, x-1])
            except :
                neighbor.append(-1)
            
            try :
                neighbor.append(self.gradientMagnitude[y-1, x])
            except :
                neighbor.append(-1)

            try :
                neighbor.append(self.gradientMagnitude[y-1, x+1])
            except :
                neighbor.append(-1)

            try :
                neighbor.append(self.gradientMagnitude[y, x-1])
            except :
                neighbor.append(-1)

            try :
                neighbor.append(self.gradientMagnitude[y, x+1])
            except :
                neighbor.append(-1)
            
            try :
                neighbor.append(self.gradientMagnitude[y+1, x-1])
            except :
                neighbor.append(-1)

            try :
                neighbor.append(self.gradientMagnitude[y+1, x])
            except :
                neighbor.append(-1)

            try :
                neighbor.append(self.gradientMagnitude[y+1, x+1])
            except :
                neighbor.append(-1)
            
            return neighbor

        for i in range(len(self.gradientMagnitude)):
            for j in range(len(self.gradientMagnitude[0])):
                currentNeighbor = getNeigbouringPixels(i, j)
                currentMagnitude = self.gradientMagnitude[i, j]
                if(self.theta[i, j] == 0 and currentMagnitude > currentNeighbor[3] and currentMagnitude > currentNeighbor[4]):
                    self.preservedGradient[i, j] = self.gradientMagnitude[i, j]
                elif(self.theta[i, j] == 45 and currentMagnitude > currentNeighbor[2] and currentMagnitude > currentNeighbor[5]):
                    self.preservedGradient[i, j] = self.gradientMagnitude[i, j]
                elif(self.theta[i, j] == 90 and currentMagnitude > currentNeighbor[1] and currentMagnitude > currentNeighbor[6]):
                    self.preservedGradient[i, j] = self.gradientMagnitude[i, j]
                elif(self.theta[i, j] == 135 and currentMagnitude > currentNeighbor[0] and currentMagnitude > currentNeighbor[7]):
                    self.preservedGradient[i, j] = self.gradientMagnitude[i, j]

    def doubleThresholding(self, lowerThreshold, higherThreshold):
        for i in range(len(self.preservedGradient)):
            for j in range(len(self.preservedGradient[0])):
                currentPixel = self.preservedGradient[i, j]
                if(currentPixel > higherThreshold):
                    self.edge[i, j] = 1
                elif(currentPixel >= lowerThreshold):
                    self.edge[i, j] = (lowerThreshold + higherThreshold)/2
                        
    def edgeTracking(self):
        def getNeigbouringPixels(y, x):
            neighbor = list() #0 upper left, 1 up, 2 upper right, 3 left, 4 right, 5 lower left, 6 down, 7 lower right
            try :
                neighbor.append(self.edge[y-1, x-1])
            except :
                neighbor.append(-1)
            
            try :
                neighbor.append(self.edge[y-1, x])
            except :
                neighbor.append(-1)

            try :
                neighbor.append(self.edge[y-1, x+1])
            except :
                neighbor.append(-1)

            try :
                neighbor.append(self.edge[y, x-1])
            except :
                neighbor.append(-1)

            try :
                neighbor.append(self.edge[y, x+1])
            except :
                neighbor.append(-1)
            
            try :
                neighbor.append(self.edge[y+1, x-1])
            except :
                neighbor.append(-1)

            try :
                neighbor.append(self.edge[y+1, x])
            except :
                neighbor.append(-1)

            try :
                neighbor.append(self.edge[y+1, x+1])
            except :
                neighbor.append(-1)
            return neighbor

        ischanged = True
        while(ischanged):
            ischanged = False
            for i in range(len(self.edge)):
                for j in range(len(self.edge[0])):
                    currentNeighbors = getNeigbouringPixels(i, j)
                    if(self.edge[i, j] != 0 and self.edge[i, j] < 1):
                        if(1 in currentNeighbors):
                            self.edge[i, j] = 1
                            ischanged = True
        
        for i in range(len(self.edge)):
            for j in range(len(self.edge[0])):
                if(self.edge[i, j] != 1):
                    self.edge[i, j] = 0

    def canny(self, lowerThreshold, higherThreshold):
        self.gaussianBlur()
        self.findGradient()
        self.nonMaximumSupression()
        self.doubleThresholding(lowerThreshold, higherThreshold)
        self.edgeTracking()

if __name__ == "__main__":
    img = GrayscaleImage("C:\\Users\\Farhan\\Desktop\\bumi.png")
    img.canny(20/255, 80/255)
    plt.imshow(img.edge)
    plt.show()