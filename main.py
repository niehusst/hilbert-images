from PIL import Image
import numpy as np

imPath = "./cat.jpg"


# use hilbert curves to make path through image
def imageToFlatArray(image):
    # enforce dimensions diviadable into 4x4
    image = np.array(image)
    (height, width) = image.shape
    dim = max(width, height)
    dim += 1*(dim & 1)
    if not (dim == width == height):
        image = np.pad(image, ((0, dim-width), (0, dim-height),), mode='reflect')

    # clone image matrix to create path;
    # start with order1 hilbert curves
    pathMat = [[([2,3] if row & 1 else [1,4])[col & 1] for col in range(dim)] for row in range(dim)]

    return pathMat
    # TODO: actually flatten
    seen = set()
    curr = (0,0)

def pixelToFreq(pixel):
    # TODO: improve/finish
# convert rgb to hex code number? but then red dominates other color amplitudes
    (r,g,b) = pixel
    return r + g + b

def frequenciesToMp3(freqs):
    pass

def imageToSound(imagePath):
    # load image from path
    rawImage = Image.open(imagePath)

    arrangedPixels = imageToFlatArray(image)

    freqs = map(arrangedPixels, pixelToFreq)

    audio = frequenciesToMp3(freqs)

    # TODO: play audio

#imageToSound(imPath)

def hilbertTest():
    testIn = [[0]*8 for _ in range(8)]
    expected = [
1,4,1,2,3,4,1,4,
2,3,4,3,2,1,2,3,
3,2,1,2,3,4,3,2,
4,1,4,3,2,1,4,1,
1,2,3,4,1,2,3,4,
4,3,2,1,4,3,2,1,
1,4,1,4,1,4,1,4,
2,3,2,3,2,3,2,3,
    ]
    actual = imageToFlatArray(testIn)
    print(actual)
    print(expected == actual)

hilbertTest()
