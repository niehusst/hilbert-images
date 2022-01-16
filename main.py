from PIL import Image
import numpy as np

imPath = "./cat.jpg"

def transpose(mat, tl, br, alongTLBR):
    num_rots = 1 if alongTLBR else 3
    mat[tl[0]:br[0],tl[1]:br[1]] = np.rot90(np.fliplr(mat[tl[0]:br[0],tl[1]:br[1]]), num_rots)

def hilbertKernel(mat, links, tl, br):
    """
    Trace the hilbert curve line path through mat. Recurse down
    to order 1 and then work way back up. alters mat in place
    (recursion depth is log_4(len(mat)) )

    mat - 2d array. already marked w/ incrementing order1 pseudo hilbert curves paths
    links - dict. connections between the end of a orderX pseudo hilbert curve and the start
            of the next that it joins to to create orderX+1 pseudo hilbert curve {(coord) : (coord)}
    tl - tuple. top left coord of focussed sub-matrix of mat
    br - tuple. bottom right coord of focussed sub-matrix of mat
    """
    if (tl[0]+1, tl[1]+1) == br:
        # reached order1 pseudo hilbert curve, work already done
        return

    # divide + conquer 4 quadrants
    mid = ((tl[0]+br[0])//2, (tl[1]+br[1])//2) 
    hilbertKernel(mat, links, tl, mid)                            # q2 (tl)
    hilbertKernel(mat, links, (tl[0], mid[1]+1), (mid[0], br[1])) # q1 (tr)
    hilbertKernel(mat, links, (mid[0]+1, tl[1]), (br[0], mid[1])) # q3 (bl)
    hilbertKernel(mat, links, (mid[0]+1, mid[1]+1), br)           # q4 (br)

    # transpose q1 + q2 quadrants
    transpose(mat, tl, mid, True)
    transpose(mat, (tl[0], mid[1]+1), (mid[0], br[1]), False)

    # link start + end of each joined quadrant for path tracing
    links[mat[mid[0]+1,tl[1]]] = mat[mid[0],tl[1]]   # tl q3 -> bl q2
    links[mat[mid[0],mid[1]]] = mat[mid[0],mid[1]+1] # br q2 -> bl q1
    links[mat[mid[0],br[1]]] = mat[mid[0]+1,br[1]]   # br q1 -> tr q4

# use hilbert curves to make path through image
def imageToFlatArray(image):
    # enforce dimensions dividable into 4x4 squares
    image = np.array(image)
    (height, width, _) = image.shape
    dim = max(width, height)
    dim += dim % 4 # make sure dims are divis by 4
    if not (dim == width == height):
        image = np.pad(image, ((0, dim-width), (0, dim-height), (0,0)), mode='reflect')

    # clone image matrix to create path;
    # TODO start with order1 hilbert curves (all nums unique, order inc w/in each curve)
    pathMat = [[(row*dim)+col for col in range(dim)] for row in range(dim)] #doesnt draw the curves yet :\
    links = {}
    # run for side effects on pathMat
    hilbertKernel(pathMat, links, (0,0), (dim,dim))

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
    rawImage = Image.open(imagePath)

    # craft hilbert line path through image
    arrangedPixels = imageToFlatArray(image)

    # convert pixel vals to audio frequencies
    freqs = map(arrangedPixels, pixelToFreq)

    # make actual sound out of our construction
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
