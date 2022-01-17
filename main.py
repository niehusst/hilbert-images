import math
from PIL import Image
from scipy.io import wavfile
import numpy as np
import simpleaudio as sa
import multiprocessing
from threading import Thread

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
    if tl == br: raise Exception('hey!!')
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
    transpose(mat, (tl[0], tl[1]), (mid[0]+1, mid[1]+1), True)
    transpose(mat, (tl[0], mid[1]+1), (mid[0]+1, br[1]+1), False)

    # link start + end of each joined quadrant for path tracing
    links[mat[mid[0],tl[1]]] = mat[mid[0]+1,tl[1]]       # tl q3 -> bl q2
    links[mat[mid[0]+1,mid[1]]] = mat[mid[0]+1,mid[1]+1] # br q2 -> bl q1
    links[mat[mid[0]+1,br[1]]] = mat[mid[0],br[1]]       # br q1 -> tr q4

# given num to find in adjacent spaces, return position of found num
def findNumPos(mat, num, loc):
    if mat[loc] == num:
        return loc
    h,w = mat.shape[:2]
    r = (loc[0]+1, loc[1])
    l = (loc[0]-1, loc[1])
    u = (loc[0], loc[1]+1)
    d = (loc[0], loc[1]-1)
    if loc[0]+1 < w and mat[r] == num:
        return r
    if loc[0]-1 >= 0 and mat[l] == num:
        return l
    if loc[1]+1 < h and mat[u] == num:
        return u
    if loc[1]-1 >= 0 and mat[d] == num:
        return d
    raise Exception(f"could not find {num} from {loc}")

# use hilbert curves to make path through image
def imageToFlatArray(image):
    # enforce dimensions dividable into 4x4 squares
    image = np.array(image)
    (height, width) = image.shape[:2]
    dim = max(width, height)
    dim += abs(dim - 2**math.ceil(math.log(dim,2))) # make sure dims are powers of 2 TODO: can this be avoided? as with squareness?
    if not (dim == width == height):
        image = np.pad(image, ((0, dim-width), (0, dim-height))+ (((0,0),)*(len(image.shape)-2)), mode='reflect')

    # clone image matrix to create path
    pathMat = np.array([[0]*dim for _ in range(dim)])
    # create order1 hilbert curves (all nums unique, order inc w/in each curve)
    i = v = 0
    while i < dim-1:
        j = 0
        while j < dim:
            pathMat[i,j] = v
            pathMat[i+1,j] = v+1
            pathMat[i+1,j+1] = v+2
            pathMat[i,j+1] = v+3
            v += 4
            j += 2
        i += 2
    links = {}
    # run for side effects on pathMat
    hilbertKernel(pathMat, links, (0,0), (dim-1,dim-1))

    # flatten path to 1d array
    curr = (0,0)
    links[pathMat[curr]] = pathMat[curr]
    curvePath = []
    debug = [] # x,y values for plotting shape
    while pathMat[curr] in links:
        # jump from end of last curve to next linked curve
        startVal = links[pathMat[curr]]
        curr = findNumPos(pathMat, startVal, curr)
        curvePath.append(image[curr])
        debug.append(curr)
        # jump through the located order1 pseudo hilbert curve
        for i in range(1,4):
            curr = findNumPos(pathMat, startVal + i, curr)
            curvePath.append(image[curr])
            debug.append(curr)

    return curvePath, debug

def pixelToAmplitude(pixel):
    # TODO: improve/finish
# convert rgb to hex code number? but then red dominates other color amplitudes
#    return int('%02x%02x%02x' % pixel, 16)
    (r,g,b) = pixel
    return int(r) + int(g) + int(b)

def createFreqsInRange(samples, freqs, start, end):
    for i in range(start, end):
        samples[i] += sum(map(lambda freq,amplitude: amplitude * np.sin(i * freq), enumerate(freqs)))

def frequenciesToWav(freqs):
    # create sampling array to convert to audio
    sample_rate = 44100
    samples = np.zeros((sample_rate,))
    # multithreading bcus it takes forever otherwise
    cpu_count = multiprocessing.cpu_count()
    div = sample_rate / cpu_count
    threads = [Thread(target=createFreqsInRange, args=(samples, freqs, int(i*div), int((i+1)*div))) for i in range(cpu_count)]
    map(lambda t: t.start(), threads)
    map(lambda t: t.join(), threads)

    # write to file
    fname = 'out.wav'
    wavfile.write(fname, sample_rate, samples)
    return fname

def imageToSound(imagePath):
    print('opening image')
    rawImage = Image.open(imagePath)

    # craft hilbert line path through image
    print('converting image to flat array')
    arrangedPixels,_ = imageToFlatArray(rawImage)

    # convert pixel vals to audio frequencies
    print('map pixels to amplitudes')
    freqs = map(pixelToAmplitude, arrangedPixels)

    # make actual sound file out of our construction
    print('write to wav file')
    return frequenciesToWav(freqs)

# play audio
audio_fname = imageToSound(imPath)
wav_obj = sa.WaveObject.from_wave_file(audio_fname)
play_obj = wav_obj.play()
play_obj.wait_done()

def hilbertTest():
    import matplotlib.pyplot as plt
    testIn = np.zeros((8,8))
    actual,pts = imageToFlatArray(testIn)
    plt.plot(list(map(lambda x: x[0], pts)),
             list(map(lambda x: x[1], pts)))
    plt.show()

#hilbertTest()
