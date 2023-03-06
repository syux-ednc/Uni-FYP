import os
import lmdb
import cv2
import numpy as np
from glob import glob

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if type(v) == str:
                v = v.encode()
            txn.put(k.encode(), v)

def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """

    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1073741824)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)
    env.close()

if __name__ == '__main__':

    outputPath = "exp100ktrain"
    trainImage = []
    trainLabel = []
    testImage = []
    testLabel = []

    imgPathL = glob('trainImages/*/*/*.jpg')

    sizeOfData = 100000

    train = int(0.8 * sizeOfData)
    test = int(0.2 * sizeOfData)

    for x in range(sizeOfData):
        if x < train:
            trainImage.append(imgPathL[x])
            trainLabel.append(imgPathL[x].split('_')[1])
        elif ((x >= train) and (x < sizeOfData)):
            testImage.append(imgPathL[x])
            testLabel.append(imgPathL[x].split('_')[1])

#     print("Length of Train Image is:", len(trainImage), "Length of Train Label is:", len(trainLabel))
#     for i in range(len(trainImage)):
#          print("Path is:", trainImage[i], "Label is:", trainLabel[i])

#     print("--------------------------------------")

#     print("Length of Test Image is:", len(testImage), "Length of Test Label is:", len(testLabel))
#     for j in range(len(testImage)):
#          print("Path is:", testImage[j], "Label is:", testLabel[j])

    createDataset(outputPath, trainImage, trainLabel)
    #createDataset(outputPath, testImage, testLabel)

    # # For testing purposes
    # imageList = ["images/c1.jpg", "images/c2.jpg", "images/c3.jpg", "images/c4.jpg", "images/c5.jpg", "images/c6.jpg",
    #              "images/c7.jpg", "images/c8.jpg", "images/c9.jpg", "images/c10.jpg"]
    # labelList = ["soothers", "potentialities", "strapped", "laughable", "littlebugs", "microscope",
    #              "routers", "brushstroke", "approaching", "duke"]
    # createDataset(outputPath, imageList, labelList)
