import lmdb
import cv2
import numpy as np
import dataset

#dbPath refers to the LMDB folder that contains the data.mdb and lock.mdb files
#dbPath = "../evaluation/IIIT5k_3000"
dbPath = "../evaluation/SVT"
#dbPath = "../evaluation/IC03_860"
#dbPath = "../evaluation/IC13_857"

def iterateDB():

    """
    Loop through the whole DB and display statistics about DB
    """

    with lmdb.open(dbPath) as env:

        txn = env.begin()

        #Show the statistics of the DB, look for the "entries" label for no. of rows available
        print("Statistics of the DB: ", txn.stat())

        for key, value in txn.cursor():
            print('Img key : ', key)
            print('Raw Img Value : ', value)
            imageBuf = np.fromstring(value, dtype=np.uint8)
            img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                cv2.imshow('image', img)
                cv2.waitKey()
            else:
                print('This is a label: {}'.format(value))

def labelAtIndex(index):

    """
    Get Image and Label at specific index
    ARGS:
        index    : row of the DB
    """

    train_dataset = dataset.lmdbDataset(dbPath)

    # Display the total number of rows in the DB
    dbSize = train_dataset.__len__()
    print("Total Number of Rows in DB: ", dbSize)

    if (index >= 0) and (index < dbSize):

        _, label = train_dataset.__getitem__(index)
        print("The label at index", index, "is :" , label)

        with lmdb.open(dbPath) as env:
            txn = env.begin()
            i = 0
            for key, value in txn.cursor():
                if i == index:
                    imageBuf = np.fromstring(value, dtype=np.uint8)
                    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        cv2.imshow('image', img)
                        cv2.waitKey()
                        return
                    else:
                        print('No image available, the corresponding label is: {}'.format(value))
                else:
                    i+=1
    else:
        print("Index Out of Range Error")

if __name__ == "__main__":

    #labelAtIndex(4)
    iterateDB()
