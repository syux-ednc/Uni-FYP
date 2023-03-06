import lmdb
import cv2
import numpy as np
import dataset

from recognition.recognize import TextRecognizer

#dbPath = "../evaluation/IC03_860"
#dbPath = "../evaluation/SVT"
dbPath = "../evaluation/IC13_857"
#dbPath = "../evaluation/IIIT5k_3000"
crnn_model_path = '../recognition/final_model.pth'

text_recognizer = TextRecognizer(crnn_model_path)

train_dataset = dataset.lmdbDataset(dbPath)

with lmdb.open(dbPath) as env:

    txn = env.begin()

    #Show the statistics of the DB, look for the "entries" label for no. of rows available
    print("Statistics of the DB: ", txn.stat())
    print("Size of dataset: " , train_dataset.__len__())

    correct = 0
    index = 0

    for key, value in txn.cursor():

        if (index != train_dataset.__len__()):

            imageBuf = np.fromstring(value, dtype=np.uint8)
            img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)

            _, result = text_recognizer.predict(img)
            _, label = train_dataset.__getitem__(index)
            print("Predicted: ", result, " Ground Truth:", label)

            if(result == label.lower()):
                correct += 1

            index += 1
        else:
            break

print("Accuracy of Model:",  correct, "/", index, "=" ,correct/index)
