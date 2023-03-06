import os
import cv2
import imutils
import argparse

from datetime import datetime
from utils import display_image
from utils import format_sentence
from detection.detect import TextDetector
from recognition.recognize import TextRecognizer

parser = argparse.ArgumentParser()
#Remove default when done testing
parser.add_argument('--liveRec', action='store_true', default=False, help='Live Text Recognition using camera')
parser.add_argument('--pureRec', action='store_true', default=False, help='Used only to recognise cropped Text')
parser.add_argument('--viewWidth', type=int, default=640, help='Width of camera frame')

east_model_path = './detection/frozen_east_text_detection.pb'
crnn_model_path = './recognition/final_model.pth'
demo_image_path = './images/non-cropped/'
demo_cropped_path = './images/cropped/'

args = parser.parse_args()
text_detector = TextDetector(east_model_path)
text_recognizer = TextRecognizer(crnn_model_path)

def compute_frame(frame):

    #start = datetime.now()

    boxes, confidences, indices, width_ratio, height_ratio = text_detector.detect(frame)
    index_map = {}

    for i in indices:

        key = i[0]
        vertices = cv2.boxPoints(boxes[i[0]])

        for j in range(4):
            vertices[j][0] *= width_ratio
            vertices[j][1] *= height_ratio

        top_left = (min([vertices[0][0], vertices[1][0]]), min([vertices[1][1], vertices[2][1]]))
        btm_right = (max([vertices[2][0], vertices[3][0]]), max([vertices[0][1], vertices[3][1]]))

        if top_left[0] < 0 or top_left[1] < 0:
            continue

        text_roi = frame[int(top_left[1]):int(btm_right[1]), int(top_left[0]):int(btm_right[0])]

        if text_roi.shape[0] > 0 and text_roi.shape[1] > 0:
            #cv2.imshow(f't_{key}', text_roi)
            _, pred = text_recognizer.predict(text_roi)
            index_map[key] = {
                'vertices': vertices,
                'pred_text': pred,
            }

    if len(index_map.keys()):
        try:
            for i in indices:
                key = i[0]
                for j in range(4):
                    p1 = (index_map[key]['vertices'][j][0], index_map[key]['vertices'][j][1])
                    p2 = (index_map[key]['vertices'][(j + 1) % 4][0], index_map[key]['vertices'][(j + 1) % 4][1])
                    cv2.line(frame, p1, p2, (255, 0, 0), 2, cv2.LINE_AA)

                cv2.putText(frame, index_map[key]['pred_text'],
                            (index_map[key]['vertices'][1][0], index_map[key]['vertices'][1][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
        except:
            pass

    #print(f'Recognition Time: {(datetime.now() - start).total_seconds() * 100:.2f} ms')

    format_sentence(index_map)

    return frame


if __name__ == "__main__":

    # Live Text Recognition using frames from camera feed
    if args.liveRec:
        print('Initializing Camera Feed')
        cap = cv2.VideoCapture(0)
        print('Press q to quit streaming')
        while (cap.isOpened()):
            ret, frame = cap.read()
            frame = imutils.resize(frame, width=args.viewWidth)

            detected_frame = compute_frame(frame)

            cv2.imshow('frame', detected_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    # Static Text Recognition of Images from Cropped Text Only
    elif args.pureRec:
        print('Running Recognition from cropped text only')
        for file_name in os.listdir(demo_cropped_path):

            #Single specific Cropped Text to test
            # if file_name != "1.png":
            #     continue

            input_cropped_path = f'{demo_cropped_path}/{file_name}'
            frame = cv2.imread(input_cropped_path)
            raw_pred, sim_pred = text_recognizer.predict(frame)
            print('%-20s => %-20s' % (raw_pred, sim_pred))
            display_image(frame, args.viewWidth)


    # Static Text Recognition using static images with natural scene
    else:
        print('Running Detection & Recognition for static scene images')
        for file_name in os.listdir(demo_image_path):

            # Single specific Image to test
            # if file_name != "welcome.jpg":
            #     continue

            input_image_path = f'{demo_image_path}/{file_name}'
            frame = cv2.imread(input_image_path)
            frame = compute_frame(frame)
            display_image(frame, args.viewWidth)
