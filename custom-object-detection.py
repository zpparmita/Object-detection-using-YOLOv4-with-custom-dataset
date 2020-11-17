# import the necessary packages
import numpy as np
import time
import cv2
import os
import argparse
import sys
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)

# Create the parser
parser = argparse.ArgumentParser(description='Custom Object Detection using yoloV4')
# Add the arguments
parser.add_argument('--classes-path', default=root + '/obj.names', type=str, help='obj.classes path')
parser.add_argument('--config-path', default=root + '/yolov4-tiny-obj.cfg', type=str,
                    help='yoloV4 cfg file path')
parser.add_argument('--weights-path', default=root + '/yolov4-tiny-obj_4000.weights',
                    type=str, help='pretrained weights file path')
parser.add_argument('--yolo-confidence', default=0.5, type=float, help='confidence')
parser.add_argument('--threshold', default=0.3, type=float, help='threshold')
args = parser.parse_args()

# load the customized class labels our YOLO model was trained on
with open(args.classes_path, 'rt') as f:
    LABELS = f.read().rstrip('\n').split('\n')

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = args.weights_path
configPath = args.config_path

# load our YOLO object detector trained on custom dataset (5 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


def detect_image(image):
    (H, W) = image.shape[:2]
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    # show timing information on YOLO

    # initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > args.yolo_confidence:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, args.threshold)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the image
            print("i={}, length={}, classIDs={}".format(i, len(classIDs), classIDs[i]))
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image


# load our input image and grab its spatial dimensions
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("We cannot open webcam")
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)
    r_image = detect_image(frame)

    cv2.imshow("Web cam input", r_image)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break

cap.release()
cv2.destroyAllWindows()
