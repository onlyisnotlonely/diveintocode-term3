import cv2
from cv2 import dnn


def detect(rgb_frame, class_names, prototxt, caffemodel):
    in_scale_factor = 0.017
    in_width = 224
    in_height = 224
    mean_val = (103.94, 116.78, 123.68)
    max_class_id = 0
    max_class_point = 0;
    net = dnn.readNetFromCaffe(prototxt, caffemodel)
    blob = dnn.blobFromImage(rgb_frame, in_scale_factor, (in_width, in_height), mean_val)
    net.setInput(blob)
    detections = net.forward()
    
    for i in range(detections.shape[1]):
        class_point = detections[0, i, 0, 0]

        if (class_point > max_class_point):
            max_class_id = i
            max_class_point = class_point

    return class_names[max_class_id]
