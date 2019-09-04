import cv2
import tensorflow as tf
import numpy as np

from typing import Tuple

def xy2wh(arr):
    pass

def wh2xy(arr):
    pass

@tf.function
def iou(bboxes1: tf.Tensor, bboxes2: tf.Tensor) -> tf.Tensor:
    """
    Compute IoU between two given bouding boxes.
    TODO: test implementation
    """
    x11, y11, x12, y12 = tf.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = tf.split(bboxes2, 4, axis=1)
    
    xA = tf.math.maximum(x11, tf.transpose(x21))
    yA = tf.math.maximum(y11, tf.transpose(y21))
    xB = tf.math.minimum(x12, tf.transpose(x22))
    yB = tf.math.minimum(y12, tf.transpose(y22))
    
    interArea = tf.math.maximum((xB - xA + 1), 0) * \
        tf.math.maximum((yB - yA + 1), 0)
    
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    
    iou = interArea / (boxAArea + tf.transpose(boxBArea) - interArea)
    
    return iou

def plot_bb(img: np.ndarray, org: np.ndarray,
            color: Tuple[int, int, int], size: int) -> np.ndarray:
    for anch in org.astype(int):
        img = cv2.rectangle(img, (anch[1], anch[2]), 
                            (anch[3], anch[4]), color, size)
    return img
