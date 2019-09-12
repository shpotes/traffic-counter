import os
from itertools import groupby
from operator import itemgetter
from typing import List, Tuple, Dict

import gin
import pandas as pd
import numpy as np
import tensorflow as tf
from .generate_anchors import generate_anchors as gen_anch
from vehicle_nowcasting.utils import iou

def build_source_from_metadata(metadata: pd.DataFrame,
                               label_map: Dict[str, int],
                               data_dir: str,
                               excluded_labels: List[str] = [],
                               mode: str = 'train') -> \
                               List[Tuple[str, List[Tuple[str, Tuple[int]]]]]:

    excluded_labels = set(excluded_labels)

    df = metadata.copy()
    df = df[df.split == mode]

    df.filepath = df.filepath.apply(lambda x: os.path.join(data_dir, x))

    includes = df.label.apply(lambda x: x not in excluded_labels)

    df.label = df.label.apply(lambda x: label_map[x])

    bbox = df[['label', 'xmin', 'ymin', 'xmax', 'ymax']].values
    bbox = [tuple(x) for x in bbox]

    source = list(zip(df.filepath, bbox))
    source = [(k, [x for _, x in g])
              for k, g in groupby(source, itemgetter(0))]

    return source

def load(row: tf.data.Dataset) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Load dataset from given reference
    """
    fill = tf.zeros(shape=(1, 5), dtype=tf.int32) - (1 << 31)
    mask = tf.not_equal(row['bbox_info'], fill)
    bbox = tf.boolean_mask(row['bbox_info'], mask)
    bbox = tf.reshape(bbox, shape=(-1, 5))
    filepath = row['image']
    img = tf.io.read_file(filepath)
    img = tf.io.decode_jpeg(img)
    return img, bbox


def generate_anchors(base_size: int = 4,
                     ratios: List[float] = [1, 1.25, 1.5, 1.75, 2],
                     scales: np.ndarray = 2 ** np.arange(3, 10),
                     dx: np.ndarray = 19 * np.arange(12),
                     dy: np.ndarray = 19 * np.arange(12)):

    delta = np.transpose([np.tile(dx, len(dy)), np.repeat(dy, len(dx))])
    delta = np.hstack([delta, delta])
    anchors = gen_anch(base_size=base_size, ratios=ratios, scales=scales)
    anchors = anchors.reshape(-1, 4, 1) + np.zeros((*anchors.shape,
                                                    delta.shape[0]))
    anchors += delta.T.reshape(1, 4, -1)
    anchors = anchors.transpose(2, 0, 1).reshape(-1, 4)
    anchors = anchors[((anchors > 0) & (anchors < 480)).all(axis=-1), :]

    return anchors



def compute_anchor_boxes(anchors, img, bbox_gt):
    iou_matrix = iou(anchors, bbox_gt[:, 1:])

    # Assign a positive label to the anchor with the highest IoU
    max_score = tf.reduce_max(iou_matrix, axis=0)
    best_loc = tf.cast(tf.where(iou_matrix == max_score), dtype=tf.int32)
    best_bbox = tf.gather(anchors, best_loc[:, 0])

    # Instead of stores the GT class, 
    # store a reference to the GT.
    # We will need all the GT bbox
    # information in order to compute the loss
    best_gt_pointer = tf.reshape(best_loc[:, 1], shape=[-1, 1])

    best = tf.concat([best_gt_pointer, best_bbox], axis=1)

    # Assign a positive label to an anchor that has an IoU > 0.7
    pos_cond = iou_matrix > 0.7
    pos_mask = tf.reduce_any(pos_cond, axis=1)
    pos_bbox = tf.boolean_mask(anchors, pos_mask)

    pos_gt_pointer = tf.reshape(tf.where(pos_cond)[:, 1], shape=[-1, 1])
    pos_gt_pointer = tf.cast(pos_gt_pointer, tf.int32)

    pos = tf.concat([pos_gt_pointer, pos_bbox], axis=1)
    
    # Assign a negative label to an anchor that has an IoU < 0.3
    neg_cond = iou_matrix < 0.3
    neg_mask = tf.reduce_any(neg_cond, axis=1)
    neg_bbox = tf.boolean_mask(anchors, neg_mask)

    # There are not a GT for this boxes, so just fill
    neg_gt_pointer = tf.zeros((len(neg_bbox), 1), tf.int32) - 1 

    neg = tf.concat([neg_gt_pointer, neg_bbox], axis=1)
    
    anchor_bbox = tf.concat([pos, neg, best], axis=0)

    return img, anchor_bbox, bbox_gt

def normalize(bbox_raw: List[Tuple[str, Tuple[int]]]) \
    -> List[Tuple[str, Tuple[int]]]:
    fill = -(1 << 31)
    max_length = max(map(len, bbox_raw))
    bbox_norm = list(map(lambda x: x + [(fill, fill, fill, fill, fill)
                                        for _ in range(max_length - len(x))],
                         bbox_raw))
    return bbox_norm

def preprocess_input(image, bbox, target_size=(224, 224)):
    size = 480, 704
    new_x, new_y = target_size

    img = tf.image.resize(image, (224, 224)) / 255

    # "Rescale" bbox is also needed
    scale = tf.constant([1, new_x / size[1], new_y / size[0],
                         new_x / size[1], new_y / size[0]])
    bbox_gt = tf.cast(tf.round(tf.cast(bbox, tf.float32) * scale), tf.int32)
    
    


    return img, bbox_gt

def hierarchical_sampling(img, anchors, gt, batch_size, N_sampling=1):
    """
    Implementation notes:
    Increase N_sampling makes quite difficult to trace the image
    """

    positive_mask = anchors[:, 0] != -1
    batch_positive = tf.boolean_mask(anchors, positive_mask)
    batch_positive = tf.random.shuffle(batch_positive)
    batch_positive_size = tf.math.minimum(len(batch_positive), batch_size // 2)
    batch_positive = batch_positive[:batch_positive_size, :]

    negative_mask = anchors[:, 0] == -1
    batch_negative = tf.boolean_mask(anchors, negative_mask)
    batch_negative = tf.boolean_mask(batch_negative, # Remove pad
                                     tf.reduce_any(batch_negative != 0, axis=1))
    batch_negative = tf.random.shuffle(batch_negative)
    batch_negative_size = batch_size - batch_positive_size
    batch_negative = batch_negative[:batch_negative_size, :]

    batch = tf.concat([batch_positive, batch_negative], axis=0)
    batch = tf.random.shuffle(batch)
    return img, batch, gt

#@gin.configurable
def make_dataset(sources: List[Tuple[str, List[Tuple[str, Tuple[int]]]]],
                 training: bool = False, batch_size: int = 32,
                 num_epochs: int = 1, num_parallel_calls: int = 1,
                 shuffle_buffer_size: int = None, mode: str = None,
                 hierarchical: bool = True) -> tf.data.Dataset:

    if shuffle_buffer_size is None:
        shuffle_buffer_size = batch_size * 4

    images, bbox_raw = zip(*sources)
    bbox_info = normalize(bbox_raw)

    ds = tf.data.Dataset.from_tensor_slices({
        'image': list(images),
        'bbox_info' : bbox_info
    })

    if training:
        ds = ds.shuffle(shuffle_buffer_size)

    ds = ds.map(load, num_parallel_calls=num_parallel_calls)
    ds = ds.map(preprocess_input)

    anchors = generate_anchors()
    anchors = tf.constant(anchors, dtype=tf.int32)
    ds = ds.map(lambda img, bbox_gt:
                compute_anchor_boxes(anchors, img, bbox_gt),
                num_parallel_calls)

    ds = ds.repeat(count=num_epochs)

    if hierarchical:
        ds = ds.map(lambda x, y, z:
                    hierarchical_sampling(x, y, z, batch_size))

    if mode == 'rpn':
        ds = ds.map(rpn_mode)

    if mode == 'detection':
        ds = ds.map(detection_mode)

    return ds
