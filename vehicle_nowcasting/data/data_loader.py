import os
from itertools import groupby
from operator import itemgetter
from typing import List, Tuple, Dict

import gin
import pandas as pd
import numpy as np
import tensorflow as tf
from .generate_anchors import generate_anchors as gen_anch
from vehicle_nowcasting.utils import *

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


def generate_anchors(dx: np.ndarray = 19 * np.arange(12),
                     dy: np.ndarray = 19 * np.arange(12)):

    delta = np.transpose([np.tile(dx, len(dy)), np.repeat(dy, len(dx))])
    delta = np.hstack([delta, delta])
    anchors = gen_anch()
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


def precompute_anchors_position(dx, dy):
    anchors = tf.constant(gen_anch(), tf.int32)
    net_dim = len(dx)
    k = len(anchors)

    print(k)

    dy = tf.reshape(dy, (-1, 1, 1))
    dx = tf.reshape(dx, (1, -1, 1))

    dx = tf.concat([dx, tf.zeros((1, net_dim, 1),
                                 dtype=tf.int32)], axis=-1)
    dy = tf.concat([tf.zeros((net_dim, 1, 1), dtype=tf.int32),
                    dy], axis=-1)

    reference = tf.reshape(dx + dy, (1, net_dim, net_dim, 1, 2))
    reference = tf.concat([reference, reference], axis=-1)

    anchors = reference + tf.reshape(anchors, (1, 1, 1, k, 4))
    anchors = tf.transpose(anchors, [0, 2, 1, 3, 4])
    return anchors

def rpn_mode(image, anchor_bbox, ground_truth, anchors, model='vgg_16'):
    k = anchors.shape[3]
    rf, stride, pad = get_RF(model)
    net_size = int((224 - rf + 2 * pad) / stride) + 1
    
    cond = tf.equal(tf.reshape(anchor_bbox[:, 1:], (-1, 1, 1, 1, 4)), anchors)
    cond = tf.reduce_all(cond, axis=-1)
    anchor_map = tf.cast(tf.where(cond), tf.int32)
    
    gt_pointer = anchor_bbox[:, :1]
    p_raw = tf.cast(gt_pointer != -1, tf.int32)
    
    p_loc_raw = tf.cast(anchor_map[:, -1:] * 2, tf.int32) + p_raw
    p_loc = tf.concat([anchor_map[:, 1:-1], p_loc_raw], axis=-1)
    p_template = tf.zeros((net_size, net_size, 2 * k), dtype=tf.float64)
    
    p = tf.tensor_scatter_nd_update(p_template,
                                    p_loc,
                                    tf.ones(32, dtype=tf.float64))
    
    gt_mask = tf.cast(p_raw[:,0], tf.bool)
    pointers = tf.gather(gt_pointer, tf.where(gt_mask)[:, 0])[:, 0]
    tf.gather(ground_truth, pointers)[:, 1:]
    
    ext_ground_truth = tf.concat([ground_truth, [[-1, 1, 2, 3, 4]]], axis=0)
    
    _gt = tf.gather(ext_ground_truth,
                tf.where(tf.equal(-1, gt_pointer),
                         len(ground_truth) * tf.ones_like(gt_pointer),
                         gt_pointer ))[:, 0, 1:]

    _gt = change_box_order(tf.cast(_gt, tf.float64))
    _bb = change_box_order(tf.cast(anchor_bbox[:, 1:], tf.float64))
    
    t_x = (_gt[:, 0] - _bb[:, 0]) / _bb[:, 2]
    t_x = tf.reshape(t_x, (-1, 1))
    t_y = (_gt[:, 1] - _bb[:, 1]) / _bb[:, 3]
    t_y = tf.reshape(t_y, (-1, 1))
    t_w = tf.math.log(_gt[:, 2] / _bb[:, 2])
    t_w = tf.reshape(t_w, (-1, 1))
    t_h = tf.math.log(_gt[:, 3] / _bb[:, 3])
    t_h = tf.reshape(t_h, (-1, 1))
    t_bbox = tf.concat([t_x, t_y, t_w, t_h], axis=-1)
    
    t_loc_raw = tf.cast(anchor_map[:, -1:], tf.int32)
    t_loc = tf.concat([anchor_map[:, 1:-1], t_loc_raw], axis=-1)
    
    t_template = tf.zeros((net_size, net_size, k, 4), dtype=tf.float64)
    t = tf.tensor_scatter_nd_update(tensor=tf.cast(t_template, tf.float64), 
                                    indices=tf.cast(t_loc, tf.int32),
                                    updates=tf.cast(t_bbox, tf.float64))
    t = tf.reshape(t, (net_size, net_size, -1))
    return image, tf.concat([p, t], axis=-1)


def make_dataset(sources: List[Tuple[str, List[Tuple[str, Tuple[int]]]]],
                 training: bool = False,
                 batch_size: int = 32,
                 num_epochs: int = 1,
                 num_parallel_calls: int = 1,
                 shuffle_buffer_size: int = None,
                 mode: str = None,
                 hierarchical: bool = True,
                 backbone_model='vgg_16') -> tf.data.Dataset:

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

    conv_move = compute_stride_from_receptive_field(backbone_model)
    conv_move = tf.constant(conv_move, dtype=tf.int32)
    
    anchors = generate_anchors(dx=conv_move, dy=conv_move)
    anchors = tf.constant(anchors, dtype=tf.int32)
    ds = ds.map(lambda img, bbox_gt:
                compute_anchor_boxes(anchors, img, bbox_gt),
                num_parallel_calls)

    ds = ds.repeat(count=num_epochs)

    if hierarchical:
        ds = ds.map(lambda x, y, z:
                    hierarchical_sampling(x, y, z, batch_size))

    if mode == 'rpn':
        anchors = precompute_anchors_position(dx=conv_move,
                                              dy=conv_move)
        ds = ds.map(lambda img, anchor_bbox, ground_truth:
                    rpn_mode(img, anchor_bbox, ground_truth, anchors))

    elif mode == 'detection':
        ds = ds.map(detection_mode)

    return ds
