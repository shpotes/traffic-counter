import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf

from itertools import groupby
from operator import itemgetter
from typing import List, Tuple, Dict
from .generate_anchors import generate_anchors as gen_anch
from .utils import iou

def build_source_from_metadata(metadata: pd.DataFrame,
                               label_map: Dict[str, int],
                               data_dir: str,
                               excluded_labels: List[str] = [],
                               mode:str = 'train') -> \
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

def load(row):
    fill = tf.zeros(shape=(1, 5), dtype=tf.int32) - (1 << 31)
    mask = tf.not_equal(row['bbox_info'], fill)
    bbox = tf.boolean_mask(row['bbox_info'], mask)
    bbox = tf.reshape(bbox, shape=(-1, 5))
    filepath = row['image']
    img = tf.io.read_file(filepath)
    img = tf.io.decode_jpeg(img)
    return img, bbox

def generate_anchors(base_size: int = 4,
                     ratios: List[float] = [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5],
                     scales: np.ndarray = 2 ** np.arange(3, 10),
                     dx: np.ndarray = 10 * np.arange(20),
                     dy: np.ndarray = 10 * np.arange(20)):
    
    delta = np.transpose([np.tile(dx, len(dy)), np.repeat(dy, len(dx))])
    delta = np.hstack([delta, delta])
    anchors = gen_anch(base_size=base_size, ratios=ratios, scales=scales)
    anchors = anchors.reshape(-1, 4, 1) + np.zeros((*anchors.shape,
                                                    delta.shape[0]))
    anchors += delta.T.reshape(1, 4, -1)
    anchors = anchors.transpose(2, 0, 1).reshape(-1, 4)
    anchors = anchors[((anchors > 0) & (anchors < 480)).all(axis=-1), :]

    return anchors

def compute_anchor_boxes(img, bbox_gt):
    anchors = generate_anchors() # TODO: parameters
    
    
def normalize(bbox_raw):
    fill = -(1 << 31)
    max_length = max(map(len, bbox_raw))
    bbox_norm = list(map(lambda x: x + [(fill, fill, fill, fill, fill)
                                        for _ in range(max_length - len(x))],
                         bbox_raw))
    return bbox_norm

def preprocess_input(image, bbox, target_size=(224, 224)):
    size = 480, 704
    new_x, new_y = target_size
    scale = tf.constant([1, new_x / size[1], new_y / size[0],
                         new_x / size[1], new_y / size[0]])
    bbox_gt = tf.cast(tf.round(tf.cast(bbox, tf.float32) * scale), tf.int32)

    img = tf.image.resize(image, (224, 224)) / 255

    return img, bbox_gt

def make_dataset(sources: List[Tuple[str, List[Tuple[str, Tuple[int]]]]],
                 training: bool = False, batch_size: int = 1,
                 num_epochs: int = 1, num_parallel_calls: int = 1,
                 shuffle_buffer_size: int = None) -> tf.data.Dataset:

    if shuffle_buffer_size is None:
        shuffle_buffer_size = batch_size * 4

    images, bbox_raw = zip(*sources)
    bbox_info = normalize(bbox_raw)
    
    ds = tf.data.Dataset.from_tensor_slices({
        'image': list(images),
        'bbox_info' : bbox_info
    })

    if training:
        ds = shuffle(shuffle_buffer_size)

    ds = ds.map(load, num_parallel_calls=num_parallel_calls)
    ds = ds.map(preprocess_input)
    #ds = ds.map(compute_anchor_boxes)
    return ds
