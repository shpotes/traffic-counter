import os
import sys
import pandas as pd
import tensorflow as tf

def prepare_dataset(data_dir, exclude_dirs=None):
    """
    TODO: generate metadata 
    """
    pass

def build_source_from_metadata(metadata, data_dir, mode='train',
                               exclude_labels=None):
    pass


def preprocess_image(image):
    pass

def load(row):
    pass

def generate_anchors():
    pass

def make_dataset(source, training=False, batch_size=1,
                 num_epochs=1, num_parallel_calls=1, shuffle_buffer_size=None):
    pass    
    
