"""
Simple script that checks if a checkpoint is corrupted with any inf/NaN values. Run like this:
  python inspect_checkpoint.py model.12345
"""
import os

import tensorflow as tf
import sys
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception(
            "Usage: python inspect_checkpoint.py <file_name> tensor_name out_dir\nNote: Do not include the .data .index or .meta part of the model checkpoint in file_name.")
    file_name = sys.argv[1]
    tensor_name = sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv) == 4 else ''
    reader = tf.train.NewCheckpointReader(file_name)
    var_to_shape_map = reader.get_variable_to_shape_map()

    finite = []
    all_infnan = []
    some_infnan = []

    if not os.path.exists(out_dir) and out_dir != '': os.makedirs(out_dir)

    for key in sorted(var_to_shape_map.keys()):
        if tensor_name == 'list':
            print(key)
        elif tensor_name in key:
            tensor = reader.get_tensor(key)
            np.save(os.path.join(out_dir, key.split('/')[-1]+'.npy'), tensor)
