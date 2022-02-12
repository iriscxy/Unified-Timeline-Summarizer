# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains some utility functions"""
import select
import sys
import time

import os
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def get_config():
    """Returns config for tf.session"""
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    return config


def load_ckpt(saver, sess, ckpt_dir="train"):
    """Load checkpoint from the ckpt_dir (if unspecified, this is train dir) and restore it to saver and sess, waiting 10 secs in the case of failure. Also returns checkpoint name."""
    try:
        latest_filename = "checkpoint_best" if ckpt_dir == "eval" else None
        ckpt_dir = os.path.join(FLAGS.log_root, ckpt_dir)
        ckpt_state = tf.train.get_checkpoint_state(ckpt_dir, latest_filename=latest_filename)
        tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
        saver.restore(sess, ckpt_state.model_checkpoint_path)
        return ckpt_state.model_checkpoint_path
    except Exception as e:
        print('The error is -->', e)
        tf.logging.info("Failed to load checkpoint from %s.", ckpt_dir)


def load_specific_ckpt(saver, sess, ckpt_dir, latest_filename):
    """Load checkpoint from the ckpt_dir (if unspecified, this is train dir) and restore it to saver and sess, waiting 10 secs in the case of failure. Also returns checkpoint name."""
    try:
        ckpt_dir = os.path.join(FLAGS.log_root, ckpt_dir, latest_filename)
        # ckpt_state = tf.train.get_checkpoint_state(ckpt_dir, latest_filename)
        tf.logging.info('Loading checkpoint %s', ckpt_dir)
        saver.restore(sess, ckpt_dir)
        return latest_filename
    except Exception as e:
        print('The error is -->', e)
        tf.logging.info("Failed to load checkpoint from %s.", ckpt_dir)


def get_input_with_timeout(prompt: str, time: int = 10, default_input=None):
    """

    :param prompt: input prompt
    :param time: timeout in seconds
    :param default_input: when timeout this function will return this value
    :return: input value or default_input
    """
    print(bcolors.GREENBACK + prompt + bcolors.ENDC)
    print(bcolors.BLINK + ' Time Limit %d seconds' % time + bcolors.ENDC)
    i, o, e = select.select([sys.stdin], [], [], time)
    if i:
        return sys.stdin.readline().strip()
    else:
        return default_input


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0;37;40m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5;41;42m'
    GREENBACK = '\033[0;40;42m'
    REDBACK = '\033[0;42;101m'
