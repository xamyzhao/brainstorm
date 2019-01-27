import numpy as np

import classification_utils
#from networks import lpat_networks
import basic_networks
import metrics
from keras.utils import generic_utils
from keras.optimizers import Adam
from keras.models import load_model
import os

import sys
import vis_utils
#os.environ['CUDA_VISIBLE_DEVICES']='1'
import cv2
import tensorflow as tf
import keras.backend as K
import re


def log_losses(progressBar, tensorBoardWriter, logger, loss_names, loss_vals, iter_count):
	if not isinstance(loss_vals, list):  # occurs when model only has one loss
		loss_vals = [loss_vals]

	if progressBar is not None:
		progressBar.add(1, values=[(loss_names[i], loss_vals[i]) for i in range(len(loss_vals))])

	if logger is not None:
		logger.debug(', '.join(['{}: {}'.format(loss_names[i], loss_vals[i]) for i in range(len(loss_vals))]))

	if tensorBoardWriter is not None:
		for i in range(len(loss_names)):
			tensorBoardWriter.add_summary(
				tf.Summary(value=[tf.Summary.Value(tag=loss_names[i], simple_value=loss_vals[i]), ]), iter_count)
			if i >= len(loss_vals):
				break
	
