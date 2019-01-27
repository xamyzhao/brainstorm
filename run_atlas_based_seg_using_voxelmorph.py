import os
import sys

import json
import mri_loader

#sys.path.append('/afs/csail.mit.edu/u/x/xamyzhao/LPAT')
sys.path.append('/afs/csail.mit.edu/u/x/xamyzhao')
from cnn_utils import file_utils, vis_utils

import cv2
import tensorflow as tf

import keras.backend as K
import keras.losses as keras_losses

import argparse
import time

#from experiments_VTE import GLTExperimentClass
from keras.models import load_model
import brainstorm_networks

import numpy as np
#import medipy_metrics
import classification_utils
import IPython
import PIL
import vte_runner
import vis_utils

sys.path.append('/afs/csail.mit.edu/u/x/xamyzhao/medipy-lib')
import medipy.metrics as medipy_metrics

'''
Script to use a pretrained voxelmorph model in atlas-based segmentation of 50 test subjects
'''

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('-gpu', nargs='*', type=int, help='unmapped gpu to use (i.e. use 3 for gpu 0 on ephesia)',
	                default=1)
	ap.add_argument('-from_dir', nargs='?', default=None, help='Load experiment from dir instead of by params')
	ap.add_argument('-data', nargs='?', default=None, help='Dataset key to evaluate on')
	ap.add_argument('-debug', action='store_true', help='Load fewer test examples, for debugging', default=False)
	ap.add_argument('-warpoh', action='store_true', help='Warp the one-hot representation of the segmentation rather than labels', default=False)
	ap.add_argument('-bck', action='store_true', help='Compute backward dice instead', default=False)
	ap.add_argument('-n', nargs='?', type=int, help='Number of test examples to load', default=None)

	args = ap.parse_args()

	# set gpu id and tf settings
	os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(g) for g in args.gpu])
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	K.tensorflow_backend.set_session(tf.Session(config=config))

	# check if our desired model is in the voxelmorph folder
	backward_model = 'bck' in os.path.basename(args.from_dir) or args.bck
	#use_glt_model = not ('voxelmorph' in os.path.dirname(args.from_dir))
	use_glt_model = False

	if args.data is not None:
		data_params = vte_runner.named_vte_data_params[args.data]
	else:
		print('Must specify what dataset to test on!')
		sys.exit()

	load_data_params = data_params.copy()
	load_data_params['n_unlabeled'] = 1  # just load one for training, we only want the validation set
	load_data_params['load_vols'] = False
	ds = mri_loader.MRIDataset(load_data_params)
	_ = ds.load_dataset(debug=False)
	vol_shape = (160, 192, 224, 1)

	if use_glt_model:
		exp = GLTExperimentClass.ExperimentGlobalLocalTransforms(data_params, arch_params)

		# do this so the loggers work
		model_name, exp_dir, figures_dir, logs_dir, models_dir = file_utils.make_output_dirs(exp.get_model_name(),
																							 exp_root='./experiments/',
																							 prompt_delete=False)
		exp.save_exp_info(exp_dir, figures_dir, models_dir, logs_dir)

		# we don't actually want the full dataset here
		exp.load_data(load_fewer=True)
		indexing = 'ij'

	if not use_glt_model:
		sys.path.append('../voxelmorph')
		sys.path.append('../neuron')
		import neuron.layers as nrn_layers
		import neuron.utils as nrn_utils
		sys.path.append('../voxelmorph-sandbox')
		import voxelmorph.networks as vm_networks
		from voxelmorph.dense_3D_spatial_transformer import Dense3DSpatialTransformer
		#from functools import partial
		import functools

		if 'ij' in os.path.basename(args.from_dir):
			indexing = 'ij'
		else:
			indexing = 'xy'

		voxelmorph_model = load_model(
			args.from_dir,
			custom_objects={'Dense3DSpatialTransformer': Dense3DSpatialTransformer, 
				'tf': tf,
				'VecInt': nrn_layers.VecInt,
				'SpatialTransformer': functools.partial(nrn_layers.SpatialTransformer, indexing=indexing),
				'nrn_utils': nrn_utils,
				'nrn_layers': nrn_layers,
			},
			compile=False,
		)
		voxelmorph_model.summary()
	elif 'GLT' in exp.model_name:
		exp.create_models()
		exp.load_models(models_dir, 'latest')
		voxelmorph_model = exp.unet_flow

	vol_warp_model = brainstorm_networks.warp_model(
		img_shape=vol_shape,
		interp_mode='linear'
	)

	label_mapping = vte_runner.voxelmorph_labels
	n_labels = len(label_mapping)

	if args.warpoh:
		pred_shape = vol_shape[:-1] + (n_labels,)
		seg_warp_model = brainstorm_networks.warp_model(
			img_shape=pred_shape,
			interp_mode='linear',
			indexing=indexing,
		)
	else:
		pred_shape = vol_shape[:-1] + (1,)
		seg_warp_model = brainstorm_networks.warp_model(
			img_shape=pred_shape,
			interp_mode='nearest',
			indexing=indexing,
		)

	from keras.models import Model
	from keras.layers import Input, Lambda, Activation
	from keras.optimizers import Adam
	#true_in = Input(pred_shape[0:-1] + (n_labels,))	
	warped_in = Input(pred_shape[0:-1] + (n_labels,))
	warped = Activation('softmax')(warped_in)
#	loss = Lambda(lambda x:tf.reduce_mean(keras_losses.categorical_crossentropy(x[0], x[1])))([true_in, warped])
#	ce_model = Model(inputs=[true_in, warped_in], outputs=loss, name='ce_model')	
	ce_model = Model(inputs=[warped_in], outputs=[warped], name='ce_model')
	ce_model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001))

	# load each subject, then evaluate each slice
	n_test_examples = len(ds.files_labeled_valid)
	if args.debug:
		n_test_examples = 10
	elif args.n is not None:
		n_test_examples = args.n

	dice_per_label = np.zeros((len(label_mapping, )))
	total_ce = 0.
	total_acc = 0.

	source_X = ds.vols_labeled_train[[0]]
	source_Y = ds.segs_labeled_train[[0]]
	if not args.warpoh:
		source_Y = source_Y[..., np.newaxis]
	else:
		source_Y = classification_utils.labels_to_onehot(source_Y, label_mapping=label_mapping)

	print('Source subject: {}'.format(ds.files_labeled_train))
	out_ims = [None] * n_test_examples

	start = time.time()
	test_gen = ds.gen_vols_batch(['labeled_valid'], batch_size=1, randomize=False, 
			load_segs=True,
			convert_onehot=args.warpoh,
			label_mapping=label_mapping
		)

	for bi in range(n_test_examples):
		if bi % 10 == 0:
			print('Evaluating test subject {} of {}'.format(bi, n_test_examples))
		target_X, target_Y = next(test_gen)

		if not args.warpoh:
			target_Y = target_Y[..., np.newaxis]

		print(ds.files_labeled_valid[bi])

		n_slices = target_X.shape[2]

		if backward_model:  # backward in that it has learned to go from the target vols (in the dataset) to source vols
			print('Evaluating backward registration (target to source)')
			moving = target_X
			fixed = source_X
			moving_Y = target_Y
			fixed_Y = source_Y
		else:
			moving = source_X
			fixed = target_X
			moving_Y = source_Y
			fixed_Y = target_Y

		preds = voxelmorph_model.predict([moving, fixed])

		if use_glt_model:
			# GLT rather than voxelmorph
			warped, warp = preds[-1], preds[0]
		else:
			if 'bidir' in args.from_dir and 'fwd' not in args.from_dir and not 'bck' in args.from_dir:
				warped, _, warp, warp_back = preds
			elif 'ij' in args.from_dir:
				warped, warp = preds
			elif 'wrapper' not in args.from_dir:
				warped, warp = preds
			else:
				warp, warped = preds

		# also warp segmentations, then compute losses
		warped_Y = seg_warp_model.predict([moving_Y, warp])
		if not args.warpoh:
			fixed_oh = classification_utils.labels_to_onehot(fixed_Y, label_mapping=label_mapping)
			warped_oh = classification_utils.labels_to_onehot(warped_Y, label_mapping=label_mapping)
		else:
			fixed_oh = fixed_Y
			warped_oh = warped_Y
		ce = np.mean(ce_model.evaluate(warped_oh.astype(np.float32), fixed_oh.astype(np.float32), verbose=False))
		
		if args.warpoh:
			warped_labels = classification_utils.onehot_to_labels(warped_Y, label_mapping=label_mapping)
			fixed_labels = classification_utils.onehot_to_labels(fixed_Y, label_mapping=label_mapping)
		else:
			warped_labels = warped_Y
			fixed_labels = fixed_Y
		subject_dice_per_label = medipy_metrics.dice(fixed_labels, warped_labels, labels=label_mapping)
		nonbkgmap = (fixed_labels > 0) * (warped_labels > 0)
		acc = np.sum(((fixed_labels == warped_labels) * nonbkgmap).astype(int)) / np.sum(nonbkgmap).astype(float)
		print(acc)
		#slice_idx = np.random.choice(source_X.shape[-2], 1)[0]
		slice_idx = 112
		out_ims[bi] = vis_utils.label_ims(
				np.concatenate([
						source_X[:, :, :, slice_idx], 
						target_X[:, :, :, slice_idx], 
						warped[:, :, :, slice_idx],
						warped_oh[:, :, :, slice_idx, [12]],
						fixed_oh[:, :, :, slice_idx, [12]],
					],axis=0),
				'slice {}, dice {}'.format(slice_idx, np.mean(subject_dice_per_label[1:])),
				concat_axis=1
			)

		dice_per_label += subject_dice_per_label
		total_ce += ce
		total_acc += acc
	dice_per_label /= float(n_test_examples)
	total_ce /= float(n_test_examples)
	total_acc /= float(n_test_examples)
	print('Mean dice: {}'.format(np.mean(dice_per_label[1:])))
	print('Mean CCE: {}'.format(total_ce))
	print('Mean acc: {}'.format(total_acc))
	print('Took {}, {} per subject'.format(time.time() - start, round((time.time() - start) / float(n_test_examples), 2)))
	print('Dice per label: {}'.format(dice_per_label))
	#for li, l in enumerate(ds.label_mapping):
		#print('{}: {}'.format(l, dice_per_label[li]))

	if not use_glt_model:
		im_file = 'abs_{}_meandice_{}.png'.format(
			os.path.basename(args.from_dir), int(round(np.mean(dice_per_label[1:]) * 100.)))
	else:
		im_file = 'abs_{}_epoch{}_meandice_{}.png'.format(
			exp.model_name, exp.latest_epoch, int(round(np.mean(dice_per_label[1:]) * 100.)))

	with open('experiments/{}_SAS.txt'.format(
			os.path.splitext(os.path.basename(args.from_dir))[0]), 'w') as f:
		f.writelines(','.join([str(d) for d in dice_per_label.tolist()]))
				
	
	cv2.imwrite(im_file,
		np.concatenate(out_ims, axis=0))
