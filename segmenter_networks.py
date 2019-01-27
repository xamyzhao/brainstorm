import sys

import keras.backend as K
import numpy as np
from keras.layers import Input, Reshape, Lambda, Activation, Multiply
from keras.layers.convolutional import UpSampling3D
from keras.layers.pooling import MaxPooling3D
from keras.models import Model
import keras.metrics as keras_metrics
import keras.losses as keras_losses

from networks import basic_networks

sys.path.append('../voxelmorph-sandbox/util')
from spatial_transforms import Dense2DSpatialTransformer, Dense3DSpatialTransformer

from networks.transform_network_utils import affineWarp, RandFlow, DilateAndBlur
import classification_utils

sys.path.append('../medipy-lib')
import medipy.metrics as medipy_metrics

def eval_seg_sas_from_gen(sas_model, atlas_vol, atlas_labels, 
		eval_gen, label_mapping, n_eval_examples, batch_size, logger=None):
	img_shape = atlas_vol.shape[1:]

	from networks import transform_network_utils
	seg_warp_model = transform_network_utils.warp_model(
		img_shape=img_shape,
		interp_mode='nearest',
		indexing='xy',
	)

	from keras.models import Model
	from keras.layers import Input, Lambda, Activation
	from keras.optimizers import Adam
	n_labels = len(label_mapping)
	#true_in = Input(pred_shape[0:-1] + (n_labels,))	
	warped_in = Input(img_shape[0:-1] + (n_labels,))
	warped = Activation('softmax')(warped_in)
#	loss = Lambda(lambda x:tf.reduce_mean(keras_losses.categorical_crossentropy(x[0], x[1])))([true_in, warped])
#	ce_model = Model(inputs=[true_in, warped_in], outputs=loss, name='ce_model')	
	ce_model = Model(inputs=[warped_in], outputs=[warped], name='ce_model')
	ce_model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001))

	# test metrics: categorical cross-entropy and dice
	dice_per_label = np.zeros((n_eval_examples, len(label_mapping))) 
	cces = np.zeros((n_eval_examples,))
	accs = np.zeros((n_eval_examples,))
	for bi in range(n_eval_examples):
		if logger is not None:
			logger.debug('Testing on subject {} of {}'.format(bi, n_eval_examples))
		else:
			print('Testing on subject {} of {}'.format(bi, n_eval_examples))
		X, Y = next(eval_gen)
		Y_oh = classification_utils.labels_to_onehot(Y, label_mapping=label_mapping)

		warped, warp = sas_model.predict([atlas_vol, X])
			
		# warp our source models according to the predicted flow field. get rid of channels
		preds_batch = seg_warp_model.predict([atlas_labels[..., np.newaxis], warp])[..., 0]
		preds_oh = classification_utils.labels_to_onehot(preds_batch, label_mapping=label_mapping)
		'''
		preds_batch, cce = segment_vol_by_slice(
			segmenter_model, X, label_mapping=label_mapping, batch_size=batch_size,
			Y_oh=Y_oh, compute_cce=True,
		)
		'''
		cce = np.mean(ce_model.evaluate(preds_oh, Y_oh, verbose=False))
		subject_dice_per_label = medipy_metrics.dice(
			Y, preds_batch, labels=label_mapping)

		nonbkgmap = (Y > 0)# * (preds_batch > 0) # only consider true labels
		acc = np.sum(((Y == preds_batch) * nonbkgmap).astype(int)) / np.sum(nonbkgmap).astype(float)
		print(acc)
		dice_per_label[bi] = subject_dice_per_label
		cces[bi] = cce
		accs[bi] = acc
	
	if logger is not None:
		logger.debug('Dice per label: {}, {}'.format(label_mapping, dice_per_label))
		logger.debug('Mean dice (no bkg): {}'.format(np.mean(dice_per_label[:, 1:])))
		logger.debug('Mean CE: {}'.format(np.mean(cces)))
		logger.debug('Mean accuracy: {}'.format(np.mean(accs)))
	else:
		print('Dice per label: {}, {}'.format(label_mapping, dice_per_label))
		print('Mean dice (no bkg): {}'.format(np.mean(dice_per_label[:, 1:])))
		print('Mean CE: {}'.format(np.mean(cces)))
		print('Mean accuracy: {}'.format(np.mean(accs)))
	return cces, dice_per_label, accs

def eval_seg_from_gen(segmenter_model, 
		eval_gen, label_mapping, n_eval_examples, batch_size, logger=None):
	# test metrics: categorical cross-entropy and dice
	dice_per_label = np.zeros((n_eval_examples, len(label_mapping))) 
	cces = np.zeros((n_eval_examples,))
	accs = np.zeros((n_eval_examples,))
	for bi in range(n_eval_examples):
		if logger is not None:
			logger.debug('Testing on subject {} of {}'.format(bi, n_eval_examples))
		else:
			print('Testing on subject {} of {}'.format(bi, n_eval_examples))
		X, Y = next(eval_gen)
		Y_oh = classification_utils.labels_to_onehot(Y, label_mapping=label_mapping)
		preds_batch, cce = segment_vol_by_slice(
			segmenter_model, X, label_mapping=label_mapping, batch_size=batch_size,
			Y_oh=Y_oh, compute_cce=True,
		)
		subject_dice_per_label = medipy_metrics.dice(
			Y, preds_batch, labels=label_mapping)

		nonbkgmap = (Y > 0)# * (preds_batch > 0) # only consider true labels
		acc = np.sum(((Y == preds_batch) * nonbkgmap).astype(int)) / np.sum(nonbkgmap).astype(float)
		print(acc)
		dice_per_label[bi] = subject_dice_per_label
		cces[bi] = cce
		accs[bi] = acc
	
	if logger is not None:
		logger.debug('Dice per label: {}, {}'.format(label_mapping, np.mean(dice_per_label, axis=0).tolist()))
		logger.debug('Mean dice (no bkg): {}'.format(np.mean(dice_per_label[:, 1:])))
		logger.debug('Mean CE: {}'.format(np.mean(cces)))
		logger.debug('Mean accuracy: {}'.format(np.mean(accs)))
	else:
		print('Dice per label: {}, {}'.format(label_mapping, np.mean(dice_per_label, axis=0).tolist()))
		print('Mean dice (no bkg): {}'.format(np.mean(dice_per_label[:, 1:])))
		print('Mean CE: {}'.format(np.mean(cces)))
		print('Mean accuracy: {}'.format(np.mean(accs)))
	return cces, dice_per_label, accs


def segment_vol_by_slice(segmenter_model, X, label_mapping, batch_size=8, Y_oh=None, compute_cce=False):
	'''
	Segments a 3D volume by running a per-slice segmenter on batches of slices
	:param segmenter_model:
	:param X: 3D volume, we assume this has a batch size of 1
	:param label_mapping:
	:param batch_size:
	:return:
	'''
	n_slices = X.shape[2]
	n_labels = len(label_mapping)
	preds = np.zeros(X.shape[:-1])
	n_batches = int(np.ceil(float(n_slices) / batch_size))
	
	cce_total = 0.
	for sbi in range(n_batches):
		# slice in z, then make slices into batch
		X_batched_slices = np.transpose(
				X[0, :, :, sbi * batch_size: min(n_slices, (sbi + 1) * batch_size)],
				(2, 0, 1, 3))

		preds_slices_oh = segmenter_model.predict(X_batched_slices)
		if compute_cce:
			'''
			slice_cce = K.eval(keras_losses.categorical_crossentropy(
				K.variable(
					np.transpose(Y_oh[0, :, :, sbi * batch_size : min(n_slices, (sbi + 1) * batch_size)], (2, 0, 1, 3))
				), 
				K.variable(preds_slices_oh)))
			'''
			slice_cce = segmenter_model.evaluate(
				X_batched_slices,
				np.transpose(Y_oh[0, :, :, sbi * batch_size : min(n_slices, (sbi + 1) * batch_size)], (2, 0, 1, 3)),
				verbose=False)[0]
			#print(slice_cce)

			# we want an average over slices, so make sure we count the correct number in the batch 
			cce_total += slice_cce * X_batched_slices.shape[0]
		# convert onehot to labels and assign to preds volume
		preds[0, :, :, sbi * batch_size: min(n_slices, (sbi + 1) * batch_size)] \
			= np.transpose(classification_utils.onehot_to_labels(
			preds_slices_oh, label_mapping=label_mapping), (1, 2, 0))
	if compute_cce:
		return preds, cce_total / float(n_slices)
	else:
		return preds




def segmenter_unet(img_shape, n_labels, params, model_name='segmenter_unet',
				   use_residuals=False, activation='softmax'):
	n_dims = len(img_shape) - 1
	x_in = Input(img_shape, name='img_input')

	if 'nf_dec' not in params.keys():
		params['nf_dec'] = list(reversed(params['nf_enc']))

	if n_dims == 2:
		x = basic_networks.unet2D(x_in, img_shape, n_labels,
								  nf_enc=params['nf_enc'],
								  nf_dec=params['nf_dec'],
								  n_convs_per_stage=params['n_convs_per_stage'],
								  use_maxpool=params['use_maxpool'],
								  include_residual=params['use_residuals'])
	elif n_dims == 3:
		x = basic_networks.unet3D(x_in, img_shape, n_labels,
								  nf_enc=params['nf_enc'],
								  nf_dec=params['nf_dec'],
								  n_convs_per_stage=params['n_convs_per_stage'],
								  use_maxpool=params['use_maxpool'],
								  include_residual=params['use_residuals'])

	if activation is not None:
		seg = Activation(activation)(x)
	else:
		seg = x

	return Model(inputs=[x_in], outputs=seg, name=model_name)


def segmenter_unet_aug(img_shape, n_labels, n_channels, model_name='segmenter_unet', n_convs_per_stage=1, use_residuals=False):
	n_dims = len(img_shape)-1

	x_in = Input(img_shape, name='img_input')
	T_in = Input((2,3), name='transform_input')
	x = Lambda(lambda x: affineWarp(x[0], x[1], 1.), output_shape=img_shape, name='lambda_transform_input')([x_in, T_in])

	x = basic_networks.unet2D(x, img_shape, n_labels,
							  nf_enc=n_channels, n_convs_per_stage=n_convs_per_stage,
							  include_residual=use_residuals)

	seg = Activation('softmax', name='activation_softmax')(x)
	return Model(inputs=[x_in, T_in], outputs=[seg], name=model_name)
	


def segmenter_unet_prewarp(img_shape, n_labels, n_channels, model_name='segmenter_unet', n_convs_per_stage=1, use_residuals=False):
	n_dims = len(img_shape)-1

	x_in = Input(img_shape, name='img_input')
	y_in = Input(img_shape[:-1] + (1,), name='label_input')
	flow_in = Input(img_shape[:-1] + (n_dims,), name='flow_input')

	if n_dims == 2:
		x_warped = Dense2DSpatialTransformer('linear', name='densespatialtransformer_img')([x_in, flow_in])
		y_warped = Dense2DSpatialTransformer('nearest', name='densespatialtransformer_label')([y_in, flow_in])
		y_warped_oh = Lambda(lambda x:K.cast(K.one_hot(K.cast(x[:,:,:,0], dtype='int32'), num_classes=n_labels), dtype='float32'),
			name='lambda_convertoh', output_shape=img_shape[:-1] + (n_labels,))(y_warped)
		x = basic_networks.unet2D(x_warped, img_shape, n_labels, nf_enc=n_channels,
								  n_convs_per_stage=n_convs_per_stage, include_residual=use_residuals)
	elif n_dims == 3:
		x_warped = Dense3DSpatialTransformer('linear', name='densespatialtransformer_img')([x_in, flow_in])
		y_warped = Dense3DSpatialTransformer('nearest', name='densespatialtransformer_label')([y_in, flow_in])
		y_warped_oh = Lambda(lambda x:K.cast(K.one_hot(K.cast(x[:,:,:,:,0], dtype='int32'), num_classes=n_labels), dtype='float32'),
			name='lambda_convertoh', output_shape=img_shape[:-1] + (n_labels,))(y_warped)
		x = basic_networks.unet3D(x_warped, img_shape, n_labels, nf_enc=n_channels,
								  n_convs_per_stage=n_convs_per_stage, include_residual=use_residuals)

	seg = Activation('softmax', name='activation_softmax')(x)
	return Model(inputs=[x_in, y_in, flow_in], outputs=[seg, x_warped, y_warped_oh], name=model_name)
	

def segmenter_unet_randwarp(img_shape,
							n_labels,
							segmenter_prewarp_model,
							model_name='segmenter_unet',
							flow_sigma=10,
							blur_sigma=5):
	n_dims = len(img_shape)-1

	x_in = Input(img_shape, name='img_input_randwarp')
	y_in = Input(img_shape[:-1] + (1,), name='label_input_randwarp')
	flow_in = Input(img_shape[:-1] + (n_dims,), name='flow_input_randwarp')  # a hack to make the random flow the correct shape
	if n_dims == 3:
		flow = MaxPooling3D(2)(flow_in)
		flow = MaxPooling3D(2)(flow)
		blur_sigma = int(np.ceil(blur_sigma/4.))
	else:
		flow = flow_in
	
	flow = RandFlow(name='randflow', img_shape=img_shape, blur_sigma=blur_sigma, flow_sigma=flow_sigma)(flow)
	if n_dims == 3:
		flow = UpSampling3D(2)(flow)
		flow = UpSampling3D(2)(flow)

	seg, x_warped, y_warped_oh = segmenter_prewarp_model([x_in, y_in, flow])
	y_warped_oh = Reshape(img_shape[:-1] + (n_labels,), name='lambda_convertoh')(y_warped_oh)

	return Model(inputs=[x_in, y_in, flow_in], outputs=[seg, x_warped, y_warped_oh], name=model_name)


def segmenter_unet_smartwarp(img_shape,
							 n_labels,
							 segmenter_prewarp_model,
							 model_name='segmenter_unet',
							 min_flow_sigma=10,
							 max_flow_sigma=30,
							 blur_sigma=5,
							 dilate_kernel_size=10):
	n_dims = len(img_shape) - 1

	x_in = Input(img_shape, name='img_input')
	y_in = Input(img_shape[:-1] + (1,), name='label_input')
	flow_in = Input(img_shape[:-1] + (n_dims,), name='flow_input')  # a hack to make the random flow the correct shape
	errormap_in = Input(img_shape[:-1] + (1,), name='errormap_input')
	baseline_mag = Input((1,), name='baselinemag_input'	)

	if n_dims == 3:
		flow = MaxPooling3D(2)(flow_in)
		flow = MaxPooling3D(2)(flow)
	else:
		flow = flow_in
	flow_rand = RandFlow(name='randflow',
	                     img_shape=img_shape,
	                     blur_sigma=blur_sigma,
	                     flow_sigma=min_flow_sigma)(flow)
	if n_dims == 3:
		flow_rand = UpSampling3D(2)(flow_rand)
		flow_rand = UpSampling3D(2)(flow_rand)
	flow_mult = DilateAndBlur(name='dilateandblur',
	                          img_shape=img_shape,
	                          dilate_kernel_size=dilate_kernel_size,
	                          blur_sigma=blur_sigma,
	                          flow_sigma=float(max_flow_sigma) / min_flow_sigma)([errormap_in, baseline_mag])

	flow = Multiply(name='smartflow')([flow_rand, flow_mult])
	seg, x_warped, y_warped_oh = segmenter_prewarp_model([x_in, y_in, flow])
	#x_warped = Reshape(img_shape, name='densespatialtransformer_img')(x_warped)
	#y_warped = Reshape(img_shape[:-1] + (1,), name='densespatialtransformer_label')(y_warped)
	y_warped_oh = Reshape(img_shape[:-1] + (n_labels,), name='lambda_convertoh')(y_warped_oh)

	return Model(inputs=[x_in, y_in, flow_in, errormap_in, baseline_mag], outputs=[seg, x_warped, y_warped_oh], name=model_name)


