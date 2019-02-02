import numpy as np
import tensorflow as tf


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