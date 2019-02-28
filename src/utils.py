import sys

import cv2
import numpy as np
import tensorflow as tf

from src import networks

sys.path.append('../pynd-lib')
from pynd import segutils

sys.path.append('../evolving_wilds')
from cnn_utils import classification_utils

sys.path.append('../medipy-lib')
import medipy.metrics as medipy_metrics


def log_losses(progressBar, tensorBoardWriter, logger, loss_names, loss_vals, iter_count):
    '''
    Writes loss names and vals to keras progress bar, tensorboard, and a python logger

    :param progressBar: keras progbar object
    :param tensorBoardWriter: tensorboard writer object
    :param logger: python logger
    :param loss_names: list of strings
    :param loss_vals: list of numbers
    :param iter_count: integer representing current iteration
    :return:
    '''
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
    '''
    Evaluates a single-atlas segmentation method on a bunch of evaluation volumes.
    :param sas_model: spatial transform model used for SAS. Can be voxelmorph.
    :param atlas_vol: atlas volume
    :param atlas_labels: atlas segmentations
    :param eval_gen: generator that yields vols_valid, segs_valid batches
    :param label_mapping: list of label ids that will appear in segs, ordered by how they map to channels
    :param n_eval_examples: total number of examples to evaluate
    :param batch_size: batch size to use in evaluation
    :param logger: python logger if we want to log messages
    :return:
    '''
    img_shape = atlas_vol.shape[1:]

    seg_warp_model = networks.warp_model(
        img_shape=img_shape,
        interp_mode='nearest',
        indexing='xy',
    )

    from keras.models import Model
    from keras.layers import Input, Activation
    from keras.optimizers import Adam
    n_labels = len(label_mapping)

    warped_in = Input(img_shape[0:-1] + (n_labels,))
    warped = Activation('softmax')(warped_in)

    ce_model = Model(inputs=[warped_in], outputs=[warped], name='ce_model')
    ce_model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001))

    # test metrics: categorical cross-entropy and dice
    dice_per_label = np.zeros((n_eval_examples, len(label_mapping)))
    cces = np.zeros((n_eval_examples,))
    accs = np.zeros((n_eval_examples,))
    all_ids = []
    for bi in range(n_eval_examples):
        if logger is not None:
            logger.debug('Testing on subject {} of {}'.format(bi, n_eval_examples))
        else:
            print('Testing on subject {} of {}'.format(bi, n_eval_examples))
        X, Y, _, ids = next(eval_gen)
        Y_oh = classification_utils.labels_to_onehot(Y, label_mapping=label_mapping)

        warped, warp = sas_model.predict([atlas_vol, X])

        # warp our source models according to the predicted flow field. get rid of channels
        if Y.shape[-1] == 1:
            Y = Y[..., 0]
        preds_batch = seg_warp_model.predict([atlas_labels[..., np.newaxis], warp])[..., 0]
        preds_oh = classification_utils.labels_to_onehot(preds_batch, label_mapping=label_mapping)

        cce = np.mean(ce_model.evaluate(preds_oh, Y_oh, verbose=False))
        subject_dice_per_label = medipy_metrics.dice(
            Y, preds_batch, labels=label_mapping)

        nonbkgmap = (Y > 0)
        acc = np.sum(((Y == preds_batch) * nonbkgmap).astype(int)) / np.sum(nonbkgmap).astype(float)
        print(acc)
        dice_per_label[bi] = subject_dice_per_label
        cces[bi] = cce
        accs[bi] = acc
        all_ids += ids

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
    return cces, dice_per_label, accs, all_ids


def eval_seg_from_gen(segmenter_model,
        eval_gen, label_mapping, n_eval_examples, batch_size, logger=None):
    '''
    Evaluates accuracy of a segmentation CNN
    :param segmenter_model: keras model for segmenter
    :param eval_gen: genrator that yields vols_valid, segs_valid
    :param label_mapping: list of label ids, ordered by how they map to channels
    :param n_eval_examples: total number of volumes to evaluate
    :param batch_size: batch size (number of slices per batch)
    :param logger: python logger (optional)
    :return:
    '''

    # test metrics: categorical cross-entropy and dice
    dice_per_label = np.zeros((n_eval_examples, len(label_mapping)))
    cces = np.zeros((n_eval_examples,))
    accs = np.zeros((n_eval_examples,))
    all_ids = []
    for bi in range(n_eval_examples):
        if logger is not None:
            logger.debug('Testing on subject {} of {}'.format(bi, n_eval_examples))
        else:
            print('Testing on subject {} of {}'.format(bi, n_eval_examples))
        X, Y, _, ids = next(eval_gen)
        Y_oh = classification_utils.labels_to_onehot(Y, label_mapping=label_mapping)
        preds_batch, cce = segment_vol_by_slice(
            segmenter_model, X, label_mapping=label_mapping, batch_size=batch_size,
            Y_oh=Y_oh, compute_cce=True,
        )
        subject_dice_per_label = medipy_metrics.dice(
            Y, preds_batch, labels=label_mapping)

        # only consider pixels where the gt label is not bkg (if we count bkg, accuracy will be very high)
        nonbkgmap = (Y > 0)

        acc = np.sum(((Y == preds_batch) * nonbkgmap).astype(int)) / np.sum(nonbkgmap).astype(float)

        print(acc)
        dice_per_label[bi] = subject_dice_per_label
        cces[bi] = cce
        accs[bi] = acc
        all_ids += ids

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
    return cces, dice_per_label, accs, all_ids


def segment_vol_by_slice(segmenter_model, X, label_mapping, batch_size=8, Y_oh=None, compute_cce=False):
    '''
    Segments a 3D volume by running a per-slice segmenter on batches of slices
    :param segmenter_model:
    :param X: 3D volume, we assume this has a batch size of 1
    :param label_mapping:
    :param batch_size:
    :return:
    '''
    n_slices = X.shape[-2]
    n_labels = len(label_mapping)
    preds = np.zeros(X.shape[:-1] + (1,))
    n_batches = int(np.ceil(float(n_slices) / batch_size))

    cce_total = 0.
    for sbi in range(n_batches):
        # slice in z, then make slices into batch
        X_batched_slices = np.transpose(
                X[0, :, :, sbi * batch_size: min(n_slices, (sbi + 1) * batch_size)],
                (2, 0, 1, 3))

        preds_slices_oh = segmenter_model.predict(X_batched_slices)
        if compute_cce:
            slice_cce = segmenter_model.evaluate(
                X_batched_slices,
                np.transpose(Y_oh[0, :, :, sbi * batch_size : min(n_slices, (sbi + 1) * batch_size)], (2, 0, 1, 3)),
                verbose=False)
            # if we have multiple losses, take the first one
            if isinstance(slice_cce, list):
                slice_cce = slice_cce[0]

            # we want an average over slices, so make sure we count the correct number in the batch
            cce_total += slice_cce * X_batched_slices.shape[0]
        # convert onehot to labels and assign to preds volume
        preds[0, :, :, sbi * batch_size: min(n_slices, (sbi + 1) * batch_size)] \
            = np.transpose(classification_utils.onehot_to_labels(
            preds_slices_oh, label_mapping=label_mapping), (1, 2, 0))[..., np.newaxis]
    if compute_cce:
        return preds, cce_total / float(n_slices)
    else:
        return preds

######################
# Visualization utils
######################
def draw_segs_on_slice(vol_slice, seg_slice,
                       include_labels=None,
                       colors=None,
                       draw_contours=False,
                       use_gradient_colormap=False):
    '''
    Overlays segmentations on a 2D slice.

    :param vol_slice: h x w image, the brain slice to overlay on top of
    :param seg_slice: h x w array, segmentations to overlay
        (in labels format, not one hot)
    :param include_labels: list, visualize only specific label values
    :param colors: n_labels x 3, specific colors to use for segmentations
    :param draw_contours: bool, visualize segmentations as contours
        rather than solid areas
    :param use_gradient_colormap: bool, create the colormap as a gradient of a
        single color rather than a rainbow

    :return: h x w x 3 image of brain slice with segmentations overlaid on top
    '''
    # if no labels are specified, simply visualize all unique label values
    if include_labels is None:
        include_labels = list(np.unique(seg_slice).astype(int))

    # if colors are not specified, make a color map
    if colors is None:
        if use_gradient_colormap:
            colors = make_cmap_gradient(
                len(include_labels) + 1, hue=0.5)
        else:
            colors = make_cmap_rainbow(
                len(include_labels) + 1)

    # make a new segmentation map with labels as ascending integers,
    # since this is what segutils expects
    pruned_slice = np.zeros(seg_slice.shape, dtype=int)
    for i, l in enumerate(include_labels):
        pruned_slice[seg_slice == l] = i + 1

    seg_im = segutils.seg_overlap(
        np.squeeze(vol_slice), pruned_slice,
        cmap=colors,
        do_contour=draw_contours)
    return seg_im


def overlay_segs_on_ims_batch(ims, segs,
                              include_labels,
                              draw_contours=False,
                              use_gradient_colormap=False,
                              subjects_axis=-1,
                              colormap=None,
                              ):

    # if the input is a single image, pretend it is a batch of size 1
    if len(ims.shape) == 2:
        ims = np.expand_dims(ims, -1)

    n_brains = ims.shape[subjects_axis]
    out_im = []

    for i in range(n_brains):
        curr_im = np.take(ims, i, axis=subjects_axis)
        curr_seg = np.take(segs, i, axis=subjects_axis)

        if len(segs.shape) > 2:
            curr_out_im = draw_segs_on_slice(
                curr_im, curr_seg,
                include_labels=include_labels,
                draw_contours=draw_contours,
                colors=colormap,
                use_gradient_colormap=use_gradient_colormap,
            )

        else:
            curr_out_im = draw_segs_on_slice(
                curr_im, segs,
                include_labels=include_labels,
                draw_contours=draw_contours,
                colors=colormap,
                use_gradient_colormap=use_gradient_colormap,
            )
        out_im.append(np.expand_dims(curr_out_im, axis=subjects_axis))
    out_im = np.concatenate(out_im, subjects_axis)

    return out_im

def make_cmap_gradient(nb_labels=256, hue=1.0):
    hue = hue * np.ones((nb_labels, 1))
    sat = np.reshape(np.linspace(1., 0., nb_labels, endpoint=True), hue.shape)
    colors = np.concatenate([hue, sat, np.ones((nb_labels, 1), dtype=np.float32)], axis=1) * 255
    colors = cv2.cvtColor(np.expand_dims(colors, 0).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)[0] / 255.0
    return colors



def make_cmap_rainbow(nb_labels=256):
    '''
    Creates a rainbow colormap (with an RGB color value for each label)

    :param nb_labels:
    :return:
    '''
    # make a rainbow gradient
    hue = np.expand_dims(np.linspace(0, 0.6, nb_labels), 1).astype(np.float32)
    colors = np.concatenate([hue, np.ones((nb_labels, 2), dtype=np.float32)], axis=1) * 255

    # convert to 0-1 range RGB
    colors = cv2.cvtColor(np.expand_dims(colors, 0).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)[0] / 255.0
    return colors

