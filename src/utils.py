import math
import os
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.optimizers import Adam
import textwrap

from src import networks

import medipy.metrics as medipy_metrics
import pynd.segutils as pynd_segutils

#############################
# File utils
#############################
def get_latest_epoch_in_dir(d, match_prefixes=()):
    model_files = [f for f in os.listdir(d) if f.endswith('.h5')]

    epoch_nums = [re.search('(?<=epoch)[0-9]*', os.path.basename(f)).group(0) for f in model_files]
    epoch_nums = list(set([int(n) for n in epoch_nums if n is not None and n is not '']))
    max_epoch_num = 0

    if len(epoch_nums) == 0:
        return None

    if len(match_prefixes) > 0:
        for n in reversed(epoch_nums):
            curr_filenames = [os.path.basename(f) for f in model_files if 'epoch{}'.format(n) in f]
            if np.all([np.any([p in f for f in curr_filenames]) for p in match_prefixes]) and n > max_epoch_num:
                max_epoch_num = n
    else:
        max_epoch_num = max(epoch_nums)

    return max_epoch_num


def make_output_dirs(experiment_base_name: str,
                     prompt_delete_existing: bool = True,
                     prompt_update_name: bool = True,
                     exp_root: str = './experiments/',
                     existing_exp_dir=None,
                     # for unit testing
                     debug_delete_input=None,
                     debug_rename_input=None
                     ):
    '''
    Creates the experiment directory (for storing log files, parameters) as well as subdirectories for
    files, logs and models.

    If a directory already exists for this experiment,

    :param experiment_base_name: desired name for the experiment
    :param prompt_delete_existing: if we find an existing directory with the same name,
        do we tell the user? if not, just continue in the existing directory by default
    :param prompt_update_name: if the new experiment name differs from the existing_exp_dir,
        do we try to rename the existing directory to the new naming scheme?
    :param exp_root: root directory for all experiments
    :param existing_exp_dir: previous directory (if any) of this experiment
    :return:
    '''

    do_rename = False

    if existing_exp_dir is None:
        # brand new experiment
        experiment_name = experiment_base_name
        target_exp_dir = os.path.join(exp_root, experiment_base_name)
    else:  # we are loading from an existing directory
        if re.search('_[0-9]*$', existing_exp_dir) is not None:
            # we are probably trying to load from a directory like experiments/<exp_name>_1,
            #  so we should track the experiment_name with the correct id
            experiment_name = os.path.basename(existing_exp_dir)
            target_exp_dir = os.path.join(exp_root, experiment_name)
        else:
            # we are trying to load from a directory, but the newly provided experiment name doesn't match.
            # this can happen when the naming scheme has changed
            target_exp_dir = os.path.join(exp_root, experiment_base_name)

            # if it has changed, we should prompt to rename the old experiment to the new one
            if prompt_update_name:
                target_exp_dir, do_rename = _prompt_rename(
                    existing_exp_dir, target_exp_dir, debug_rename_input)

                if do_rename: # we might have changed the model name to something that exists, so prompt if so
                    print('Renaming {} to {}!'.format(existing_exp_dir, target_exp_dir))
                    prompt_delete_existing = True
            else:
                target_exp_dir = existing_exp_dir # just assume we want to continue in the old one

            experiment_name = os.path.basename(target_exp_dir)

    print('Existing exp dir: {}'.format(existing_exp_dir))
    print('Target exp dir: {}'.format(target_exp_dir))

    figures_dir = os.path.join(target_exp_dir, 'figures')
    logs_dir = os.path.join(target_exp_dir, 'logs')
    models_dir = os.path.join(target_exp_dir, 'models')

    copy_count = 0

    # check all existing dirs with the same prefix (and all suffixes e.g. _1, _2)
    while os.path.isdir(figures_dir) or os.path.isdir(logs_dir) or os.path.isdir(models_dir):
        # list existing files
        if os.path.isdir(figures_dir):
            figure_files = [os.path.join(figures_dir, f) for f in os.listdir(figures_dir) if
                            f.endswith('.jpg') or f.endswith('.png')]
        else:
            figure_files = []

        # check for .log files
        if os.path.isdir(logs_dir):
            log_files = [os.path.join(logs_dir, l) for l in os.listdir(logs_dir) \
                         if os.path.isfile(os.path.join(logs_dir, l))] \
                        + [os.path.join(target_exp_dir, f) for f in os.listdir(target_exp_dir) if f.endswith('.log')]
        else:
            log_files = []

        # check for model files
        if os.path.isdir(models_dir):
            model_files = [os.path.join(models_dir, m) for m in os.listdir(models_dir) \
                           if os.path.isfile(os.path.join(models_dir, m))]
        else:
            model_files = []

        if prompt_delete_existing and (len(figure_files) > 0 or len(log_files) > 0 or len(model_files) > 0):
            # TODO: print some of the latest figures, logs and models so we can see what epoch
            # these experiments trained until
            print(
                'Remove \n\t{} figures from {}\n\t{} logs from {}\n\t{} models from {}?[y]es / [n]o (create new folder) / [C]ontinue existing / remove [m]odels too: [y/n/C/m]'.format(
                    len(figure_files), figures_dir, len(log_files), logs_dir, len(model_files), models_dir))

            if debug_delete_input:
                print('Debug input: {}'.format(debug_delete_input))
                choice = debug_delete_input
            else:
                choice = input().lower()

            remove_choices = ['yes', 'y', 'ye']
            make_new_choices = ['no', 'n']
            continue_choices = ['c', '']
            remove_models_too = ['m']

            if choice in remove_choices:
                for f in figure_files + log_files:
                    print('Removing {}'.format(f))
                    os.remove(f)
            elif choice in remove_models_too:
                for f in figure_files + log_files + model_files:
                    print('Removing {}'.format(f))
                    os.remove(f)
            elif choice in continue_choices:
                print('Continuing in existing folder...')
                break

            elif choice in make_new_choices:
                copy_count += 1
                experiment_name = experiment_base_name + '_{}'.format(copy_count)
                target_exp_dir = os.path.join(exp_root, experiment_name)

                figures_dir = os.path.join(exp_root, experiment_name, 'figures')
                logs_dir = os.path.join(exp_root, experiment_name, 'logs')
                models_dir = os.path.join(exp_root, experiment_name, 'models')
        else:
            break

    if do_rename:
        # simply rename the existing old_exp_dir to exp_dir, rather than creating a new one
        os.rename(existing_exp_dir, target_exp_dir)
    else:
        # create each directory
        if not os.path.isdir(target_exp_dir):
            os.mkdir(target_exp_dir)

    # make subdirectories if they do not exist already
    if not os.path.isdir(figures_dir):
        os.mkdir(figures_dir)
    if not os.path.isdir(logs_dir):
        os.mkdir(logs_dir)
    if not os.path.isdir(models_dir):
        os.mkdir(models_dir)

    return experiment_name, target_exp_dir, figures_dir, logs_dir, models_dir


def _prompt_rename(old_dir, new_dir, debug_input=None):
    print('Rename dir \n{} to \n{} [y/N]?'.format(old_dir, new_dir))

    if debug_input:
        print('Debug input: {}'.format(debug_input))
        choice = debug_input
    else:
        choice = input().lower()

    rename_choices = ['yes', 'y']

    if choice in rename_choices:
        return new_dir, True
    else:
        return old_dir, False



#############################
# Batch utils
#############################

def gen_batch(ims_data, labels_data,
              batch_size, randomize=False):
    '''
    :param ims_data: list of images, or an image.
    If a single image, it will be automatically converted to a list

    :param labels_data: list of other data (e.g. labels) that do not require
    image normalization or augmentation, but might need to be converted to onehot

    :param batch_size:
    :param randomize: bool to randomize indices per batch

    :return:
    '''

    # make sure everything is a list
    if not isinstance(ims_data, list):
        ims_data = [ims_data]

    # if we have labels that we want to generate from,
    # put everything into a list for consistency
    # (useful if we have labels and aux data)
    if labels_data is not None:
        if not isinstance(labels_data, list):
            labels_data = [labels_data]


    idxs = [-1]

    n_ims = ims_data[0].shape[0]

    while True:
        if randomize:
            idxs = np.random.choice(n_ims, batch_size, replace=True)
        else:
            idxs = np.linspace(idxs[-1] + 1, idxs[-1] + 1 + batch_size - 1, batch_size, dtype=int)
            restart_idxs = False
            while np.any(idxs >= n_ims):
                idxs[np.where(idxs >= n_ims)] = idxs[np.where(idxs >= n_ims)] - n_ims
                restart_idxs = True

        ims_batches = []
        for i, im_data in enumerate(ims_data):
            X_batch = im_data[idxs]

            if not X_batch.dtype == np.float32 and not X_batch.dtype == np.float64:
                X_batch = (X_batch / 255.).astype(np.float32)

            ims_batches.append(X_batch)

        if labels_data is not None:
            labels_batches = []
            for li, Y in enumerate(labels_data):
                if Y is None:
                    Y_batch = None
                else:
                    if isinstance(Y, np.ndarray):
                        Y_batch = Y[idxs]
                    else: # in case it's a list
                        Y_batch = [Y[idx] for idx in idxs]
                labels_batches.append(Y_batch)
        else:
            labels_batches = None

        if not randomize and restart_idxs:
            idxs[-1] = -1

        yield tuple(ims_batches) + tuple(labels_batches)

#############################
# Segmentation losses and metrics
############################
def onehot_to_labels(oh, n_classes=0, label_mapping=None):
    # assume oh is batch_size (x R x C) x n_labels
    if n_classes > 0 and label_mapping is None:
        label_mapping = np.arange(0, n_classes)
    elif n_classes == 0 and label_mapping is None:
        label_mapping = list(np.arange(0, oh.shape[-1]).astype(int))

    argmax_idxs = np.argmax(oh, axis=-1).astype(int)
    labels = np.reshape(np.asarray(label_mapping)[argmax_idxs.flatten()], oh.shape[:-1]).astype(type(label_mapping[0]))

    return labels

def labels_to_onehot(labels, n_classes=0, label_mapping=None):
    if labels is None:
        return labels
    # we can use either n_classes (which assumes labels from 0 to n_classes-1) or a label_mapping
    if label_mapping is None and n_classes == 0:
        label_mapping = list(np.unique(labels))
        n_classes = len(np.unique(labels))  # np.max(labels)
    elif n_classes > 0 and label_mapping is None:
        # infer label mapping from # of classes
        label_mapping = np.linspace(0, n_classes, n_classes, endpoint=False).astype(int).tolist()
    elif n_classes == 0 and label_mapping is not None:
        n_classes = len(label_mapping)

    if labels.shape[-1] == len(label_mapping) and np.max(labels) <= 1. and np.min(labels) >= 0.:
        # already onehot
        return labels

    if labels.shape[-1] == 1:
        labels = np.take(labels, 0, axis=-1)

    labels = np.asarray(labels)

    if len(label_mapping) == 2 and 0 in label_mapping and 1 in label_mapping and type(
            labels) == np.ndarray and np.array_equal(np.max(labels, axis=-1), np.ones((labels.shape[0],))):
        return labels

    labels_flat = labels.flatten()
    onehot_flat = np.zeros(labels_flat.shape + (n_classes,), dtype=int)
    for li in range(n_classes):
        onehot_flat[np.where(labels_flat == label_mapping[li]), li] = 1

    onehot = np.reshape(onehot_flat, labels.shape + (n_classes,)).astype(np.float32)
    return onehot

class SpatialSegmentSmoothness(object):
    def __init__(self, n_chans, n_dims,
                 warped_contours_layer_output=None,
                 lambda_i=1.
                 ):
        self.n_dims = n_dims
        self.warped_contours_layer_output = warped_contours_layer_output
        self.lambda_i = lambda_i

    def compute_loss(self, y_true, y_pred):
        loss = 0
        segments_mask = 1. - self.warped_contours_layer_output

        for d in range(self.n_dims):
            # we use x to indicate the current spatial dimension, not just the first
            dCdx = tf.gather(y_pred, tf.range(1, tf.shape(y_pred)[d + 1]), axis=d + 1) \
                   - tf.gather(y_pred, tf.range(0, tf.shape(y_pred)[d + 1] - 1), axis=d + 1)

            # average across spatial dims and color channels
            loss += tf.reduce_mean(tf.abs(dCdx * tf.gather(segments_mask, tf.range(1, tf.shape(y_pred)[d+1]), axis=d+1)))
        return loss


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
        Y_oh = labels_to_onehot(Y, label_mapping=label_mapping)

        warped, warp = sas_model.predict([atlas_vol, X])

        # warp our source models according to the predicted flow field. get rid of channels
        if Y.shape[-1] == 1:
            Y = Y[..., 0]
        preds_batch = seg_warp_model.predict([atlas_labels[..., np.newaxis], warp])[..., 0]
        preds_oh = labels_to_onehot(preds_batch, label_mapping=label_mapping)

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
        Y_oh = labels_to_onehot(Y, label_mapping=label_mapping)
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
            = np.transpose(onehot_to_labels(
            preds_slices_oh, label_mapping=label_mapping), (1, 2, 0))[..., np.newaxis]
    if compute_cce:
        return preds, cce_total / float(n_slices)
    else:
        return preds

######################
# Visualization utils
######################
def label_ims(ims_batch, labels=None,
              normalize=False,
              display_h=128):
    '''
    Displays a batch of matrices as an image.

    :param ims_batch: n_batches x h x w x c array of images.
    :param labels: optional labels. Can be an n_batches length list of tuples, floats or strings
    :param normalize: boolean to normalize any [min, max] to [0, 255]
    :param display_h: integer number of pixels for the height of each image to display
    :return: an image (h' x w' x 3) with elements of the batch organized into rows
    '''

    if len(ims_batch.shape) == 3 and ims_batch.shape[-1] == 3:
        # already an image
        return ims_batch

    batch_size, h, w = ims_batch.shape[:3]
    if len(ims_batch.shape) == 3:
        n_chans = 1
    else:
        n_chans = ims_batch.shape[-1]

    if type(labels) == list and len(labels) == 1:
        # only label the first image
        labels = labels + [''] * (batch_size - 1)
    elif labels is not None and not type(labels) == list and not type(labels) == np.ndarray:
        # replicate labels for each row in the batch
        labels = [labels] * batch_size

    scale_factor = display_h / float(h)

    # make sure we have a channels dimension
    if len(ims_batch.shape) < 4:
        ims_batch = np.expand_dims(ims_batch, 3)

    if normalize:
        flattened_dims = np.prod(ims_batch.shape[1:])

        X_spatially_flat = np.reshape(ims_batch, (batch_size, -1, n_chans))
        X_orig_min = np.min(X_spatially_flat, axis=1)
        X_orig_max = np.max(X_spatially_flat, axis=1)

        # now actually flatten and normalize across channels
        X_flat = np.reshape(ims_batch, (batch_size, -1))

        X_flat = X_flat - np.tile(np.min(X_flat, axis=1, keepdims=True), (1, flattened_dims))
        # avoid dividing by 0
        X_flat = X_flat / np.clip(
            np.tile(np.max(X_flat, axis=1, keepdims=True), (1, flattened_dims)), 1e-5, None)

        ims_batch = np.reshape(X_flat, ims_batch.shape)
        ims_batch = np.clip(ims_batch.astype(np.float32), 0., 1.)

        for i in range(batch_size):
            if labels is not None and len(labels) > 0:
                if labels[i] is not None:
                    labels[i] = '{},'.format(labels[i])
                else:
                    labels[i] = ''
                # show the min, max of each channel
                for c in range(n_chans):
                    labels[i] += '({:.2f}, {:.2f})'.format(round(X_orig_min[i, c], 2), round(X_orig_max[i, c], 2))
    else:
        ims_batch = np.clip(ims_batch, 0., 1.)

    if np.max(ims_batch) <= 1.0:
        ims_batch = ims_batch * 255.0

    out_im = []
    for i in range(batch_size):
        # convert grayscale to rgb if needed
        if len(ims_batch[i].shape) == 2:
            curr_im = np.tile(np.expand_dims(ims_batch[i], axis=-1), (1, 1, 3))
        elif ims_batch.shape[-1] == 1:
            curr_im = np.tile(ims_batch[i], (1, 1, 3))
        else:
            curr_im = ims_batch[i]

        # scale to specified display size
        if scale_factor > 2:  # if we are upsampling by a lot, nearest neighbor can look really noisy
            interp = cv2.INTER_NEAREST
        else:
            interp = cv2.INTER_LINEAR

        if not scale_factor == 1:
            curr_im = cv2.resize(curr_im, None, fx=scale_factor, fy=scale_factor, interpolation=interp)

        out_im.append(curr_im)

    out_im = np.concatenate(out_im, axis=0).astype(np.uint8)

    # draw text labels on images if specified
    font_size = 15
    max_text_width = int(17 * display_h / 128.)  # empirically determined

    if labels is not None and len(labels) > 0:
        im_pil = Image.fromarray(out_im)
        draw = ImageDraw.Draw(im_pil)

        for i in range(batch_size):
            if len(labels) > i:  # if we have a label for this image
                if type(labels[i]) == tuple or type(labels[i]) == list:
                    # format tuple or list nicely
                    formatted_text = ', '.join([
                        labels[i][j].decode('UTF-8') if type(labels[i][j]) == np.unicode_ \
                            else labels[i][j] if type(labels[i][j]) == str \
                            else str(round(labels[i][j], 2)) if isinstance(labels[i][j], float) \
                            else str(labels[i][j]) for j in range(len(labels[i]))])
                elif type(labels[i]) == float or type(labels[i]) == np.float32:
                    formatted_text = str(round(labels[i], 2))  # round floats to 2 digits
                elif isinstance(labels[i], np.ndarray):
                    # assume that this is a 1D array
                    curr_labels = np.squeeze(labels[i]).astype(np.float32)
                    formatted_text = np.array2string(curr_labels, precision=2, separator=',')
                    #', '.join(['{}'.format(
                    #	np.around(labels[i][j], 2)) for j in range(labels[i].size)])
                else:
                    formatted_text = '{}'.format(labels[i])

                font = ImageFont.truetype('Ubuntu-M.ttf', font_size)
                # wrap the text so it fits
                formatted_text = textwrap.wrap(formatted_text, width=max_text_width)


                for li, line in enumerate(formatted_text):
                    draw.text((5, i * display_h + 5 + 14 * li), line, font=font, fill=(50, 50, 255))

        out_im = np.asarray(im_pil)

    return out_im


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

    seg_im = pynd_segutils.seg_overlap(
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

def concatenate_with_pad(ims_list, pad_to_im_idx=None, axis=None, pad_val=0.):
    padded_ims_list = pad_images_to_size(ims_list, pad_to_im_idx, ignore_axes=axis, pad_val=pad_val)
    return np.concatenate(padded_ims_list, axis=axis)

def pad_images_to_size(ims_list, pad_to_im_idx=None, ignore_axes=None, pad_val=0.):
    if pad_to_im_idx is not None:
        pad_to_shape = ims_list[pad_to_im_idx].shape
    else:
        im_shapes = np.reshape([im.shape for im in ims_list], (len(ims_list), -1))
        pad_to_shape = np.max(im_shapes, axis=0).tolist()

    if ignore_axes is not None:
        if not isinstance(ignore_axes, list):
            ignore_axes = [ignore_axes]
        for a in ignore_axes:
            pad_to_shape[a] = None

    ims_list = [pad_or_crop_to_shape(im, pad_to_shape, border_color=pad_val) \
        for i, im in enumerate(ims_list)]
    return ims_list

def pad_or_crop_to_shape(
        I,
        out_shape,
        border_color=(255, 255, 255)):
    if not isinstance(border_color, tuple):
        n_chans = I.shape[-1]
        border_color = tuple([border_color] * n_chans)

    # an out_shape with a dimension value of None means just don't crop or pad in that dim
    border_size = [out_shape[d] - I.shape[d]
                   if out_shape[d] is not None else 0 for d in range(2)]

    if border_size[0] > 0:
        I = cv2.copyMakeBorder(I,
                               int(math.floor(border_size[0] / 2.0)),
                               int(math.ceil(border_size[0] / 2.0)), 0, 0,
                               cv2.BORDER_CONSTANT, value=border_color)
    elif border_size[0] < 0:
        I = I[-int(math.floor(border_size[0] / 2.0)): I.shape[0]
                                                      + int(math.ceil(border_size[0] / 2.0)), :, :]
    if border_size[1] > 0:
        I = cv2.copyMakeBorder(I, 0, 0,
                               int(math.floor(border_size[1] / 2.0)),
                               int(math.ceil(border_size[1] / 2.0)),
                               cv2.BORDER_CONSTANT, value=border_color)
    elif border_size[1] < 0:
        I = I[:, -int(math.floor(border_size[1] / 2.0)): I.shape[1]
                                                         + int(math.ceil(border_size[1] / 2.0)), :]

    return I