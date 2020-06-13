import functools
import json
import os
import re
import sys
import time

import cv2
import tensorflow.keras.metrics as keras_metrics
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np

from src import mri_loader, utils
from src import networks, experiment_base

import ext.neuron.neuron.layers as nrn_layers


class SegmenterTrainer(experiment_base.Experiment):
    def get_model_name(self):
        exp_name = 'FewShotSeg'

        exp_name += '_{}'.format(self.dataset_name)

        if 'warpoh' in self.arch_params.keys() and self.arch_params['warpoh']:
            exp_name += '_warpoh'

        # random flow augmentation params
        if self.aug_rand:
            if 'flow_sigma' in self.data_params['aug_params'] \
                    and self.data_params['aug_params']['flow_sigma'] is not None:
                exp_name += '_augflow-sigma{}_blur{}'.format(
                    self.data_params['aug_params']['flow_sigma'],
                    self.data_params['aug_params']['blur_sigma'])
            elif 'flow_amp' in self.data_params['aug_params'] \
                    and self.data_params['aug_params']['flow_amp'] is not None:
                exp_name += '_augflow-amp{}_blur{}'.format(
                    self.data_params['aug_params']['flow_amp'],
                    self.data_params['aug_params']['blur_sigma'])
            if 'offset_amp' in self.data_params['aug_params']:
                exp_name += '_randoffset{}'.format(self.data_params['aug_params']['offset_amp'])
            if 'mult_amp' in self.data_params['aug_params']:
                exp_name += '_randmult{}'.format(self.data_params['aug_params']['mult_amp'])

        if self.aug_rand:
            exp_name += '_rand-aug'
            if self.data_params['n_flow_aug'] is not None and not self.data_params['aug_in_gen']:
                exp_name += '{}'.format(self.data_params['n_flow_aug'])
            else:
                exp_name += '-gen'

        if self.aug_tm:
            exp_name += '_tm-aug'
            if self.data_params['n_tm_aug'] is not None and not self.data_params['aug_in_gen']:
                exp_name += '{}'.format(self.data_params['n_tm_aug'])
            else:
                exp_name += '-gen'

            exp_name += '-{}'.format(self.aug_model_name)

            if self.arch_params['do_coupled_sampling']:
                exp_name += '_coupled'


        if self.aug_sas:
            exp_name += '_sasaug'
            if self.data_params['n_sas_aug'] is not None and not self.data_params['aug_in_gen']:
                exp_name += '{}'.format(self.data_params['n_sas_aug'])
            else:
                exp_name += '-gen'
            exp_name += '-{}'.format(self.aug_model_name)


        self.model_name = exp_name
        exp_name = super(SegmenterTrainer, self).get_model_name()
        self.model_name = exp_name
        return exp_name


    def __init__(self, data_params, arch_params, debug=False, prompt_delete=True):
        self.logger = None

        self.arch_params = arch_params
        self.data_params = data_params

        # i.e. are we segmenting/augmenting slices or volumes
        self.n_seg_dims = arch_params['n_seg_dims']
        self.n_aug_dims = arch_params['n_aug_dims']

        self.pred_img_shape = data_params['img_shape'][:self.n_seg_dims] + (1,)
        self.aug_img_shape = data_params['img_shape'][:self.n_aug_dims] + (1,)

        self.display_slice_idx = 112

        self.logger = None
        self.profiler_logger = None
        self.do_profile = True
        self.profiled_iters = 0

        self.epoch_count = 0
        self.batch_count = 0
        if 'pretrain_l2' in self.arch_params.keys():
            self.loss_fn = keras_metrics.mean_squared_error
            self.loss_name = 'l2'
        else:
            self.loss_fn = keras_metrics.categorical_crossentropy
            self.loss_name = 'CE'

        # warp the onehot representation of labels instead of labels themselves
        if 'warpoh' not in self.arch_params.keys():
            self.arch_params['warpoh'] = False

        self.n_aug = None  # do augmentation through the generator by default

        self.aug_tm = False # flow and appearance transform models
        self.aug_sas = False # flow model for single-atlas segmentation
        self.aug_rand = False # random flow and multiplicative intensity

        if 'aug_rand' in data_params.keys() and data_params['aug_rand']:
            self.aug_rand = True
            if 'n_flow_aug' in data_params.keys() and data_params['n_flow_aug'] is not None:
                self.n_aug = data_params['n_flow_aug']
        elif 'aug_randmult' in data_params.keys() and data_params['aug_randmult']:
            self.aug_randmult = True

        if data_params['aug_tm']:
            self.aug_tm = True
            self.aug_sas = False
            if 'n_tm_aug' in data_params.keys() and data_params['n_tm_aug'] is not None:
                self.n_aug = data_params['n_tm_aug']
        elif data_params['aug_sas']:
            self.aug_sas = True
            self.aug_tm = False
            if 'n_sas_aug' in data_params.keys() and data_params['n_sas_aug'] is not None:
                self.n_aug = data_params['n_sas_aug']

        # come up with a short name for our flow and color models so we can put them in this model name
        if self.arch_params['tm_flow_model'] is not None and self.arch_params['tm_color_model'] is not None \
                and self.aug_tm:
            if 'epoch' in self.arch_params['tm_flow_model']:
                flow_epoch = re.search('(?<=_epoch)[0-9]*', self.arch_params['tm_flow_model']).group(0)
            else:
                flow_epoch = int(int(re.search('(?<=_iter)[0-9]*', self.arch_params['tm_flow_model']).group(0)) / 100)
            color_epoch = re.search('(?<=_epoch)[0-9]*', self.arch_params['tm_color_model']).group(0)

            # only include the color model in the name if we are doing both flow and color aug
            self.aug_model_name = 'tmflow-e{}-colore{}'.format(
                flow_epoch, color_epoch)
        elif self.arch_params['tm_flow_model'] is not None:
            self.aug_model_name = 'tmflow-{}'.format(
                os.path.basename(self.arch_params['tm_flow_model'].split('/models/')[0]))

        # do augmentation through generator, or pre-augment training set
        if 'aug_in_gen' not in data_params.keys():
            self.data_params['aug_in_gen'] = False
        if self.data_params['aug_in_gen']:
            self.n_aug = None

        # let dataset loader figure out short name
        self.data_params['n_dims'] = self.n_aug_dims
        self.dataset = mri_loader.MRIDataset(self.data_params, self.logger)
        self.dataset_name = self.dataset.create_display_name()


        # automatic early stopping based on validation loss
        if 'patience' in arch_params.keys():
            validation_losses_buff_len = arch_params['patience']
        else:
            validation_losses_buff_len = 10
        super(SegmenterTrainer, self).__init__(data_params, arch_params,
                                               prompt_delete_existing=prompt_delete)

        self.validation_losses_buffer = [np.nan] * validation_losses_buff_len

        # keep track of all ids the network sees as a sanity check
        self.all_ul_ids = []
        self.all_train_ids = []


    def load_data(self, load_n=None):
        self.dataset.logger = self.logger
        self.dataset.profiler_logger = self.profiler_logger

        (self.X_unlabeled, _, _, self.ids_unlabeled),\
        (self.X_labeled_train, self.segs_labeled_train, self.contours_labeled_train, self.ids_labeled_train), \
        (self.X_labeled_valid, self.segs_labeled_valid, _, self.ids_labeled_valid), \
        self.label_mapping \
        = self.dataset.load_dataset(load_n=load_n)

        # make sure we have a channels dimension
        if len(self.segs_labeled_train.shape) < 5:
            self.segs_labeled_train = self.segs_labeled_train[..., np.newaxis]
            self.segs_labeled_valid = self.segs_labeled_valid[..., np.newaxis]

        # assert that none of the validation examples are also in the trainig set
        assert not np.any([id_lv in self.ids_labeled_train for id_lv in self.ids_labeled_valid])

        self.n_labels = len(self.label_mapping)

        self.aug_img_shape = self.X_unlabeled.shape[1:]
        self.pred_segs_oh_shape = self.pred_img_shape[:-1] + (self.n_labels, )
        self.pred_segs_shape = self.pred_img_shape[:-1] + (1,)

        if self.logger is not None:
            self.logger.debug('Unlabeled images {}'.format(self.X_unlabeled.shape))
            self.logger.debug('Labeled training images {} and labels {}'.format(self.X_labeled_train.shape, self.segs_labeled_train.shape))
            self.logger.debug('Labeled validation images {} and labels {}'.format(self.X_labeled_valid.shape, self.segs_labeled_valid.shape))
            self.logger.debug('Number of labels: {}, e.g. {}...'.format(len(self.label_mapping), self.label_mapping[:10]))

        # save files to logs
        with open(os.path.join(self.exp_dir, 'ids_unlabeled.txt'), 'w') as f:
            f.writelines(['{}\n'.format(f) for f in self.ids_unlabeled])
        with open(os.path.join(self.exp_dir, 'ids_labeled_train.txt'), 'w') as f:
            f.writelines(['{}\n'.format(f) for f in self.ids_labeled_train])
        with open(os.path.join(self.exp_dir, 'ids_labeled_validation.txt'), 'w') as f:
            f.writelines(['{}\n'.format(f) for f in self.ids_labeled_valid])

        # save dataset and architecture parameters to files
        with open(os.path.join(self.exp_dir, 'arch_params.json'), 'w') as f:
            json.dump(self.arch_params, f)
        with open(os.path.join(self.exp_dir, 'data_params.json'), 'w') as f:
            json.dump(self.data_params, f)


    def create_generators(self, batch_size):
        self.batch_size = batch_size

        # generator for labeled training examples that we wish to augment
        self.source_gen = self.dataset.gen_vols_batch(
            dataset_splits=['labeled_train'], batch_size=1,
            load_segs=True, # always load segmentations since this is our labeled set
            load_contours=self.aug_tm, # only load contours if we need them as aux data for our appearance model
            randomize=True,
            return_ids=True,
        )

        # actually more like a target generator
        self.unlabeled_gen_raw = self.dataset.gen_vols_batch(
            dataset_splits=['labeled_train', 'unlabeled_train'], batch_size=1,
            load_segs=False, load_contours=False,
            randomize=True,
            return_ids=True,
        )

        self._create_augmentation_models()

        # simply append augmented examples to training set
        # NOTE: we can only do this if we are not synthesizing that many examples (i.e. <= 1000)
        self.aug_train_gen = None
        if self.n_aug is not None:
            self._create_augmented_examples()
            self.logger.debug('Augmented classifier training set: vols {}, segs {}'.format(
                self.X_labeled_train.shape, self.segs_labeled_train.shape))
        elif self.aug_tm or self.aug_rand:
            # augmentation by transform model or random flow+intensity requires synthesizing
            # a lot of examples, so we'll just do this per-batch
            aug_by = []
            if self.aug_tm:
                aug_by += ['tm']
            if self.aug_rand:
                aug_by += ['rand']

            # these need to be done in the generator
            self.aug_train_gen = self._generate_augmented_batch(
                    source_gen=self.source_gen, use_single_atlas=self.X_labeled_train.shape[0] == 1,
                    aug_by=aug_by,
                    convert_onehot=True, return_transforms=True,
                    do_slice_z=(self.n_seg_dims == 2)  # if we are predicting on slices, then get slices
            )

        # generates slices from the training volumes
        self.train_gen = self._generate_slices_batchs_from_vols(
            self.X_labeled_train, self.segs_labeled_train, self.ids_labeled_train,
            vol_gen=None,
            convert_onehot=True,
            batch_size=self.batch_size, randomize=True
        )

        # load each subject, then evaluate each slice
        self.eval_valid_gen = self.dataset.gen_vols_batch(
            ['labeled_valid'],
            batch_size=1, randomize=False,
            convert_onehot=False,
            load_segs=True,
            label_mapping=self.label_mapping,
            return_ids=True,
        )

        # we will compute validation losses on all validation volumes
        # just pick some random slices to display for the validation set later
        rand_subjs = np.random.choice(self.X_labeled_valid.shape[0], batch_size)
        rand_slices = np.random.choice(self.aug_img_shape[2], batch_size, replace=False)

        self.X_valid_batch = np.zeros((batch_size,) + self.pred_img_shape)
        self.Y_valid_batch = np.zeros((batch_size,) + self.pred_segs_shape)
        for ei in range(batch_size):
            self.X_valid_batch[ei] = self.X_labeled_valid[rand_subjs[ei], :, :, rand_slices[ei]]
            self.Y_valid_batch[ei] = self.segs_labeled_valid[rand_subjs[ei], :, :, rand_slices[ei]]

        self.Y_valid_batch = utils.labels_to_onehot(self.Y_valid_batch, label_mapping=self.label_mapping)


    def _generate_slices_batchs_from_vols(self, vols, segs, ids, vol_gen=None,
                                          randomize=False, batch_size=16,
                                          vol_batch_size=1,
                                          convert_onehot=False,
                                          ):
        if vol_gen is None:
            vol_gen = utils.gen_batch(vols, [segs, ids],
                                            randomize=randomize,
                                            batch_size=vol_batch_size)

        while True:
            X, Y, ids_batch = next(vol_gen)

            n_slices = X.shape[-2]
            slice_idxs = np.random.choice(n_slices, self.batch_size, replace=True)

            X_slices = np.reshape(
                np.transpose(X[:, :, :, slice_idxs],
                             (0, 3, 1, 2, 4)), (-1,) + self.pred_img_shape)

            if Y is not None and self.arch_params['warpoh']:
                # we used the onehot warper in these cases
                Y = np.reshape(np.transpose(Y[:, :, :, slice_idxs], (0, 3, 1, 2, 4)),
                               (-1,) + self.pred_img_shape[:-1] + (self.n_labels,))
            else:
                Y = np.reshape(np.transpose(Y[:, :, :, slice_idxs], (0, 3, 1, 2, 4)),
                               (-1,) + self.pred_img_shape[:-1] + (1,))

            if Y is not None and convert_onehot:
                Y = utils.labels_to_onehot(Y, label_mapping=self.label_mapping)

            yield X_slices, Y, ids_batch



    def _create_augmentation_models(self, indexing='xy'):
        # TODO: put this in a param somewhere

        self.vol_warp_model = networks.warp_model(
            img_shape=self.aug_img_shape,
            interp_mode='linear',
            indexing=indexing,
        )

        # segmentation warpers that take in a flow field
        if 'warpoh' in self.arch_params.keys() and self.arch_params['warpoh']:
            self.seg_warp_model = networks.warp_model(
                img_shape=self.aug_img_shape[:-1] + (self.n_labels,),  # (1,),
                interp_mode='linear',
                indexing=indexing,
            )
        else:
            self.seg_warp_model = networks.warp_model(
                img_shape=self.aug_img_shape[:-1] + (1,),
                interp_mode='nearest',
                indexing=indexing
            )

        if self.aug_rand:
            if self.data_params['aug_params']['randflow_type'] == 'ronneberger':
                self.flow_rand_aug_model = networks.randflow_ronneberger_model(
                    img_shape=self.aug_img_shape,
                    model=None,
                    interp_mode='linear',
                    model_name='randflow_ronneberger_model',
                    flow_sigma=self.data_params['aug_params']['flow_sigma'],
                    blur_sigma=self.data_params['aug_params']['blur_sigma']
                )
                self.logger.debug('Random flow Ronneberger augmentation model')
            else:
                self.flow_rand_aug_model = networks.randflow_model(
                    img_shape=self.aug_img_shape,
                    model=None,
                    interp_mode='linear',
                    model_name='randflow_model',
                    flow_sigma=self.data_params['aug_params']['flow_sigma'],
                    flow_amp=self.data_params['aug_params']['flow_amp'],
                    blur_sigma=self.data_params['aug_params']['blur_sigma'],
                    indexing=indexing,
                )
                self.logger.debug('Random flow augmentation model')

            self.flow_rand_aug_model.summary(print_fn=self.logger.debug)

        if self.aug_tm or self.aug_sas:
            self.flow_aug_model = load_model(
                self.arch_params['tm_flow_model'],
                custom_objects={
                    'SpatialTransformer': functools.partial(
                        nrn_layers.SpatialTransformer,
                        indexing=indexing)
                }, compile=False)

            if self.arch_params['tm_color_model'] is not None and self.aug_tm:
                self.flow_bck_aug_model = load_model(
                    self.arch_params['tm_flow_bck_model'],
                    custom_objects={
                        'SpatialTransformer': functools.partial(
                            nrn_layers.SpatialTransformer,
                            indexing=indexing)
                    }, compile=False)
                self.color_aug_model = load_model(
                    self.arch_params['tm_color_model'],
                    custom_objects={
                        'SpatialTransformer': functools.partial(
                            nrn_layers.SpatialTransformer,
                            indexing=indexing)
                    }, compile=False)

                if 'l2-tgt' in self.arch_params['tm_color_model']:
                    # if the color model transforms to the target space by default, create
                    # a wrapper that gets the source space
                    self.color_aug_model = Model(
                        inputs=self.color_aug_model.inputs[:-1],
                        outputs=[
                            self.color_aug_model.outputs[0],
                            self.color_aug_model.get_layer('add_color_delta').output,
                            self.color_aug_model.outputs[2]],
                        name='color_model_wrapper')



    def _create_augmented_examples(self):
        preview_augmented_examples = True

        if self.aug_sas:
            aug_name = 'SAS'
            # just label a bunch of examples using our SAS model, and then append them to the training set
            source_X = self.X_labeled_train
            source_Y = self.segs_labeled_train

            unlabeled_labeler_gen = self.dataset.gen_vols_batch(
                dataset_splits=['unlabeled_train'],
                batch_size=1, randomize=False, return_ids=True)

            X_train_aug = np.zeros((self.n_aug,) + self.aug_img_shape)
            Y_train_aug = np.zeros((self.n_aug,) + self.aug_img_shape[:-1] + (1,))
            ids_train_aug = []#['sas_aug_{}'.format(i) for i in range(self.n_aug)]
            for i in range(self.n_aug):
                self.logger.debug('Pseudo-labeling UL example {} of {} using SAS!'.format(i, self.n_aug))
                unlabeled_X, _, _, ul_ids = next(unlabeled_labeler_gen)

                # warp labeled example to unlabeled example
                X_aug, flow = self.flow_aug_model.predict([source_X, unlabeled_X])
                # warp labeled segs similarly
                Y_aug = self.seg_warp_model.predict([source_Y, flow])

                X_train_aug[i] = unlabeled_X
                Y_train_aug[i] = Y_aug
                ids_train_aug += ['sas_{}'.format(ul_id) for ul_id in ul_ids]

        elif self.aug_tm and self.data_params['n_tm_aug'] is not None \
                and self.data_params['n_tm_aug'] <= 100:

            # TODO: this section is not well tested since it is no longer really used, we do even coupled
            # augmentation in the generator
            aug_name = 'tm'
            source_train_gen_tmaug = self._generate_augmented_batch(
                self.source_gen,
                aug_by='tm',
                use_single_atlas=self.X_labeled_train.shape[0] == 1
            )

            # augment and append to training set
            n_aug_batches = int(np.ceil(self.data_params['n_tm_aug'] / float(self.batch_size)))

            X_preaug = [None] * n_aug_batches
            Y_preaug = [None] * n_aug_batches
            X_train_aug = [None] * n_aug_batches
            Y_train_aug = [None] * n_aug_batches
            ids_train_aug = []

            for i in range(n_aug_batches):
                # get source examples to perform augmentation on
                X_preaug[i], Y_preaug[i], X_train_aug[i], Y_train_aug[i], aug_ids_batch = next(source_train_gen_tmaug)
                ids_train_aug += aug_ids_batch

            self.X_tm_aug = np.concatenate(X_train_aug, axis=0)[:self.data_params['n_tm_aug']]
            self.Y_tm_aug = np.concatenate(Y_train_aug, axis=0)[:self.data_params['n_tm_aug']]

        self.X_labeled_train = np.concatenate([self.X_labeled_train, X_train_aug], axis=0)
        self.segs_labeled_train = np.concatenate([self.segs_labeled_train, Y_train_aug], axis=0)
        self.ids_labeled_train += ids_train_aug
        self.logger.debug('Added {} {}-augmented batches to training set!'.format(len(X_train_aug), aug_name))

        if preview_augmented_examples:
            print_batch_size = 10
            show_slice_idx = 112
            n_aug_batches = int(np.ceil(X_train_aug.shape[0] / float(print_batch_size)))
            aug_out_im = []
            for bi in range(min(20, n_aug_batches)):
                aug_im = utils.concatenate_with_pad([
                    utils.label_ims(
                        X_train_aug[bi * print_batch_size:min(X_train_aug.shape[0], (bi + 1) * print_batch_size), ..., show_slice_idx, :], []),
                    utils.label_ims(
                        Y_train_aug[bi * print_batch_size:min(X_train_aug.shape[0], (bi + 1) * print_batch_size), ..., show_slice_idx, :], []),
                ], axis=1)
                aug_out_im.append(aug_im)
            aug_out_im = np.concatenate(aug_out_im, axis=0)
            cv2.imwrite(os.path.join(self.exp_dir, 'aug_{}_examples.jpg'.format(aug_name)), aug_out_im)


    def _generate_augmented_batch(
            self,
            source_gen,
            use_single_atlas=False,
            aug_by=None,
            return_transforms=False,
            convert_onehot=False,
            do_slice_z=False,
    ):

        if use_single_atlas: # single atlas
            # single atlas, dont bother with generator
            self.logger.debug('Single atlas, not using source generator for augmenter')
            source_X, source_segs, source_contours, source_ids = next(source_gen)
        else:
            self.logger.debug('Multiple atlases, sampling source vols from generator')

        while True:
            if not use_single_atlas:
                # randomly select an atlas from our source volume generator
                source_X, source_segs, source_contours, source_ids = next(source_gen)

            # keep track of which unlabeled subjects we are using in training
            ul_ids = []

            if len(aug_by) > 1:
                # multiple augmentation schemes. let's flip a coin
                do_aug_by_idx = np.random.rand(1)[0] * float(len(aug_by) - 1)
                aug_batch_by = aug_by[int(round(do_aug_by_idx))]
            else:
                aug_batch_by = aug_by[0]

            color_delta = None
            start = time.time()
            if aug_batch_by == 'rand':
                X_aug, flow = self.flow_rand_aug_model.predict(source_X)

                # color augmentation by additive or multiplicative factor. randomize the factor and then tile to correct shape
                if 'offset_amp' in self.data_params['aug_params']:
                    X_aug += np.tile(
                        (np.random.rand(X_aug.shape[0], 1, 1, 1) * 2. - 1.) * self.data_params['aug_params'][
                            'offset_amp'],
                        (1,) + X_aug.shape[1:])
                    X_aug = np.clip(X_aug, 0., 1.)
                if 'mult_amp' in self.data_params['aug_params']:
                    X_aug *= np.tile(
                        1. + (np.random.rand(X_aug.shape[0], 1, 1, 1) * 2. - 1.) * self.data_params['aug_params'][
                            'mult_amp'],
                        (1,) + X_aug.shape[1:])
                    X_aug = np.clip(X_aug, 0., 1.)

                # Y_to_aug = classification_utils.labels_to_onehot(Y_to_aug, label_mapping=self.label_mapping)
                Y_aug = self.seg_warp_model.predict([source_segs, flow])
            elif aug_batch_by == 'tm':
                if self.arch_params['do_coupled_sampling']:
                    # use the same target for flow and color
                    X_flowtgt, _, _, ul_flow_ids = next(self.unlabeled_gen_raw)
                    X_colortgt = X_flowtgt
                    ul_ids += ul_flow_ids
                else:
                    X_flowtgt, _, _, ul_flow_ids = next(self.unlabeled_gen_raw)
                    X_colortgt, _, _, ul_color_ids = next(self.unlabeled_gen_raw)
                    ul_ids += ul_flow_ids + ul_color_ids

                    if self.do_profile:
                        self.profiler_logger.info('Sampling aug tgt took {}'.format(time.time() - st))

                self.aug_target = X_flowtgt

                # compute forward flow, which we will use for the spatial transformation
                _, flow = self.flow_aug_model.predict([source_X, X_flowtgt])

                # warp color target back to the atlas space so that we can compute the color transformation
                X_colortgt_src, _ = self.flow_bck_aug_model.predict([X_colortgt, source_X])
                colored_vol, color_delta, _ = self.color_aug_model.predict([source_X, X_colortgt_src, source_contours, flow])

                self.aug_colored = colored_vol

                # color the source volume, and then apply the spatial transformation
                X_aug = self.vol_warp_model.predict([colored_vol, flow])

                if self.do_profile:
                    self.profiler_logger.info('Running color and flow aug took {}'.format(time.time() - st))

                st = time.time()
                # now warp the labels to match
                Y_aug = self.seg_warp_model.predict([source_segs, flow])
                if self.do_profile:
                    self.profiler_logger.info('Warping labels took {}'.format(time.time() - st))
            else:
                # no aug
                X_aug = source_X
                Y_aug = source_segs

            if self.do_profile:
                self.profiler_logger.info('Augmenting input batch took {}'.format(time.time() - start))

            if do_slice_z:
                # get a random slice in the z dimension
                start = time.time()
                n_total_slices = source_X.shape[-2]

                # always take random slices in the z dimension
                slice_idxs = np.random.choice(n_total_slices, self.batch_size, replace=True)

                X_to_aug_slices = np.reshape(
                    np.transpose(  # roll z-slices into batch
                        source_X[:, :, :, slice_idxs],
                        (0, 3, 1, 2, 4)),
                    (-1,) + tuple(self.pred_img_shape))

                X_aug = np.reshape(
                    np.transpose(
                        X_aug[:, :, :, slice_idxs],
                        (0, 3, 1, 2, 4)),
                    (-1,) + self.pred_img_shape)

                if self.aug_target is not None:
                    self.aug_target = np.reshape(
                        np.transpose(self.aug_target[..., slice_idxs, :], (0, 3, 1, 2, 4)), (-1,) + self.pred_img_shape)

                    if self.aug_colored is not None:
                        # not relevant if we arent using a color transform model
                        self.aug_colored = np.reshape(
                            np.transpose(self.aug_colored[..., slice_idxs, :], (0, 3, 1, 2, 4)), (-1,) + self.pred_img_shape)

                if (aug_by == 'rand' or aug_by == 'tm') and self.arch_params['warpoh']:
                    # we used the onehot warper in these cases
                    Y_to_aug_slices = np.reshape(
                        np.transpose(
                            source_segs[:, :, :, slice_idxs],
                            (0, 3, 1, 2, 4)),
                        (-1,) + self.pred_segs_oh_shape)

                    Y_aug = np.reshape(np.transpose(Y_aug[:, :, :, slice_idxs], (0, 3, 1, 2, 4)),
                                       (-1,) + self.pred_segs_oh_shape)
                else:
                    Y_to_aug_slices = np.reshape(np.transpose(
                        source_segs[:, :, :, slice_idxs], (0, 3, 1, 2, 4)),
                     (-1,) + self.pred_segs_shape)
                    Y_aug = np.reshape(np.transpose(
                        Y_aug[:, :, :, slice_idxs], (0, 3, 1, 2, 4)),
                        (-1,) + self.pred_segs_shape)

                if self.do_profile:
                    self.profiler_logger.info('Slicing batch took {}'.format(time.time() - start))

                if return_transforms:
                    st = time.time()
                    flow = np.reshape(
                        np.transpose(
                            flow[:, :, :, slice_idxs, :],
                            (0, 3, 1, 2, 4)),
                        (-1,) + self.pred_img_shape[:-1] + (self.n_aug_dims,))
                    if self.do_profile:
                        self.profiler_logger.info('Slicing flow took {}'.format(time.time() - st))

                if color_delta is not None:
                    color_delta = np.reshape(
                        np.transpose(
                            color_delta[:, :, :, slice_idxs, :],
                            (0, 3, 1, 2, 4)),
                        (-1,) + self.pred_img_shape)
            else:
                # no slicing, no aug?
                X_to_aug_slices = source_X
                Y_to_aug_slices = source_segs

            if convert_onehot and not ((aug_by == 'rand' or aug_by == 'tm') and self.arch_params['warpoh']):
                # if we don't have onehot segs already, convert them after slicing
                start = time.time()
                Y_to_aug_slices = utils.labels_to_onehot(
                    Y_to_aug_slices, label_mapping=self.label_mapping)
                Y_aug = utils.labels_to_onehot(
                    Y_aug, label_mapping=self.label_mapping)
                if self.do_profile:
                    self.profiler_logger.info('Converting onehot took {}'.format(time.time() - start))

            if return_transforms:
                yield X_to_aug_slices, Y_to_aug_slices, flow, color_delta, X_aug, Y_aug, ul_ids
            else:
                yield X_to_aug_slices, Y_to_aug_slices, X_aug, Y_aug, ul_ids

    def save_exp_info(self, exp_dir, figures_dir, models_dir, logs_dir):
        super(SegmenterTrainer, self).save_exp_info(
            exp_dir, figures_dir, models_dir, logs_dir)
        self.dataset.logger = self.logger


    def save_models(self, epoch, iter_count=None):
        super(SegmenterTrainer, self).save_models(epoch, iter_count=iter_count)


    def create_models(self):
        print('Creating segmenter model on pred shape {}'.format(self.pred_img_shape))
        self.base_segmenter_model = networks.segmenter_unet(
            img_shape=self.pred_img_shape,
            n_labels=self.n_labels,
            model_name='segmenter_unet',
            params=self.arch_params,
        )

        self.logger.debug('Base segmenter model')
        self.base_segmenter_model.summary(print_fn=self.logger.debug)
        if 'pretrain_l2' in self.arch_params.keys():
            # pretrain the network using L2 on the layer before the softmax. this helps with convergence
            self.segmenter_model = Model(
                inputs=self.base_segmenter_model.inputs,
                outputs=self.base_segmenter_model.get_layer('unet_dec_conv2D_final').output,
                name='segmenter_pre_softmax')

        self.models = [self.segmenter_model]

        super(SegmenterTrainer, self).create_models()


    def load_models(self, load_epoch=None, stop_on_missing=True, init_layers=False):
        if load_epoch == 'latest':
            load_epoch = utils.get_latest_epoch_in_dir(self.models_dir)
            self.logger.debug('Found latest epoch {} in dir {}'.format(load_epoch, self.models_dir))

        # do this first so we look for the correct (not pre-softmax) model
        if load_epoch is not None:
            self.update_epoch_count(int(load_epoch))

        start_epoch = super(SegmenterTrainer, self).load_models(load_epoch)

        return start_epoch


    def compile_models(self, run_options=None, run_metadata=None):
        if not run_options is None and not run_metadata is None:
            self.segmenter_model.compile(
                loss=self.loss_fn,
                optimizer=Adam(lr=self.arch_params['lr'], amsgrad=True),
                options=run_options, run_metadata=run_metadata
            )
        else:
            self.segmenter_model.compile(
                loss=self.loss_fn,
                optimizer=Adam(lr=self.arch_params['lr'], amsgrad=True),
            )

        self.loss_names = [self.loss_name]
        super(SegmenterTrainer, self).compile_models()


    def make_train_results_im(self):
        Y_pred = self.segmenter_model.predict(self.X_train_batch)

        # keep track of which ids we are training on
        with open(os.path.join(self.exp_dir, 'train_ids.txt'), 'w') as f:
            f.writelines([i + '\n' for i in self.all_train_ids])
        with open(os.path.join(self.exp_dir, 'ul_ids.txt'), 'w') as f:
            f.writelines([i + '\n' for i in self.all_train_ids])

        if self.aug_target is None:
            # no augmentation, just show direct predictions
            return self._make_results_im(
                    [self.X_train_batch, self.Y_train_batch, Y_pred],
                    ['input_im', 'gt_seg', 'pred_seg'],
                    is_seg=[False, True, True],
                    overlay_on_ims=[None, self.X_train_batch, self.X_train_batch],
                )
        else:
            # no augmentation, just show direct predictions
            return self._make_results_im(
                    [self.aug_target, self.aug_colored, self.X_train_batch, self.Y_train_batch, Y_pred],
                    ['aug_tgt', 'colored', 'input_im', 'gt_seg', 'pred_seg'],
                    is_seg=[False, False, False, True, True],
                    overlay_on_ims=[None, None, None, self.X_train_batch, self.X_train_batch],
                )


    def make_test_results_im(self, epoch_num=None):
        Y_pred = self.segmenter_model.predict(self.X_valid_batch)

        return self._make_results_im(
            [self.X_valid_batch, self.Y_valid_batch, Y_pred],
            ['input_im', 'gt_seg', 'pred_seg'],
            is_seg=[False, True, True],
            overlay_on_ims=[None, self.X_valid_batch, self.X_valid_batch],
        )


    def _make_results_im(self, input_im_batches, labels,
                        overlay_on_ims=None,
                        do_normalize=None, is_seg=None,
                         max_batch_size=32):
        # batch_size = inputs_im.shape[0]
        batch_size = self.batch_size
        display_batch_size = min(max_batch_size, batch_size)
        zeros_batch = np.zeros((batch_size,) + self.pred_img_shape)

        if do_normalize is None:
            do_normalize = [False] * len(input_im_batches)
        if is_seg is None:
            is_seg = [False] * len(input_im_batches)

        if display_batch_size < batch_size:
            input_im_batches = [batch[:display_batch_size] for batch in input_im_batches]
            overlay_on_ims = [im[:display_batch_size] if im is not None else None for im in overlay_on_ims]

        show_label_idx = 12 # cerebral wm
        out_im = np.concatenate([
            utils.label_ims(batch, labels[i]) if not is_seg[i] else
            np.concatenate([  # we want two images here: overlay and a single label
                utils.label_ims(np.transpose(
                        utils.overlay_segs_on_ims_batch(
                            ims=np.transpose(overlay_on_ims[i], (1, 2, 3, 0)),
                            segs=np.transpose(
                                utils.onehot_to_labels(
                                    batch, label_mapping=self.label_mapping), (1, 2, 0)),
                            include_labels=self.label_mapping,
                            draw_contours=True,
                        ),
                        (3, 0, 1, 2)), []),
                utils.label_ims(batch[..., [show_label_idx]],
                                    'label {}'.format(self.label_mapping[show_label_idx]), normalize=True)], axis=1) \
            for i, batch in enumerate(input_im_batches) if batch is not None
        ], axis=1)

        return out_im


    def get_n_train(self):
        # number of volumes times number of z-slices
        return min(100, len(self.dataset.files_labeled_train) * self.X_labeled_train.shape[-2])

    def get_n_test(self):
        # number of volumes only, since we will evaluate all slices
        return self.X_labeled_valid.shape[0]


    def train_on_batch(self):
        start = time.time()
        #self.iter_count += 1
        self.aug_target = None
        self.aug_colored = None

        if self.aug_train_gen is None:
            # just get examples from the generator with pre-augmented training set
            self.X_train_preaug = None
            self.Y_train_preaug = None


            self.X_train_batch, self.Y_train_batch, self.train_ids = next(self.train_gen)
            self.all_train_ids = list(set(self.all_train_ids + self.train_ids))
        else:
            self.X_train_preaug, self.Y_train_preaug, self.train_augflow, self.train_augcolor, \
            self.X_train_batch, self.Y_train_batch, self.ul_train_ids = next(self.aug_train_gen)
            self.all_ul_ids = list(set(self.all_ul_ids + self.ul_train_ids))

        if self.do_profile:
            self.profiler_logger.info('Generating training batch took {}'.format(time.time() - start))

        # if we are pre-training without softmax (using L2 on log of prediction)
        if self.loss_name == 'l2':
            self.Y_train_batch = np.clip(np.log(self.Y_train_batch), -100., 0.)

        start = time.time()

        segmenter_loss = self.segmenter_model.train_on_batch(
            self.X_train_batch, self.Y_train_batch)
        if not isinstance(segmenter_loss, list):
            segmenter_loss = [segmenter_loss]

        if self.do_profile:
            self.profiler_logger.info('train_on_batch took {}'.format(time.time() - start))
        segmenter_loss_names = ['train_' + ln for ln in self.loss_names]

        self.batch_count += 1
        self.test_batch_count = 0

        # only profile for 100 iterations -- probably accurate enough
        if self.do_profile:
            self.profiled_iters += 1

            if self.profiled_iters >= 100:
                self.do_profile = False
                self.profiler_logger.handlers[0].close()
                self.dataset.profiler_logger = None

        assert len(segmenter_loss) == len(segmenter_loss_names)
        return segmenter_loss, segmenter_loss_names


    def test_batches(self):
        valid_losses = []

        # just validate on a smaller set during training so it doesnt take too long
        n_test_examples = len(self.dataset.files_labeled_valid)
        cces, dice, accs = self._eval_from_gen(self.eval_valid_gen, n_test_examples)

        valid_losses += [np.mean(cces), np.mean(dice[:, 1:]), np.mean(accs)]
        valid_loss_names = ['valid_' + ln for ln in ['CCE', 'dice_nobkg', 'acc']]

        # shift buffer up
        self.validation_losses_buffer[:-1] = self.validation_losses_buffer[1:]

        # since our loss is -dice, include -loss in the buffer because experiment logic checks if it is going up
        self.validation_losses_buffer[-1] = -valid_losses[1]

        assert len(valid_losses) == len(valid_loss_names)
        return valid_losses, valid_loss_names

    def _eval_from_gen(self, eval_gen, n_test_examples):
        batch_size = min(self.batch_size, 10) # just make batches manually, this way we can make sure we test everything
        # test metrics: categorical cross-entropy and dice

        cces, dice_per_label, accs, all_eval_ids = utils.eval_seg_from_gen(
            self.segmenter_model, eval_gen=eval_gen, label_mapping=self.label_mapping,
            n_eval_examples=n_test_examples, batch_size=batch_size, logger=self.logger)
        return cces, dice_per_label, accs


    def update_epoch_count(self, epoch):
        self.epoch_count += 1
        if 'pretrain_l2' in self.arch_params.keys() and epoch >= self.arch_params['pretrain_l2'] \
                and self.loss_name == 'l2':  # important so that we're not constantly recompiling
            self.loss_fn = keras_metrics.categorical_crossentropy
            self.loss_name = 'CE'

            self.segmenter_model = self.base_segmenter_model
            self.models = [self.segmenter_model]

            self.compile_models()
