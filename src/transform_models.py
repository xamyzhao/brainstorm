import functools
import json
import os
import time

import numpy as np
import tensorflow.keras.metrics as keras_metrics
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from src import experiment_base, mri_loader, networks, utils
from src import metrics as my_metrics

# from ext.voxelmorph.src import networks as vm_networks
from ext.neuron.neuron import layers as nrn_layers


class TransformModelTrainer(experiment_base.Experiment):
    def get_model_name(self):
        exp_name = 'TransformModel'

        exp_name += '_{}'.format(self.dataset.display_name)
        exp_name += '_{}'.format(self.arch_params['model_arch'])

        if 'flow' in self.arch_params['model_arch']:
            # flow smoothness and reconstruction losses
            if self.transform_reg_name is not None:
                exp_name += '_{}-regfwt{}'.format(self.transform_reg_name,
                                                  self.transform_reg_wt)
            if self.recon_loss_name is not None:
                exp_name += '_{}'.format(self.recon_loss_name)
                if 'l2' in self.recon_loss_name:
                    exp_name += '-wt{}'.format(self.recon_loss_wt)
                elif 'cc' in self.recon_loss_name:
                    exp_name += '-win{}'.format(self.cc_win_size_Iw)
                    exp_name += '-wt{}'.format(self.cc_loss_weight)

        elif 'color' in self.arch_params['model_arch']:
            if self.arch_params['use_aux_reg'] is not None \
                    and self.arch_params['do_include_aux_input']:
                exp_name += '_auxinput-{}'.format(self.arch_params['use_aux_reg'])

            # color smoothness and reconstruction losses
            if self.transform_reg_name is not None:
                exp_name += '_{}_regcwt{}'.format(self.transform_reg_name,
                                                  self.transform_reg_wt)

            if self.recon_loss_name is not None:
                exp_name += '_{}-wt{}'.format(self.recon_loss_name, self.recon_loss_wt)

        self.model_name = exp_name

        exp_name = super(TransformModelTrainer, self).get_model_name()
        self.model_name = exp_name
        return exp_name


    def __init__(self, data_params, arch_params):
        self.data_params = data_params
        self.arch_params = arch_params

        # if we are profiling our model, only do it for a few iterations
        # since there is some overhead that will slow us down
        self.do_profile = True
        self.profiled_iters = 0

        self.epoch_count = 0

        self.img_shape = data_params['img_shape']
        self.n_chans = data_params['img_shape'][-1]
        self.n_dims = len(self.img_shape) - 1

        # name our source domain according to our dataset parameters
        self.logger = None

        # initialize our dataset
        self.dataset = mri_loader.MRIDataset(self.data_params, self.logger)

        if 'use_aux_reg' not in arch_params.keys():
            self.arch_params['use_aux_reg'] = None

        # enc/dec architecture
        # parse params for flow portion of network
        if 'flow' in self.arch_params['model_arch']:
            self.transform_reg_name = self.arch_params['transform_reg_flow']

            if 'grad_l2' in self.transform_reg_name:
                self.transform_reg_fn = my_metrics.gradient_loss_l2(n_dims=self.n_dims)
                self.transform_reg_wt = self.arch_params['transform_reg_lambda_flow']
            else:
                self.transform_reg_fn = None
                self.transform_reg_wt = 0.

            self.recon_loss_name = self.arch_params['recon_loss_Iw']
            if self.recon_loss_name is None:  # still have this output node, but don't weight it
                self.recon_loss_fn = keras_metrics.mean_squared_error
                self.recon_loss_wt = 0
            elif 'cc_vm' in self.recon_loss_name:
                self.cc_loss_weight = self.arch_params['cc_loss_weight']
                self.cc_win_size_Iw = self.arch_params['cc_win_size_Iw']
                self.recon_loss_fn = my_metrics.NCC().loss
                self.recon_loss_wt = self.cc_loss_weight

        # parse params for color portion of network
        if 'color' in self.arch_params['model_arch']:
            self.recon_loss_name = self.arch_params['recon_loss_I']
            self.transform_reg_name = self.arch_params['transform_reg_color']

            if 'seg-l2' in self.transform_reg_name:
                self.transform_reg_wt = self.arch_params['transform_reg_lambda_color']
                self.transform_reg_fn = utils.SpatialSegmentSmoothness(
                    n_dims=self.n_dims,
                    n_chans=self.n_chans,
                ).compute_loss
            else:
                self.transform_reg_fn = None
                self.transform_reg_wt = 0.

            if self.recon_loss_name is None:  # still have this output node, but don't weight it
                self.recon_loss_fn = keras_metrics.mean_squared_error
                self.recon_loss_wt = 0
            elif 'l2' in self.recon_loss_name:
                self.recon_loss_fn = keras_metrics.mean_squared_error

                # set a constant weight for reconstruction
                self.recon_loss_wt = self.arch_params['recon_loss_wt']

        if 'latest_epoch' in arch_params.keys():
            self.latest_epoch = arch_params['latest_epoch']
        else:
            self.latest_epoch = 0

        super(TransformModelTrainer, self).__init__(
            data_params=self.data_params, arch_params=self.arch_params,
            prompt_delete_existing=True, prompt_update_name=True)

    def get_dirs(self):
        return self.exp_dir, self.figures_dir, self.logs_dir, self.models_dir

    def compile_models(self, run_options=None, run_metadata=None):
        if 'color' in self.arch_params['model_arch']:  # if we have a color transform, we might need to update some losses
            # point all of these regularizations at the color model inputs -- we assume everything
            # has been back-warped to the source space
            if 'seg-l2' in self.transform_reg_name:
                self.transform_reg_fn = my_metrics.SpatialSegmentSmoothness(
                    n_dims=self.n_dims,
                    n_chans=self.n_chans,
                    warped_contours_layer_output=self.transform_model.get_layer('aux').output
                ).compute_loss


        if 'bidir' in self.arch_params['model_arch']:
            loss_fns = [self.recon_loss_fn, self.recon_loss_fn, self.transform_reg_fn, self.transform_reg_fn]
            if 'separate' in self.arch_params['model_arch']:
                # need to regularize each model separately
                loss_weights = [self.recon_loss_wt, self.recon_loss_wt, self.transform_reg_wt, self.transform_reg_wt]
            else:
                loss_weights = [self.recon_loss_wt, self.recon_loss_wt, self.transform_reg_wt, self.transform_reg_wt]
            self.loss_names = [self.recon_loss_name + '_fwd', self.recon_loss_name + '_bck', self.transform_reg_name, 'flow_bck_dummy']
        elif 'flow' in self.arch_params['model_arch'] and 'bidir' not in self.arch_params['model_arch']:
            # voxelmorph returns warped, flow
            loss_fns = [self.recon_loss_fn, self.transform_reg_fn]
            loss_weights = [self.recon_loss_wt, self.transform_reg_wt]
            self.loss_names = [self.recon_loss_name, self.transform_reg_name]
        else:
            # appearance transform model outputs: transformed, transform, auxiliary data

            # dummy loss at the end for aux inputs so that we can do regularization
            loss_fns = [self.recon_loss_fn, self.transform_reg_fn, keras_metrics.mean_squared_error]
            loss_weights = [self.recon_loss_wt, self.transform_reg_wt, 0.]
            self.loss_names = [self.recon_loss_name, self.transform_reg_name, 'dummy_aux']


        self.logger.debug('Transform model')
        self.transform_model.summary(print_fn=self.logger.debug, line_length=120)

        self.loss_names = ['total'] + self.loss_names
        self.logger.debug('Compiling transform model with {} losses: {}'.format(len(loss_fns), self.loss_names))
        for li, lf in enumerate(loss_fns):
            self.logger.debug('Model output {}: {}, loss fn: {}'.format(
                li,
                self.transform_model.outputs[li],
                lf))
        self.logger.debug('and {} weights {}'.format(len(loss_weights), loss_weights))

        if run_options is not None:
            self.transform_model.compile(loss=loss_fns, loss_weights=loss_weights,
                                   optimizer=Adam(lr=self.arch_params['lr']),
                                    options=run_options, run_metadata=run_metadata,
            )

        else:
            self.transform_model.compile(loss=loss_fns, loss_weights=loss_weights,
                                   optimizer=Adam(lr=self.arch_params['lr']))

        self.arch_params['loss_weights'] = loss_weights
        self.arch_params['loss_fns'] = [lf.__name__ for lf in loss_fns]

        with open(os.path.join(self.exp_dir, 'arch_params.json'), 'w') as f:
            json.dump(self.arch_params, f)
        with open( os.path.join( self.exp_dir, 'data_params.json'), 'w') as f:
            json.dump(self.data_params, f)


    def load_data(self, load_n=None):
        # set loggers so that dataset will log any messages while loading volumes
        self.dataset.logger = self.logger
        self.dataset.profiler_logger = self.profiler_logger

        # by default, mri dataset loads ims as X and segs as Y
        (self.X_target_train, _, _, self.target_train_files), \
        (self.X_source_train,
         self.segs_source_train, self.contours_source_train,
         self.source_train_files), \
        (self.X_target_test, _, _, self.target_test_files), self.label_mapping \
            = self.dataset.load_dataset(
            load_n=load_n,
            load_source_segs=False,
            load_source_contours=True,
            load_segs=False,
        )

        # we use the same source volumes for training and validation
        self.X_source_test = self.X_source_train
        self.source_test_files = self.source_train_files

        if 'color' in self.arch_params['model_arch']:
            self.flow_fwd_model = load_model(
                self.arch_params['flow_fwd_model'],
                custom_objects={
                    'SpatialTransformer': functools.partial(
                        nrn_layers.SpatialTransformer,
                        indexing='xy')
                },
                compile=False
            )
            self.flow_bck_model = load_model(
                self.arch_params['flow_bck_model'],
                custom_objects={
                    'SpatialTransformer': functools.partial(
                        nrn_layers.SpatialTransformer,
                        indexing='xy')
                },
                compile=False
            )

            if self.X_source_train.shape[0] == 1 and self.recon_loss_name == 'l2-src':
                # back-warp all target vols to the source space. We can only do this for single atlas, and only
                # if we want to compute the reconstruction loss in the src space
                for i in range(self.X_target_train.shape[0]):
                    if i % 10 == 0:
                        self.logger.debug('Back-warping target example {} of {}'.format(
                            i, self.X_target_train.shape[0]))
                    preds = self.flow_bck_model.predict([
                        self.X_target_train[[i]], self.X_source_train[[0]]])

                    # assumes that transformed vol is the first pred
                    # TODO: if this is a bidir model, then back-warped vol is the 2nd pred
                    self.X_target_train[i] = preds[0]

                for i in range(self.X_target_test.shape[0]):
                    # warp our target towards our source space
                    preds = self.flow_bck_model.predict([
                        self.X_target_test[[i]], self.X_source_train[[0]]])

                    # assumes that transformed vol is the first pred
                    self.X_target_test[i] = preds[0]


        self.n_labels = len(self.label_mapping)

        self.logger.debug('X source/target train shapes')
        self.logger.debug(self.X_source_train.shape)
        self.logger.debug(self.X_target_train.shape)
        self.logger.debug('X source/target test shapes')
        self.logger.debug(self.X_source_test.shape)
        self.logger.debug(self.X_target_test.shape)

        assert set(self.target_train_files).isdisjoint(set(self.target_test_files))

        self.data_params['source_train_files'] = self.source_train_files
        self.data_params['source_test_files'] = self.source_test_files
        self.data_params['target_train_files'] = self.target_train_files
        self.data_params['target_test_files'] = self.target_test_files
        self.img_shape = self.X_source_train.shape[1:]

        # save info about our datasets in experiment dir
        with open( os.path.join( self.exp_dir, 'source_train_files.txt'), 'w') as f:
            f.writelines( [s + '\n' for s in self.source_train_files] )
        with open( os.path.join( self.exp_dir, 'target_train_files.txt'), 'w') as f:
            f.writelines( [s + '\n' for s in self.target_train_files] )
        with open( os.path.join( self.exp_dir, 'source_test_files.txt'), 'w') as f:
            f.writelines( [s + '\n' for s in self.source_test_files] )
        with open( os.path.join( self.exp_dir, 'target_test_files.txt'), 'w') as f:
            f.writelines( [s + '\n' for s in self.target_test_files] )


    def save_models(self, epoch, iter_count=None):
        super(TransformModelTrainer, self).save_models(epoch, iter_count=iter_count)


    def _create_flow_model(self):
        # parse the flow architecture name to create the correct model
        if 'flow_fwd' in self.arch_params['model_arch'] \
                or 'flow_bck' in self.arch_params['model_arch']:
            # train a fwd model only
            nf_enc = [16, 32, 32, 32]
            nf_dec = [32, 32, 32, 32, 32, 16, 16]

            self.transform_model = networks.cvpr2018_net(
                vol_size=(160, 192, 224),
                enc_nf=nf_enc,
                dec_nf=nf_dec,
                indexing='xy',
                name=self.arch_params['model_arch']
            )

            self.models = [self.transform_model]
        elif 'bidir_separate' in self.arch_params['model_arch']:
            # train a fwd model and back model
            nf_enc = [16, 32, 32, 32]
            nf_dec = [32, 32, 32, 32, 32, 16, 16]

            self.flow_bck_model = networks.cvpr2018_net(
                vol_size=(160, 192, 224),
                enc_nf=nf_enc,
                dec_nf=nf_dec,
                indexing='xy',
                name='vm_bidir_bck_model'
            )
            self.flow_models = [self.flow_bck_model]

            # vm2 model
            self.flow_fwd_model = networks.cvpr2018_net(
                vol_size=(160, 192, 224),
                enc_nf=nf_enc,
                dec_nf=nf_dec,
                indexing='xy',
                name='vm_bidir_fwd_model'
            )

            self.transform_model = networks.bidir_wrapper(
                img_shape=self.img_shape,
                fwd_model=self.flow_fwd_model,
                bck_model=self.flow_bck_model,
            )

            self.models += [self.flow_fwd_model, self.flow_bck_model, self.transform_model]
        else:
            raise NotImplementedError('Only separate bidirectional spatial transform models are implemented in this version!')

        if 'init_weights_from' in self.arch_params.keys():
            # this is not the right indexing, but it doesnt matter since we are only loading conv weights
            init_weights_from_models = [
                load_model(
                    m,
                    custom_objects={
                        'SpatialTransformer': nrn_layers.SpatialTransformer
                    },
                    compile=False
                    ) if m is not None else None for m in self.arch_params['init_weights_from']
            ]

            for mi, m in enumerate(self.models):
                # nothing to load from for this model, skip it
                if mi >= len(init_weights_from_models) or init_weights_from_models[mi] is None:
                    continue

                for li, l in enumerate(m.layers):
                    if li >= len(init_weights_from_models[mi].layers):
                        break

                    # TODO: this assumes matching layer nums, roughly...
                    init_from_layer = init_weights_from_models[mi].layers[li]
                    if 'conv' in l.name.lower()	and 'conv' in init_from_layer.name.lower():
                        our_weights = l.get_weights()
                        init_from_weights = init_from_layer.get_weights()

                        if np.all(our_weights[0].shape == init_from_weights[0].shape):
                            m.layers[li].set_weights(init_from_weights)
                            self.logger.debug('Copying weights from {} layer {} to {} layer {}'.format(
                                init_weights_from_models[mi].name,
                                init_from_layer.name,
                                m.name,
                                l.name))
                        else:
                            self.logger.debug('Unable to copy weights from {} layer {} to {} layer {}, shapes {} and {}'.format(
                                init_weights_from_models[mi].name,
                                init_from_layer.name,
                                m.name,
                                l.name,
                                our_weights[0].shape,
                                init_from_weights[0].shape
                            ))
            #self.flow_fwd_model, self.flow_bck_model = self.models[:2]
            if 'bidir_separate' in self.arch_params['model_arch']:
                # recreate wrapper?
                self.transform_model = networks.bidir_wrapper(
                    img_shape=self.img_shape,
                    fwd_model=self.models[0],
                    bck_model=self.models[1],
                )
                self.models[-1] = self.transform_model


    def _create_color_model(self):
        if self.arch_params['use_aux_reg'] is None:
            # no auxiliary input (e.g. contours, segmentations)
            self.aux_input_shape = None
        else:
            if 'segs_oh' in self.arch_params['use_aux_reg']:
                self.aux_input_shape = tuple(self.segs_source_train.shape[1:-1]) + (self.n_labels,)
            elif 'segs' in self.arch_params['use_aux_reg']:
                self.aux_input_shape = self.segs_source_train.shape[1:]
            else:
                self.aux_input_shape = None

            # if contours are also included, add then in a stack. otherwise, it is the only aux input
            if 'contours' in self.arch_params['use_aux_reg'] and self.aux_input_shape is not None:
                self.aux_input_shape = self.aux_input_shape[:-1] + (self.aux_input_shape[-1] + 1,)
            elif 'contours' in self.arch_params['use_aux_reg'] and self.aux_input_shape is None:
                self.aux_input_shape = self.contours_source_train.shape[1:]

        self.logger.debug('Auxiliary input shape: {}'.format(self.aux_input_shape))

        # parse the color architecture name to create the correct model
        if 'unet' in self.arch_params['model_arch']:
            color_model_name = 'color_delta_unet'

            # TODO: include a src-to-tgt space warp model in here if we want to compute recon in the tgt space
            self.transform_model = networks.color_delta_unet_model(
                img_shape=self.img_shape,
                n_output_chans=self.n_chans,
                enc_params={
                    'nf_enc': [16, 32, 32, 32, 32, 32],
                    'nf_dec': [64, 64, 32, 32, 32, 16, 16],
                    'use_maxpool': True,
                    'use_residuals': False,
                    'n_convs_per_stage': 1,
                },
                model_name=color_model_name,
                include_aux_input=self.arch_params['use_aux_reg'] is not None and self.arch_params['do_include_aux_input'],
                aux_input_shape=self.aux_input_shape,
                do_warp_to_target_space='tgt' in self.recon_loss_name,
            )
            self.models += [self.transform_model]


    def create_models(self):
        self.models = []
        if 'flow' in self.arch_params['model_arch']:
            self._create_flow_model()
        elif 'color' in self.arch_params['model_arch']:
            self._create_color_model()

        super(TransformModelTrainer, self).create_models()
        return self.models


    def load_models(self, load_epoch=None, stop_on_missing=True, init_layers=False):
        start_epoch = super(TransformModelTrainer, self).load_models(load_epoch,
            stop_on_missing=stop_on_missing)
        return start_epoch



    def create_generators(self, batch_size):
        self.batch_size = batch_size

        source_vol_gen = self.dataset.gen_vols_batch(
            dataset_splits=['labeled_train'],
            batch_size=batch_size,
            load_segs=self.arch_params['use_aux_reg'] is not None and 'segs' in self.arch_params['use_aux_reg'],
            load_contours=self.arch_params['use_aux_reg'] is not None and 'contours' in self.arch_params['use_aux_reg'],
            randomize=True,
            return_ids=True,
        )

        target_train_vol_gen = self.dataset.gen_vols_batch(
            dataset_splits=['unlabeled_train', 'labeled_train'],
            batch_size=batch_size,
            load_segs=False, load_contours=False,
            randomize=True,
            return_ids=True
        )

        target_valid_vol_gen = self.dataset.gen_vols_batch(
            dataset_splits=['labeled_valid'],
            batch_size=batch_size,
            load_segs=False, load_contours=False,
            randomize=True,
            return_ids=True
        )

        self.train_gen = self._generate_source_target_pairs(
            source_vol_gen=source_vol_gen,
            target_vol_gen=target_train_vol_gen,
            return_ids=True
        )

        self.valid_gen = self._generate_source_target_pairs(
            source_vol_gen=source_vol_gen,
            target_vol_gen=target_valid_vol_gen,
            return_ids=True
        )


    def _generate_source_target_pairs(self, source_vol_gen=None, target_vol_gen=None, return_ids=False):
        if self.X_source_train.shape[0] == 1:
            # single atlas, no need to sample from generator
            X_source = self.X_source_train
            id_source = [self.source_train_files[0]]
            source_aux_inputs = self.contours_source_train

        while True:
            # if there is more than one source volume, sample it
            if self.X_source_train.shape[0] > 1:
                X_source, segs_source, source_aux_inputs, id_source = next(source_vol_gen)

            # sample a random target volume
            X_target, _, _, id_target = next(target_vol_gen)

            if self.arch_params['model_arch'] == 'flow_fwd':
                inputs = [X_source, X_target]
                targets = [X_target, X_target] # target, flow reg
            elif self.arch_params['model_arch'] == 'flow_bck':
                inputs = [X_target, X_source]
                targets = [X_source, X_source]  # source, flow reg
            elif self.arch_params['model_arch'] == 'flow_bidir_separate':
                inputs = [X_source, X_target]
                # forward target, backward target, forward flow reg, backward flow reg
                targets = [X_target, X_source, X_target, X_source]
            elif 'color' in self.arch_params['model_arch']:
                if self.X_source_train.shape[0] > 1 and self.recon_loss_name == 'l2-src' \
                        or self.recon_loss_name == 'l2-tgt':
                    # more than one atlas, so we need to back-warp depending on our atlas
                    # OR, if we are computing reconstruction loss in the target space,
                    # we still need to give the color model the src-space target
                    X_target_srcspace = self.flow_bck_model.predict([X_target, X_source])[0]

                inputs = [X_source, X_target_srcspace, source_aux_inputs]

                # target image, and two dummy outputs for the transform and aux labels, 
                # which we will use when we compute the regularization loss
                targets = [X_target] * 3 

                if self.recon_loss_name == 'l2-tgt':
                    _, flow_batch = self.flow_fwd_model.predict([X_source, X_target])
                    # reconstruction loss in the target space
                    inputs += [flow_batch]

            if not return_ids:
                yield inputs, targets
            else:
                yield inputs, targets, id_source, id_target


    def make_train_results_im(self):
        return self._make_results_im(self.train_gen)

    def make_test_results_im(self):
        return self._make_results_im(self.valid_gen)


    def eval(self):
        return 0


    def get_n_train(self):
        return min(100, self.X_target_train.shape[0])

    def get_n_test(self):
        return min(100, self.X_target_test.shape[0])

    def save_exp_info(self, exp_dir, figures_dir, logs_dir, models_dir):
        return 0

    def update_epoch_count(self, epoch):
        self.epoch_count += 1
        return 0

    def train_on_batch(self):
        X, Y, X_oh, Y_oh = next(self.train_gen)

        start = time.time()
        loss_vals = self.transform_model.train_on_batch(X, Y)

        if self.do_profile:
            self.profiler_logger.info('train_on_batch took {}'.format(time.time() - start))
            self.profiled_iters += 1

            if self.profiled_iters > 100:
                self.do_profile = False
        loss_names = ['train_' + ln for ln in self.loss_names]
        assert len(loss_vals) == len(loss_names)
        return loss_vals, loss_names


    def test_batches(self):
        n_test_batches = max(1, int(np.ceil(self.get_n_test() / self.batch_size)))
        self.logger.debug('Testing {} batches'.format(n_test_batches))
        for i in range(n_test_batches):
            X, Y, X_oh, Y_oh = next(self.valid_gen)

            loss_names = ['test_' + ln for ln in self.loss_names]
            test_loss = np.asarray(
                self.transform_model.evaluate(
                    X, Y,
                    verbose=False))

            if i == 0:
                total_test_loss = test_loss
            else:
                total_test_loss += test_loss
            assert len(total_test_loss) == len(loss_names)

        return (total_test_loss / float(n_test_batches)).tolist(), loss_names


    def _make_results_im(self, data_gen, max_batch_size=32):
        inputs, targets, ids_source, ids_target = next(data_gen)
        preds = self.transform_model.predict(inputs)

        input_im_batches = inputs[:2]
        labels = [
            [os.path.basename(ids) for ids in ids_source],
            [os.path.basename(idt) for idt in ids_target]]
        do_normalize = [False, False]

        if self.arch_params['use_aux_reg'] is not None and 'contours' in self.arch_params['use_aux_reg']:
            input_im_batches += [inputs[2][..., [-1]]]
            labels += ['aux_contours']
            do_normalize += [True]

        if 'bidir' in self.arch_params['model_arch']:
            # fwd flow, fwd transformed im
            input_im_batches += [preds[2], preds[0]]
        else:
            # spatial and appearance transforom models both output [transformed, transform, ...]
            input_im_batches += [preds[1], preds[0]]
        labels += ['transform', 'transformed']

        # if we are learning a color transform, normalize it for display purposes
        do_normalize += ['color' in self.arch_params['model_arch'], False]

        # batch_size = inputs_im.shape[0]
        batch_size = self.batch_size
        display_batch_size = min(max_batch_size, batch_size)
        zeros_batch = np.zeros((batch_size,) + self.img_shape)

        if display_batch_size < batch_size:
            input_im_batches = [batch[:display_batch_size] for batch in input_im_batches]

        if do_normalize is None:
            do_normalize = [False] * len(input_im_batches)

        if self.n_dims == 2:
            out_im = np.concatenate([
                utils.label_ims(
                    batch, labels[i],
                    normalize=do_normalize[i]
                ) for i, batch in enumerate(input_im_batches)
            ], axis=1)
        else:
            # pick a slice that is somewhat in the middle
            slice_idx = np.random.choice(
                range(int(round(self.img_shape[-2] * 0.25)), int(round(self.img_shape[-2] * 0.75))),
                1, replace=False)

            out_im = np.concatenate([
                utils.label_ims(
                    batch[:, :, :, slice_idx[0]], labels[i],
                    normalize=do_normalize[i]
                ) for i, batch in enumerate(input_im_batches)
            ], axis=1)

        return out_im
