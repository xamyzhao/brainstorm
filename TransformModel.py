import functools
import json
import os
import sys
import time

import numpy as np

import keras.metrics as keras_metrics
from keras.optimizers import Adam

import mri_loader

import voxelmorph_networks as my_voxelmorph_networks
import brainstorm_networks

sys.path.append('../evolving_wilds')
from cnn_utils import classification_utils, file_utils, vis_utils
from cnn_utils import metrics as my_metrics
from cnn_utils import ExperimentClassBase

sys.path.append('../voxelmorph')
import src.losses as vm_losses
import src.networks as vm_networks

sys.path.append('../neuron')
import neuron.layers as nrn_layers


class TransformModelTrainer(ExperimentClassBase.Experiment):
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
					exp_name += '-sigIw{}'.format(self.sigma_Iw)
				elif 'cc' in self.recon_loss_name:
					exp_name += '-win{}'.format(self.cc_win_size_Iw)
					exp_name += '-wt{}'.format(self.cc_loss_weight)

		elif 'color' in self.arch_params['model_arch']:
			exp_name += '_invflow-{}'.format(os.path.splitext(os.path.basename(
				self.arch_params['flow_bck_model'].split('/models/'
			)[0]))[0])
			if 'pretrain_flow' in self.arch_params.keys():
				exp_name += '_pt{}'.format(self.arch_params['pretrain_flow'])

			if self.arch_params['input_aux_labels'] is not None:
				if 'segs_oh' in self.arch_params['input_aux_labels']:
					exp_name += '_insegsoh'
				elif 'segs' in self.arch_params['input_aux_labels']:
					exp_name += '_insegs'

				if 'contours' in self.arch_params['input_aux_labels']:
					exp_name += '_incontours'

			# color smoothness and reconstruction losses
			if self.transform_reg_name is not None:
				exp_name += '_{}_regcwt{}'.format(self.transform_reg_name,
				                                  self.transform_reg_wt)

			if self.recon_loss_name is not None:
				exp_name += '_{}'.format(self.recon_loss_name)
				if 'l2' in self.recon_loss_name:
					exp_name += '_sigI{}'.format(self.sigma_I)
					
		self.model_name = exp_name

		exp_name = super(TransformModelTrainer, self).get_model_name()
		self.model_name = exp_name
		return exp_name


	def __init__(self, data_params, arch_params, exp_root='experiments'):
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

		if 'lr' in arch_params.keys():
			self.lr = arch_params['lr']

		if 'input_aux_labels' not in arch_params.keys():
			self.arch_params['input_aux_labels'] = None

		# enc/dec architecture
		# parse params for flow portion of network
		if 'flow' in self.arch_params['model_arch']:
			self.transform_reg_name = self.arch_params['transform_reg_flow']

			if 'grad_l2_vm' in self.transform_reg_name:
				# debugging gradient function, try using voxelmorph's
				self.transform_reg_fn = vm_losses.gradientLoss(penalty='l2')
				self.transform_reg_wt = self.arch_params['transform_reg_lambda_flow']
			elif 'grad_l2' in self.transform_reg_name:
				self.transform_reg_fn = my_metrics.gradient_loss_l2(n_dims=self.n_dims).compute_loss
				self.transform_reg_wt = self.arch_params['transform_reg_lambda_flow']
			elif 'prec' in self.transform_reg_name:
				from metrics import VoxelmorphMetrics
				self.transform_reg_fn = VoxelmorphMetrics(alpha=1.).smoothness_precision_loss_zeromean
				# let's just use lambda_flow to represent alpha
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
#				self.reconstruction_loss_fn_flow = my_metrics.cc2D_loss(self.cc_win_size_Iw, n_chans=self.n_chans)#, n_dims=self.n_dims)
				self.recon_loss_fn = vm_losses.NCC().loss
				self.recon_loss_wt = self.cc_loss_weight
				self.sigma_Iw = None
			elif 'cc' in self.recon_loss_name:
				self.cc_loss_weight = self.arch_params['cc_loss_weight']
				self.cc_win_size_Iw = self.arch_params['cc_win_size_Iw']
#				self.reconstruction_loss_fn_flow = my_metrics.cc2D_loss(self.cc_win_size_Iw, n_chans=self.n_chans)#, n_dims=self.n_dims)
				self.recon_loss_fn = my_metrics.ccnD(
					self.cc_win_size_Iw, n_chans=self.n_chans, n_dims=self.n_dims)
				self.recon_loss_wt = self.cc_loss_weight
				self.sigma_Iw = None

		# parse params for color portion of network
		if 'color' in self.arch_params['model_arch']:
			self.recon_loss_name = self.arch_params['recon_loss_I']
			self.transform_reg_name = self.arch_params['transform_reg_color']
			if 'grad_l2' in self.transform_reg_name:
				self.transform_reg_fn = my_metrics.gradient_loss_l2(n_dims=self.n_dims).compute_loss
				self.transform_reg_wt = self.arch_params['transform_reg_lambda_color']
			elif 'prec' in self.transform_reg_name:
				from metrics import VoxelmorphMetrics
				self.transform_reg_fn = VoxelmorphMetrics(alpha=1.).smoothness_precision_loss_zeromean
				# let's just use lambda_color to represent alpha
				self.transform_reg_wt = self.arch_params['transform_reg_lambda_color']
			elif 'seg-l2' in self.transform_reg_name:
				self.transform_reg_wt = self.arch_params['transform_reg_lambda_color']
				self.transform_reg_fn = my_metrics.SpatialSegmentSmoothness(
					n_dims=self.n_dims,
					n_chans=self.n_chans,
				).compute_loss
			elif 'grad-si-l2_l1reg' in self.transform_reg_name:
				# gradient in space and intensity, l2 regularization
				self.transform_reg_fn = my_metrics.SummedLosses(
					loss_fns=[
						my_metrics.SpatialIntensitySmoothness(
							n_dims=self.n_dims,
							n_chans=self.n_chans,
							use_true_gradients='predgrad' in self.transform_reg_name,
						).compute_loss, 
						my_metrics.l1_norm],
						loss_weights=arch_params['transform_reg_lambdas_color'],
				).compute_loss

				self.transform_reg_wt = self.arch_params['transform_reg_lambda_color']
			elif 'grad-si-l2' in self.transform_reg_name:
				# gradient in space and intensity, l2 regularization
				self.transform_reg_fn = my_metrics.SpatialIntensitySmoothness(
					n_dims=self.n_dims,
					n_chans=self.n_chans,
					use_true_gradients='predgrad' in self.transform_reg_name,
				).compute_loss

				self.transform_reg_wt = self.arch_params['transform_reg_lambda_color']
			else:
				self.transform_reg_fn = None
				self.transform_reg_wt = 0.

			if self.recon_loss_name is None:  # still have this output node, but don't weight it
				self.recon_loss_fn = keras_metrics.mean_squared_error
				self.recon_loss_wt = 0
			elif 'l2' in self.recon_loss_name:
				self.sigma_I = self.arch_params['sigma_I']
				self.recon_loss_fn = keras_metrics.mean_squared_error

				# set a constant weight for reconstruction
				self.recon_loss_wt = 0.5 / self.sigma_I ** 2

			#if self.arch_params['pretrain_flow'] > 0:
			#	self.started_seq_training = False
			#	self.color_lambda = 0
		else:
			# no color transform
			self.arch_params['pretrain_flow'] = 0

		if 'latest_epoch' in arch_params.keys():
			self.latest_epoch = arch_params['latest_epoch']
		else:
			self.latest_epoch = 0

		self.get_model_name()

		self.model_name, \
		self.exp_dir, \
		self.figures_dir, self.logs_dir, self.models_dir \
			= file_utils.make_output_dirs(self.model_name, exp_root='./{}/'.format(exp_root))

		super(TransformModelTrainer, self).__init__(data_params=self.data_params, arch_params=self.arch_params)

	def get_dirs(self):
		return self.exp_dir, self.figures_dir, self.logs_dir, self.models_dir

	def compile_models(self, run_options=None, run_metadata=None):
		if 'color' in self.arch_params['model_arch']:  # if we have a color transform, we might need to update some losses
			# point all of these regularizations at the color model inputs -- we assume everything
			# has been back-warped to the source space
			if 'grad-si-l2_l1reg' in self.transform_reg_name:  # do this here since we need to point to the model
				self.transform_reg_fn = my_metrics.SummedLosses(
					loss_fns=[
						my_metrics.SpatialIntensitySmoothness(
							n_dims=self.n_dims,
							n_chans=self.n_chans,
							use_true_gradients='predgrad' in self.transform_reg_name,
							pred_image_output=self.transform_model.get_layer('input_src').output
						).compute_loss,
						my_metrics.l1_norm,
					], loss_weights=[1, 1]).compute_loss
			elif 'seg-l2' in self.transform_reg_name and self.started_seq_training:  # otherwise the layer will not exist yet
				self.transform_reg_fn = my_metrics.SpatialSegmentSmoothness(
					n_dims=self.n_dims,
					n_chans=self.n_chans,
					warped_contours_layer_output=self.transform_model.get_layer('input_src_seg').output
				).compute_loss

			elif 'grad-si-l2' in self.transform_reg_name:  # do this here since we need to point to the model
				self.transform_reg_fn = my_metrics.SpatialIntensitySmoothness(
					n_dims=self.n_dims,
					n_chans=self.n_chans,
					use_true_gradients='predgrad' in self.transform_reg_name,
					pred_image_output=self.transform_model.get_layer('input_src').output
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
			loss_fns = [self.transform_reg_fn, self.recon_loss_fn]
			loss_weights = [self.transform_reg_wt, self.recon_loss_wt]
			self.loss_names = [self.transform_reg_name, self.recon_loss_name]


		self.logger.debug('Transform model')
		self.transform_model.summary(print_fn=self.logger.debug, line_length=120)

		self.loss_names = ['total'] + self.loss_names
		self.logger.debug('Compiling full VTE model with {} losses: {}'.format(len(loss_fns), self.loss_names))
		for li, lf in enumerate(loss_fns):
			self.logger.debug('Model output: {}, loss fn: {}'.format(
				self.transform_model.outputs[li],
				lf))
		self.logger.debug('and {} weights {}'.format(len(loss_weights), loss_weights))
		print([type(w) for w in loss_weights])
		if run_options is not None:
			self.transform_model.compile(loss=loss_fns, loss_weights=loss_weights,
		                           optimizer=Adam(lr=self.lr), 
									options=run_options, run_metadata=run_metadata,
			)

		else:
			self.transform_model.compile(loss=loss_fns, loss_weights=loss_weights,
		                           optimizer=Adam(lr=self.lr))

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
		(self.X_source_train,
		 self.segs_source_train, self.contours_source_train,
		 self.source_train_files), \
		(self.X_target_train, _, _, self.target_train_files), \
		(self.X_source_test, self.segs_source_test, self.contours_source_test, self.source_test_files), \
		(self.X_target_test, _, _, self.target_test_files), self.label_mapping \
			= self.dataset.load_source_target(
			load_n=load_n,
			load_source_segs=self.arch_params['input_aux_labels'] is not None
			                 and 'segs' in self.arch_params['input_aux_labels'])


		if 'color' in self.arch_params['model_arch'] \
				and not self.arch_params['color_transform_in_tgt_space'] \
				and 'src' in self.recon_loss_name:  # we are computing color recon loss in src space
			# warp all target volumes back to source space

			from keras.models import load_model
			flow_bck_model = load_model(
				self.arch_params['flow_bck_model'],
				custom_objects={
					'SpatialTransformer': functools.partial(
						nrn_layers.SpatialTransformer,
						indexing='xy')
				},
				compile=False
			)


			# back-warp all target vols to the source space
			# we shouldn't have to deal with aux inputs since those should be in the input space already
			for i in range(self.X_target_train.shape[0]):
				if i % 10 == 0:
					self.logger.debug('Back-warping target example {} of {}'.format(
						i, self.X_target_train.shape[0]))
				preds = flow_bck_model.predict([
					self.X_target_train[[i]], self.X_source_train[[0]]])

				# assumes that transformed vol is the first pred
				# TODO: if this is a bidir model, then back-warped vol is the 2nd pred
				self.X_target_train[i] = preds[0]

			for i in range(self.X_target_test.shape[0]):
				# warp our target towards our source space
				preds = flow_bck_model.predict([
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
		if 'voxelmorph2guha' in self.arch_params['model_arch']:
			self.transform_model = my_voxelmorph_networks.voxelmorph_wrapper(
				img_shape=self.img_shape,
				voxelmorph_arch='vm2_guha'
			)
			self.models += [self.transform_model]

		elif 'bidir_separate' in self.arch_params['model_arch']:
			# TODO: separate forward/backward models, or true miccai bidir
			# hack for subpar bidir performance -- load a fwd model and back model
			nf_enc = [16, 32, 32, 32]
			nf_dec = [32, 32, 32, 32, 32, 16, 16]

			self.flow_bck_model = vm_networks.cvpr2018_net(
				vol_size=(160, 192, 224),
				enc_nf=nf_enc,
				dec_nf=nf_dec,
				indexing='xy'
			)
			self.flow_bck_model.name = 'vm_bidir_bck_model'
			self.flow_models = [self.flow_bck_model]
			
			# vm2 model
			self.flow_fwd_model = vm_networks.cvpr2018_net(
				vol_size=(160, 192, 224),
				enc_nf=nf_enc,
				dec_nf=nf_dec,
				indexing='xy'
			)
			self.flow_fwd_model.name = 'vm_bidir_fwd_model'

			# TODO: wrapper model! otherwise this will not train
			import voxelmorph_networks
			self.transform_model = voxelmorph_networks.bidir_wrapper(
				img_shape=self.img_shape,
				fwd_model=self.flow_fwd_model,
				bck_model=self.flow_bck_model,
			)

			self.models += [self.flow_fwd_model, self.flow_bck_model, self.transform_model]

		elif 'bidir' in self.arch_params['model_arch']:
			nf_enc = [16, 32, 64, 64]
			nf_dec = [64, 64, 64, 32, 32, 32, 16]

			# TODO: check this
			self.transform_model = vm_networks.miccai2018_bidir(
				vol_size=(160, 192, 224),
				enc_nf=nf_enc,
				dec_nf=nf_dec,
				use_miccai_int=True,
				indexing='xy',
				bidir=True,
				halfres=True,
			)
			self.transform_model.name = 'vm_bidir_model'
			self.models += [self.transform_model]

		if 'init_weights_from' in self.arch_params.keys():
			from keras.models import load_model
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
				import voxelmorph_networks
				self.transform_model = voxelmorph_networks.bidir_wrapper(
					img_shape=self.img_shape,
					fwd_model=self.models[0],
					bck_model=self.models[1],
				)
				self.models[-1] = self.transform_model


	def _create_color_model(self):
		if self.arch_params['input_aux_labels'] is None:
			self.aux_input_shape = None
		else:
			if 'segs_oh' in self.arch_params['input_aux_labels']:
				self.aux_input_shape = tuple(self.segs_source_train.shape[1:-1]) + (self.n_labels,)
			elif 'segs' in self.arch_params['input_aux_labels']:
				self.aux_input_shape = self.segs_source_train.shape[1:]
			else:
				self.aux_input_shape = None

			# if contours are also included, add then in a stack. otherwise, it is the only aux input
			if 'contours' in self.arch_params['input_aux_labels'] and self.aux_input_shape is not None:
				self.aux_input_shape = self.aux_input_shape[:-1] + (self.aux_input_shape[-1] + 1,)
			elif 'contours' in self.arch_params['input_aux_labels'] and self.aux_input_shape is None:
				self.aux_input_shape = self.contours_source_train.shape[1:]

		self.logger.debug('Auxiliary input shape: {}'.format(self.aux_input_shape))

		# parse the color architecture name to create the correct model
		if 'unet' in self.arch_params['model_arch']:
			color_model_name = 'color_delta_unet'
			if 'color_transform_in_tgt_space' in self.arch_params.keys() and self.arch_params[
				'color_transform_in_tgt_space']:
				color_model_name += '_tgtspace'
			else:
				color_model_name += '_srcspace'

			# TODO: include a src-to-tgt space warp model in here if we want to compute recon in the tgt space
			self.transform_model = brainstorm_networks.color_delta_unet_model(
				img_shape=self.img_shape,
				n_output_chans=self.n_chans,
				enc_params={
					'nf_enc': [16, 32, 32, 32],
					'nf_dec': [32, 32, 32, 32, 32, 16, 16],
					'use_maxpool': True,
					'use_residuals': False,
					'n_convs_per_stage': 1,	
				},
				model_name=color_model_name,
				include_aux_input=self.arch_params['input_aux_labels'] is not None,
				aux_input_shape=self.aux_input_shape,
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



	# TODO: streamline generators since we no longer need to get same-class pairs
	def create_generators(self, batch_size):
		self.batch_size = batch_size

		target_train_vol_gen = self.dataset.gen_vols_batch(
			dataset_splits=['unlabeled_train', 'labeled_train'],
			batch_size=batch_size, load_segs=False, randomize=True)
		target_valid_vol_gen = self.dataset.gen_vols_batch(
			dataset_splits=['labeled_valid'],
			batch_size=batch_size, load_segs=False, randomize=True)

		# returns inputs, targets
		self.train_gen = self._generate_source_target_pairs(
			self.batch_size, target_vol_gen=target_train_vol_gen)
		# for printing images
		self.train_gen_verbose = self._generate_source_target_pairs(
			self.batch_size,
			target_vol_gen=self.dataset.gen_vols_batch(
				dataset_splits=['unlabeled_train', 'labeled_train'],
				batch_size=batch_size, load_segs=False, randomize=True,
				return_ids=True),
			return_ids=True
		)

		self.valid_gen = self._generate_source_target_pairs(
			self.batch_size, target_vol_gen=target_valid_vol_gen)

		# for printing images
		self.valid_gen_verbose = self._generate_source_target_pairs(
			self.batch_size,
			target_vol_gen=self.dataset.gen_vols_batch(
				dataset_splits=['labeled_valid'],
				batch_size=batch_size, load_segs=False, randomize=True,
				return_ids=True),
			return_ids=True
		)

		# make a new generator just for sampling, in case we are using fit_generator and it is already
		# using the other test_gen
		self.sample_tgt_valid_gen = self.dataset.gen_vols_batch(
			dataset_splits=['labeled_valid'],
			batch_size=batch_size, load_segs=False, randomize=True)

		self.sample_tgt_train_gen = self.dataset.gen_vols_batch(
			dataset_splits=['unlabeled_train', 'labeled_train'],
			batch_size=batch_size, load_segs=False, randomize=True)

		self.sample_tgt_train_valid_gen = self.dataset.gen_vols_batch(
			dataset_splits=['unlabeled_train', 'labeled_train', 'labeled_valid'],
			batch_size=batch_size, load_segs=False, randomize=True)



	def _generate_source_target_pairs(self, batch_size, source_vol_gen=None, target_vol_gen=None, return_ids=False):
		# create source aux input here in case we need it later
		if self.arch_params['input_aux_labels'] is not None:
			if 'segs_oh' in self.arch_params['input_aux_labels']:
				# first channel will be segs in label form
				Y_source_onehot = classification_utils.labels_to_onehot(
					self.segs_source_train[..., [0]], label_mapping=self.label_mapping)
				self.source_aux_inputs = Y_source_onehot
			elif 'segs' in self.arch_params['input_aux_labels']:
				self.source_aux_inputs = self.segs_source_train
			else:
				self.source_aux_inputs = None

			if 'contours' in self.arch_params['input_aux_labels'] and self.source_aux_inputs is not None:
				self.source_aux_inputs = np.concatenate([self.source_aux_inputs, self.contours_source_train], axis=-1)
			elif 'contours' in self.arch_params['input_aux_labels'] and self.source_aux_inputs is None:
				self.source_aux_inputs = self.contours_source_train

		while True:
			if source_vol_gen is None and self.X_source_train.shape[0] == 1:
				X_source = self.X_source_train  # don't sample this
				id_source = os.path.splitext(os.path.basename(self.source_train_files[0]))[0]
			else:
				X_source, Y_source = next(source_vol_gen)
				id_source = None

			if return_ids:
				X_target, Y_target, id_target = next(target_vol_gen)
			else:
				X_target, Y_target = next(target_vol_gen)
				id_target = None

			if self.arch_params['input_aux_labels'] is not None:
				inputs = [X_source, X_target, self.source_aux_inputs]
			else:
				inputs = [X_source, X_target]

			if 'bidir' in self.arch_params['model_arch']:
				# forward target, backward target, forward flow reg, backward flow reg
				targets = [X_target, X_source, X_target, X_source]
			else:
				targets = [X_target] * 2

			if not return_ids:
				yield inputs, targets
			else:
				yield inputs, targets, id_source, id_target


	def make_train_results_im(self):
		inputs, targets, id_source,  ids_target = next(self.train_gen_verbose)
		preds = self.transform_model.predict(inputs)

		# TODO: put logic of order of outputs in model class...
		ims = inputs[:2]
		labels = [id_source, ids_target]
		do_normalize = [False, False]
		
		if self.arch_params['input_aux_labels'] is not None \
				and 'segs_oh' in self.arch_params['input_aux_labels']:
			# last input will be aux info
			ims += [inputs[-1][..., 16]]
			labels += ['aux_oh']
			do_normalize += [True]

		if 'bidir' in self.arch_params['model_arch']:
			# fwd flow, fwd transformed im
			ims += [preds[i] for i in [2, 0]]
		else:
			ims += preds[:2]
		labels += ['transform', 'transformed']
		# if we are learning a color transform, normalize it
		do_normalize += ['color' in self.arch_params['model_arch'], False]

		return self._make_results_im(ims, labels, do_normalize)


	def _sample_from_prior(self, I=None, labels_onehot=None, batch_size=None, sample_from_gen=None, sample_targets_from=['valid'], verbose=True):
		if batch_size is None:
			batch_size = self.batch_size
		
		# if our dataset has a single source image
		if I is None:
			I = self.X_source_train
		else:
			labels = None
			aux_inputs = None

		if sample_from_gen is None:
			if 'valid' in sample_targets_from and 'train' in sample_targets_from:
				if verbose:
					self.logger.debug('Sampling tgt from train+valid set')
				sample_from_gen = self.sample_tgt_train_valid_gen
			elif 'valid' in sample_targets_from and 'train' not in sample_targets_from:
				if verbose:
					self.logger.debug('Sampling tgt from valid set')
				sample_from_gen = self.sample_tgt_valid_gen
			else:
				if verbose:
					self.logger.debug('Sampling tgt from train set')
			sample_from_gen = self.sample_tgt_train_gen

		tgt, _ = next(sample_from_gen)
		return [tgt]



	def make_test_results_im(self, epoch_num=None):
		inputs, targets, id_source, ids_target = next(self.valid_gen_verbose)
		preds = self.transform_model.predict(inputs)

		# TODO: put logic of order of outputs in model class...
		ims = inputs[:2]
		labels = [id_source, ids_target]
		do_normalize = [False, False]

		if self.arch_params['input_aux_labels'] is not None \
				and 'segs_oh' in self.arch_params['input_aux_labels']:
			ims += [inputs[-1][..., 16]]
			labels += ['aux_oh']
			do_normalize += [True]

		if 'bidir' in self.arch_params['model_arch']:
			# fwd flow, fwd transformed im
			ims += [preds[i] for i in [2, 0]]
		else:
			ims += preds[:2]
		labels += ['transform', 'transformed']
		# if we are learning a color transform, normalize it
		do_normalize += ['color' in self.arch_params['model_arch'], False]

		return self._make_results_im(ims, labels, do_normalize)


	def _make_eval_gen(self, batch_size=1):
		return 0

	def eval(self):
		return 0


	def get_n_train(self):
		return min(100, len(self.dataset.files_unlabeled_train))


	def get_n_test(self):
		return min(100, len(self.dataset.files_labeled_valid))


	def train_discriminator(self):
		return [], []


	def save_exp_info(self, exp_dir, figures_dir, logs_dir, models_dir):
		return 0



	def update_epoch_count(self, epoch):
		self.epoch_count += 1
		return 0


	def train_joint(self):
		X, Y, X_oh, Y_oh = next(self.train_gen_verbose)
		'''
		self.IJ_train_batch = X
		self.I_train_oh = X_oh
		self.J_train_oh = Y_oh
		'''
		start = time.time()
		loss_vals = self.transform_model.train_on_batch(
			X, Y)
		if self.do_profile:
			self.profiler_logger.info('train_on_batch took {}'.format(time.time() - start))
			self.profiled_iters += 1

			if self.profiled_iters > 100:
				self.do_profile = False
		loss_names = ['train_' + ln for ln in self.loss_names]
		assert len(loss_vals) == len(loss_names)
		return loss_vals, loss_names


	def test_joint(self):
		n_test_batches = max(1, int(np.ceil(self.get_n_test() / self.batch_size)))
		self.logger.debug('Testing {} batches'.format(n_test_batches))
		for i in range(n_test_batches):
			X, Y, X_oh, Y_oh = next(self.valid_gen_verbose)
			'''
			self.IJ_test_batch = X
			self.I_test_oh = X_oh
			self.J_test_oh = Y_oh
			'''
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



	def _make_results_im(self, input_im_batches, labels, do_normalize=None,
	                     max_batch_size=32):
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
				vis_utils.label_ims(
					batch, labels[i], 
					inverse_normalize=False, 
					normalize=do_normalize[i]
				) for i, batch in enumerate(input_im_batches)
				#vis_utils.label_ims(recon_pred, 'pred_recon', inverse_normalize=True),
			], axis=1)
		else:
			slice_idx = np.random.choice(self.img_shape[-2], 1, replace=False)
			out_im = np.concatenate([
				vis_utils.label_ims(
					batch[:, :, :, slice_idx[0]], labels[i], 
					inverse_normalize=False,
					normalize=do_normalize[i]
				) for i, batch in enumerate(input_im_batches)
				#vis_utils.label_ims(recon_pred, 'pred_recon', inverse_normalize=True),
			], axis=1)

	
		return out_im