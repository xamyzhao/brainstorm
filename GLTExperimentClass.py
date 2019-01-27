import copy
import json
import logging
import os
import shutil
import sys
import time

import cv2
import numpy as np
import re
import scipy.io as sio
import keras.metrics as keras_metrics
from keras.optimizers import Adam
import tensorflow as tf

from networks import SequentialTransformEncoders
from networks import CVAEClass
import networks.voxelmorph_networks as my_voxelmorph_networks
from networks import basic_networks, transform_network_utils

import classification_utils
import data_utils
import transform_utils

from ExperimentClassBase import Experiment
from dataset_utils import vte_data_loader

import metrics as my_metrics
from networks import color_transform_utils

sys.path.append('../')
from cnn_utils import file_utils, vis_utils

sys.path.append('../voxelmorph')
import src.losses as vm_losses
import src.networks as vm_networks




class ExperimentGlobalLocalTransforms(Experiment):
	def get_model_name(self):
		exp_name = 'GLT'

		long_dataset_name = '_{}-to-{}'.format(self.source_name, self.target_name)
		if self.source_name == self.target_name:
			short_dataset_name = '_{}-to-all'.format(self.source_name)
		else:
			short_dataset_name = long_dataset_name
		exp_name += short_dataset_name

		exp_name += '_{}'.format(self.arch_params['model_arch'])

		if 'bidir' in self.arch_params['model_arch']:
			arch_names = [self.arch_params['flow_arch'], self.arch_params['color_arch']]

			for ai, ap in enumerate([self.flow_arch_params, self.color_arch_params]):
				if 'cvae' in arch_names[ai]:
					# exp_name += '_cvaeenc{}'.format(self.arch_params['global_transform_enc_params']['nf_enc'])
					# exp_name += '_ncps{}'.format(self.arch_params['global_transform_enc_params']['n_convs_per_stage'])
					if ap['global_transform_enc_params']['fully_conv']:
						exp_name += '_tfclatent{}'.format(ap['global_transform_enc_params']['latent_chans'])
						if ap['condition_on_image'] and not ap['condition_by_concat']:
							exp_name += '_ifclatent{}'.format(ap['img_enc_params']['latent_chans'])
					else:
						exp_name += '_glatent{}'.format(ap['global_transform_latent_dim'])

					if ap['img_enc_params']['include_skips']:
						exp_name += '_imgskips'
					if 'condition_by_concat' in ap.keys() and ap['condition_by_concat']:
						exp_name += '_condcat'
					# exp_name += '_unet{}'.format(self.arch_params['local_transform_enc_params']['nf_enc'])
				# else:
				elif 'voxelmorph' not in arch_names[ai] and 'vm' not in arch_names[ai]:
					if not ap['local_only']:
						# exp_name += '_globalenc{}'.format(self.arch_params['global_transform_enc_params']['nf_enc'])
						# exp_name += '_ncps{}'.format(self.arch_params['global_transform_enc_params']['n_convs_per_stage'])
						exp_name += '_glatent{}'.format(ap['global_transform_latent_dim'])

					if not ap['global_only']:
						if ap['local_transform_enc_params']['fully_conv']:
							exp_name += '_localenc-fc'

						# exp_name += '_imgae{}'.format(self.arch_params['img_enc_params']['nf_enc'])
						if ap['condition_on_class']:
							exp_name += '_condclass'

						# if self.arch_params['img_enc_params']['use_maxpool']:
						#	exp_name += '_maxpool'
						exp_name += '_llatent{}'.format(ap['local_transform_latent_dim'])

			if 'color_transform_in_tgt_space' in self.arch_params.keys() and self.arch_params['color_transform_in_tgt_space']:
				exp_name += '_c-tgtsp'
			else:
				exp_name += '_c-srcsp'
			if self.arch_params['input_segmentations_to_color'] and self.arch_params['input_contours']:
				exp_name += '_incontours'
			elif self.arch_params['input_segmentations_to_color'] and ('input_oh' not in self.arch_params.keys() or not self.arch_params['input_oh']):
				exp_name += '_insegs'
			elif self.arch_params['input_segmentations_to_color']:
				exp_name += '_insegsoh'

		if 'pretrain_flow' in self.arch_params.keys():
			exp_name += '_pt{}'.format(self.arch_params['pretrain_flow'])
		if 'freeze_flow' in self.arch_params.keys() and self.arch_params['freeze_flow']:
			exp_name += '_frzflow'
		# flow smoothness and reconstruction losses
		if self.transform_reg_flow is not None:
			if self.transform_reg_lambda_flow > 10000:
				exp_name += '_{}_regfwt{:.1e}'.format(self.transform_reg_flow,
												  self.transform_reg_lambda_flow)
			else:
				exp_name += '_{}_regfwt{}'.format(self.transform_reg_flow,
												  self.transform_reg_lambda_flow)

		if self.recon_loss_Iw is not None:
			exp_name += '_{}'.format(self.recon_loss_Iw)
			if 'l2' in self.recon_loss_Iw:
				exp_name += '_sigIw{}'.format(self.sigma_Iw)
			elif 'cc' in self.recon_loss_Iw:
				if self.cc_loss_weight > 10000:
					exp_name += '_wt{:.1e}'.format(self.cc_loss_weight)
				else:
					exp_name += '_wt{}'.format(self.cc_loss_weight)
				exp_name += '_win{}'.format(self.cc_win_size_Iw)

		# color smoothness and reconstruction losses
		if self.transform_reg_color is not None:
			if self.transform_reg_lambda_color > 10000:
				exp_name += '_{}_regcwt{:.1e}'.format(self.transform_reg_color,
												  self.transform_reg_lambda_color)
			else:
				exp_name += '_{}_regcwt{}'.format(self.transform_reg_color,
												  self.transform_reg_lambda_color)

		if self.recon_loss_I is not None:
			exp_name += '_{}'.format(self.recon_loss_I)
			if 'l2' in self.recon_loss_I:
				exp_name += '_sigI{}'.format(self.sigma_I)
					
		self.model_name = exp_name

		exp_name = super(ExperimentGlobalLocalTransforms, self).get_model_name()
		self.model_name = exp_name
		return exp_name


	def __init__(self, data_params, arch_params):
		super(ExperimentGlobalLocalTransforms, self).__init__(data_params, arch_params)
		self.data_params = data_params
		self.arch_params = arch_params

		self.do_profile = True
		self.profiled_iters = 0

		self.epoch_count = 0

		if isinstance(data_params['source_name'], list):
			self.source_name = '+'.join(data_params['source_name'])
		else:
			self.source_name = data_params['source_name']

		if isinstance(data_params['target_name'], list):
			self.target_name = '+'.join(data_params['target_name'])
		else:
			self.target_name = data_params['target_name']

		self.img_shape = data_params['img_shape']
		self.n_chans = data_params['img_shape'][-1]
		self.n_dims = len(self.img_shape) - 1

		# name our source domain according to our dataset parameters
		self.logger = None
		import mri_loader
		self.dataset = mri_loader.MRIDataset(self.data_params, self.logger)
		self.source_name = self.dataset.create_display_name()
		self.img_shape = self.dataset.params['img_shape']

		# copy the name over if source and target datasets are identical		
		if data_params['target_name'] == data_params['source_name']:
			self.target_name = self.source_name

		if 'lr' in arch_params.keys():
			self.lr = arch_params['lr']

		if 'input_segmentations_to_color' not in arch_params.keys():
			self.arch_params['input_segmentations_to_color'] = False
		if 'input_oh' not in arch_params.keys():
			self.arch_params['input_oh'] = False
		if 'input_contours' not in arch_params.keys():
			self.arch_params['input_contours'] = False

		# separate flow params from color params
		if 'flow_arch_params' in arch_params.keys():
			self.flow_arch_params = arch_params['flow_arch_params']
		else:
			self.flow_arch_params = self.arch_params

		if 'color_arch_params' in arch_params.keys():
			self.color_arch_params = arch_params['color_arch_params']
		else:
			self.color_arch_params = self.arch_params
		print(self.arch_params)
		print(self.flow_arch_params)
		print(self.color_arch_params)
		#sys.exit()
		print(arch_params.keys())

		# enc/dec architecture
		# parse params for flow portion of network
		self.recon_loss_Iw = self.flow_arch_params['recon_loss_Iw']
		if 'transform_reg_flow' in self.flow_arch_params.keys() and self.flow_arch_params['transform_reg_flow'] is not None:
			self.transform_reg_flow = self.flow_arch_params['transform_reg_flow']
			if 'grad_l2_vm' in self.transform_reg_flow:
				# debugging gradient function, try using voxelmorph's
				self.transform_reg_fn_flow = vm_losses.gradientLoss(penalty='l2')
				self.transform_reg_lambda_flow = self.flow_arch_params['transform_reg_lambda_flow']
				if 'voxelmorph' not in self.flow_arch_params['flow_arch']:
					# a bit hacky, but we only want a sum if we need to compare against a VAE z...
					self.transform_reg_lambda_flow *= np.prod(self.img_shape[:-1]).astype(float) * float(self.n_dims)
			elif 'grad_l2' in self.transform_reg_flow:
				self.transform_reg_fn_flow = my_metrics.gradient_loss_l2(n_dims=self.n_dims)
				self.transform_reg_lambda_flow = self.flow_arch_params['transform_reg_lambda_flow']
				if 'voxelmorph' not in arch_params['flow_arch']:
					# a bit hacky, but we only want a sum if we need to compare against a VAE z...
					self.transform_reg_lambda_flow *= np.prod(self.img_shape[:-1]).astype(float) * float(self.n_dims)
			elif 'prec' in self.transform_reg_flow:
				from metrics import VoxelmorphMetrics
				self.transform_reg_fn_flow = VoxelmorphMetrics(alpha=1.).smoothness_precision_loss_zeromean
				# let's just use lambda_flow to represent alpha
				self.transform_reg_lambda_flow = self.flow_arch_params['transform_reg_lambda_flow'] 
			else:
				self.transform_reg_flow = None
				self.transform_reg_fn_flow = None
				self.transform_reg_lambda_flow = 0.

			if self.recon_loss_Iw is None:  # still have this output node, but don't weight it
				self.reconstruction_loss_fn_flow = keras_metrics.mean_squared_error
				self.reconstruction_loss_weight_flow = 0
			elif 'l2' in self.recon_loss_Iw:
				self.sigma_Iw = self.flow_arch_params['sigma_Iw']
				self.reconstruction_loss_fn_flow = keras_metrics.mean_squared_error

				if not self.sigma_Iw == 'learned':
					# set a constant weight for reconstruction
					self.reconstruction_loss_weight_flow = 0.5 / self.sigma_Iw ** 2 * np.prod(self.img_shape)
			elif 'cc_vm' in self.recon_loss_Iw:
				self.cc_loss_weight = self.flow_arch_params['cc_loss_weight']
				self.cc_win_size_Iw = self.flow_arch_params['cc_win_size_Iw']
#				self.reconstruction_loss_fn_flow = my_metrics.cc2D_loss(self.cc_win_size_Iw, n_chans=self.n_chans)#, n_dims=self.n_dims)
				self.reconstruction_loss_fn_flow = vm_losses.NCC().loss
				self.reconstruction_loss_weight_flow = self.cc_loss_weight
				self.sigma_Iw = None
			elif 'cc' in self.recon_loss_Iw:
				self.cc_loss_weight = self.flow_arch_params['cc_loss_weight']
				self.cc_win_size_Iw = self.flow_arch_params['cc_win_size_Iw']
#				self.reconstruction_loss_fn_flow = my_metrics.cc2D_loss(self.cc_win_size_Iw, n_chans=self.n_chans)#, n_dims=self.n_dims)
				self.reconstruction_loss_fn_flow = my_metrics.ccnD(
					self.cc_win_size_Iw, n_chans=self.n_chans, n_dims=self.n_dims)
				self.reconstruction_loss_weight_flow = self.cc_loss_weight
				self.sigma_Iw = None

		# parse params for color portion of network
		if 'transform_reg_color' in self.color_arch_params.keys() and self.color_arch_params['transform_reg_color'] is not None:
			self.recon_loss_I = self.color_arch_params['recon_loss_I']
			self.transform_reg_color = self.color_arch_params['transform_reg_color']
			if 'grad_l2' in self.transform_reg_color:
				self.transform_reg_fn_color = my_metrics.gradient_loss_l2
				self.transform_reg_lambda_color = self.color_arch_params['transform_reg_lambda_color']
				if 'voxelmorph' not in self.color_arch_params['color_arch']:
					# a bit hacky, but we only want a sum if we need to compare against a VAE z...
					self.transform_reg_lambda_color *= np.prod(self.img_shape)
			elif 'prec' in self.transform_reg_color:
				from metrics import VoxelmorphMetrics
				self.transform_reg_fn_color = VoxelmorphMetrics(alpha=1.).smoothness_precision_loss_zeromean
				# let's just use lambda_color to represent alpha
				self.transform_reg_lambda_color = self.color_arch_params['transform_reg_lambda_color']
			elif 'seg-l2' in self.transform_reg_color:
				self.transform_reg_lambda_color = self.color_arch_params['transform_reg_lambda_color']
				self.transform_reg_fn_color = my_metrics.SpatialSegmentSmoothness(
					n_dims=self.n_dims,
					n_chans=self.n_chans,
				).compute_loss
				if 'voxelmorph' not in self.color_arch_params['color_arch']:
					# a bit hacky, but we only want a sum if we need to compare against a VAE z...
					self.transform_reg_lambda_color *= np.prod(self.img_shape).astype(float)
			elif 'grad-si-l2_l1reg' in self.transform_reg_color:
				# gradient in space and intensity, l2 regularization
				self.transform_reg_fn_color = my_metrics.SummedLosses(
					loss_fns=[
						my_metrics.SpatialIntensitySmoothness(
							n_dims=self.n_dims,
							n_chans=self.n_chans,
							use_true_gradients='predgrad' in self.transform_reg_color,
						).compute_loss, 
						my_metrics.l1_norm],
						loss_weights=arch_params['transform_reg_lambdas_color'],
				).compute_loss

				self.transform_reg_lambda_color = self.color_arch_params['transform_reg_lambda_color']
				if 'voxelmorph' not in self.color_arch_params['color_arch']:
					# a bit hacky, but we only want a sum if we need to compare against a VAE z...
					self.transform_reg_lambda_color *= np.prod(self.img_shape).astype(float)
			elif 'grad-si-l2' in self.transform_reg_color:
				# gradient in space and intensity, l2 regularization
				self.transform_reg_fn_color = my_metrics.SpatialIntensitySmoothness(
					n_dims=self.n_dims,
					n_chans=self.n_chans,
					use_true_gradients='predgrad' in self.transform_reg_color,
				).compute_loss

				self.transform_reg_lambda_color = self.color_arch_params['transform_reg_lambda_color']
				if 'voxelmorph' not in self.color_arch_params['color_arch']:
					# a bit hacky, but we only want a sum if we need to compare against a VAE z...
					self.transform_reg_lambda_color *= np.prod(self.img_shape)
			else:
				self.transform_reg_color = None
				self.transform_reg_fn_color = None
				self.transform_reg_lambda_color = 0.

			if self.recon_loss_I is None:  # still have this output node, but don't weight it
				self.reconstruction_loss_fn_color = keras_metrics.mean_squared_error
				self.reconstruction_loss_weight_color = 0
			elif 'l2' in self.recon_loss_I:
				self.sigma_I = self.color_arch_params['sigma_I']
				self.reconstruction_loss_fn_color = keras_metrics.mean_squared_error

				if not self.sigma_I == 'learned':
					# set a constant weight for reconstruction
					self.reconstruction_loss_weight_color = 0.5 / self.sigma_I ** 2
					if 'voxelmorph' not in self.color_arch_params['color_arch']:
						# a bit hacky, but we only want a sum if we need to compare against a VAE z...
						self.reconstruction_loss_weight_color *= np.prod(self.img_shape)

			if self.arch_params['pretrain_flow'] > 0:
				self.started_seq_training = False
				self.color_lambda = 0
		else:
			# no color transform
			self.arch_params['pretrain_flow'] = 0
			self.transform_reg_color = None
			self.recon_loss_I = None

		if 'latest_epoch' in arch_params.keys():
			self.latest_epoch = arch_params['latest_epoch']
		else:
			self.latest_epoch = 0


	def compile_models(self, run_options=None, run_metadata=None):
		if 'color' in self.arch_params['model_arch']:  # if we have a color transform, we might need to update some losses
			if 'grad-si-l2_l1reg' in self.transform_reg_color:  # do this here since we need to point to the model
				self.transform_reg_fn_color = my_metrics.SummedLosses(
					loss_fns=[
						my_metrics.SpatialIntensitySmoothness(
							n_dims=self.n_dims,
							n_chans=self.n_chans,
							use_true_gradients='predgrad' in self.transform_reg_color,
							pred_image_output=self.trainer_model.get_layer('spatial_transformer').output
						).compute_loss,
						my_metrics.l1_norm,
					], loss_weights=[1, 1]).compute_loss
			elif 'seg-l2' in self.transform_reg_color and self.started_seq_training:  # otherwise the layer will not exist yet
				self.transform_reg_fn_color = my_metrics.SpatialSegmentSmoothness(
					n_dims=self.n_dims,
					n_chans=self.n_chans,
					warped_contours_layer_output=self.trainer_model.get_layer('warped_labels').output
				).compute_loss

			elif 'grad-si-l2' in self.transform_reg_color:  # do this here since we need to point to the model
				self.transform_reg_fn_color = my_metrics.SpatialIntensitySmoothness(
					n_dims=self.n_dims,
					n_chans=self.n_chans,
					use_true_gradients='predgrad' in self.transform_reg_color,
					pred_image_output=self.trainer_model.get_layer('spatial_transformer').output
				).compute_loss
		
			# we actually want to compare the color delta (src space) to the inv-warped target (src space)
			if 'bidir' in self.arch_params['model_arch'] \
					and not self.arch_params['color_transform_in_tgt_space'] \
					and self.arch_params['recon_loss_I']=='l2_src':  # recon loss in src space
				self.reconstruction_loss_fn_color = my_metrics.BoundLoss(
					y_true=self.trainer_model.get_layer('color_transformed_tgt').output,
					loss_fn=keras_metrics.mean_squared_error).compute_loss

		if 'flow' in self.arch_params['model_arch'] \
				and 'color' in self.arch_params['model_arch'] \
				and self.started_seq_training:
			self.loss_names, loss_fns, loss_weights = self.vae.get_losses(
				transform_reg_fn_flow=self.transform_reg_fn_flow, 
				transform_reg_lambda_flow=self.transform_reg_lambda_flow, 
				transform_reg_name_flow=self.arch_params['transform_reg_flow'], 
				recon_loss_fn_flow=self.reconstruction_loss_fn_flow, 
				recon_loss_weight_flow=self.reconstruction_loss_weight_flow, 
				recon_loss_name_flow=self.arch_params['recon_loss_Iw'], 
				transform_reg_fn_color=self.transform_reg_fn_color, 
				transform_reg_lambda_color=self.transform_reg_lambda_color, 
				transform_reg_name_color=self.arch_params['transform_reg_color'], 
				recon_loss_fn_color=self.reconstruction_loss_fn_color, 
				recon_loss_weight_color=self.reconstruction_loss_weight_color, 
				recon_loss_name_color=self.arch_params['recon_loss_I'], 
			)
		elif 'voxelmorph' in self.arch_params['flow_arch'] \
				and (not self.started_seq_training or not 'color' in self.arch_params['model_arch']): # flow-only unet:
			self.loss_names = ['total', self.arch_params['transform_reg_flow'], self.arch_params['recon_loss_Iw']]
			loss_weights = [self.transform_reg_lambda_flow, self.reconstruction_loss_weight_flow]
			loss_fns = [self.transform_reg_fn_flow, self.reconstruction_loss_fn_flow]
		else:
			print(self.vae)
			# flow only
			self.loss_names, loss_fns, loss_weights = self.vae.get_losses(
				transform_reg_fn=self.transform_reg_fn_flow, 
				transform_reg_lambda=self.transform_reg_lambda_flow,
				transform_reg_name=self.arch_params['transform_reg_flow'], 
				recon_loss_fn=self.reconstruction_loss_fn_flow, 
				recon_loss_weight=self.reconstruction_loss_weight_flow,
				recon_loss_name=self.arch_params['recon_loss_Iw'], 
			)

		self.logger.debug('Trainer model')
		self.trainer_model.summary(print_fn=self.logger.debug, line_length=120)

		self.logger.debug('Compiling full VTE model with {} losses: {}'.format(len(loss_fns), self.loss_names))
		for li, lf in enumerate(loss_fns):
			self.logger.debug('Model output: {}, loss fn: {}'.format(
				self.trainer_model.outputs[li],
				lf))
		self.logger.debug('and {} weights {}'.format(len(loss_weights), loss_weights))
		print([type(w) for w in loss_weights])
		if run_options is not None:
			self.trainer_model.compile(loss=loss_fns, loss_weights=loss_weights,
		                           optimizer=Adam(lr=self.lr), 
									options=run_options, run_metadata=run_metadata,
			)

		else:
			self.trainer_model.compile(loss=loss_fns, loss_weights=loss_weights,
		                           optimizer=Adam(lr=self.lr))

		self.arch_params['loss_weights'] = loss_weights
		self.arch_params['loss_fns'] = [lf.__name__ for lf in loss_fns]

		with open(os.path.join(self.exp_dir, 'arch_params.json'), 'w') as f:
			json.dump(self.arch_params, f)
		with open( os.path.join( self.exp_dir, 'data_params.json'), 'w') as f:
			json.dump( self.data_params, f)


	def load_data( self, load_fewer = False ):
		self.dataset.logger = self.logger
		self.dataset.profiler_logger = self.profiler_logger

		# by default, adni dataset loads ims as X and segs as Y
		(self.X_source_train, self.Y_source_train, self.source_train_files), \
		(self.X_target_train, self.Y_target_train, self.target_train_files), \
		(self.X_source_test, self.Y_source_test, self.source_test_files), \
		(self.X_target_test, self.Y_target_test, self.target_test_files), self.label_mapping \
			= self.dataset.load_source_target(debug=load_fewer,
					load_source_segs=(
						self.arch_params['input_segmentations_to_color'] and not self.arch_params['input_contours']))
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
		self._save_dataset_info()


	def _save_dataset_info( self ):
		with open( os.path.join( self.exp_dir, 'source_train_files.txt'), 'w') as f:
			f.writelines( [s + '\n' for s in self.source_train_files] )
		with open( os.path.join( self.exp_dir, 'target_train_files.txt'), 'w') as f:
			f.writelines( [s + '\n' for s in self.target_train_files] )
		with open( os.path.join( self.exp_dir, 'source_test_files.txt'), 'w') as f:
			f.writelines( [s + '\n' for s in self.source_test_files] )
		with open( os.path.join( self.exp_dir, 'target_test_files.txt'), 'w') as f:
			f.writelines( [s + '\n' for s in self.target_test_files] )


	def save_models(self, models_dir, epoch, iter_count=None):
		super(ExperimentGlobalLocalTransforms, self).save_models(models_dir, epoch, iter_count=iter_count)


	def _create_flow_model(self):
		# parse the flow architecture name to create the correct model
		if 'voxelmorph2guha' in self.arch_params['flow_arch']:
			self.unet_flow = my_voxelmorph_networks.voxelmorph_wrapper(
				img_shape=self.img_shape,
				voxelmorph_arch='vm2_guha'
			)
			self.models += [self.unet_flow]
			self.flow_models = [self.unet_flow]
		elif 'bidir' in self.arch_params['flow_arch']:
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
			self.models += [self.flow_fwd_model, self.flow_bck_model]
			self.flow_models.append(self.flow_fwd_model)


	def _create_color_model(self):
		# parse the color architecture name to create the correct model
		if 'unet' in self.arch_params['color_arch']:
			color_model_name = 'color_delta_unet'
			if 'bidir' in self.arch_params['model_arch']:
				if 'color_transform_in_tgt_space' in self.color_arch_params.keys() and self.color_arch_params[
					'color_transform_in_tgt_space']:
					color_model_name += '_tgtspace'
				else:
					color_model_name += '_srcspace'

			self.unet_color = transform_network_utils.color_delta_unet_model(
				img_shape=self.img_shape,
				n_output_chans=self.n_chans,
				enc_params=self.color_arch_params['local_transform_enc_params'],
				model_name=color_model_name,
				input_segmentations=self.color_arch_params['input_segmentations_to_color'],
				labels_shape=self.input_labels_shape
			)


	def create_models(self):
		self.models = []
		if self.arch_params['input_oh'] and self.arch_params['warp_labels']: # TODO: rename this param, this is horrible
			self.input_labels_shape = tuple(self.Y_source_train.shape[1:-1]) + (self.n_labels + 1,) 
		elif not self.arch_params['input_oh'] and self.arch_params['warp_labels'] and self.arch_params['input_segmentations_to_color'] and not self.arch_params['input_contours']:
			self.input_labels_shape = tuple(self.Y_source_train.shape[1:-1]) + (2,) 
		elif self.arch_params['warp_labels'] or self.arch_params['input_contours']:
			self.input_labels_shape = self.Y_source_train.shape[1:]
		else:
			self.input_labels_shape = None
		self.logger.debug('Auxiliary input shape: {}'.format(self.input_labels_shape))
	
		self._create_flow_model()  # creates either self.unet_flow (a Model) or self.vae_flow (a CVAEClass)
		if self.arch_params['color_arch'] is not None:
			self._create_color_model() # creates either self.unet_color (a Model) or self.vae_color (a CVAEClass)
		else:
			# this is actually a bit nonsensical since there is no seq training in flow-only
			# but we do this so that other things dont break
			self.started_seq_training = True


		# make sure we aren't training the flow model further
		if self.started_seq_training \
				and 'freeze_flow' in self.arch_params.keys() \
				and self.arch_params['freeze_flow']:
			for fm in self.flow_models:
				fm.trainable = False
				for l in fm.layers:
					l.trainable = False

		# voxelmorph all the waaaay
		if 'voxelmorph' in self.arch_params['flow_arch'] \
				and 'voxelmorph' in self.arch_params['color_arch']:

			if 'flow' in self.arch_params['model_arch'] and 'color' in self.arch_params['model_arch'] \
					and self.started_seq_training:
				# special bidirectional model
				if 'bidir' in self.arch_params['model_arch']:
					self.vae = SequentialTransformEncoders.SequentialUnetsBidir(
						img_shape=self.img_shape,
						flow_fwd_model=self.flow_fwd_model,
						unet_flow_bck=self.flow_bck_model,
						color_model=self.unet_color,
						color_transform_type='B_RGB',  # we don't support other color types right now
						color_lambda=self.color_lambda,
						do_warp_labels=self.arch_params['warp_labels'],
						predict_color_in_tgt_space=self.arch_params['color_transform_in_tgt_space'],
						color_loss_in_tgt_space='src' not in self.arch_params['recon_loss_I'],
						labels_shape=self.input_labels_shape
					)
				else:
					self.vae = SequentialTransformEncoders.SequentialUnets(
						img_shape=self.img_shape,
						unet_flow=self.unet_flow,
						unet_color=self.unet_color,
						color_transform_type='B_RGB',  # we don't support other color types right now
						color_lambda=self.color_lambda,
						do_warp_labels=self.arch_params['warp_labels'],
					)
				self.vae.create_modules()
				self.vae.create_train_wrapper()

				self.trainer_model = self.vae.trainer_model
				self.tester_model = self.vae.tester_model

				self.models = self.vae.get_models()
			else:
				if self.arch_params['pretrain_flow'] > 0:
					self.unet_color.trainable = False

				#self.models = [self.unet_flow]
				if 'bidir' in self.arch_params['model_arch']:
					# we shouldn't ever really use this trainer model since we don't train bidir in here
					self.trainer_model = self.flow_fwd_model
					self.tester_model = self.flow_fwd_model
				else:
					self.trainer_model = self.unet_flow
					self.tester_model = self.unet_flow
		else:
			# unet for flow
			if 'voxelmorph' in self.arch_params['flow_arch']:
				self.vae_flow = self.unet_flow  # just copy over the variable name

			# only training flow
			self.trainer_model = self.vae_flow.trainer_model
			self.tester_model = self.vae_flow.tester_model

			self.models = self.vae_flow.get_models()
		super(ExperimentGlobalLocalTransforms, self).create_models()
		return self.models


	def load_models(self, models_dir, load_epoch=None, stop_on_missing=True):
		if load_epoch == 'latest':
			self.logger.debug('Looking for epoch {} in {}'.format(load_epoch, models_dir))
			load_epoch = file_utils.get_latest_epoch_in_dir(models_dir)

		if load_epoch is None:
			load_epoch = 0
		else:
			load_epoch = int(load_epoch)

		if 'pretrain_flow' in self.arch_params.keys() and load_epoch >= self.arch_params['pretrain_flow']:
			# make sure we are using the correct models for this point in training
			self._start_sequential_training(load_epoch)
			if load_epoch == self.arch_params['pretrain_flow']:
				stop_on_missing = False  # when we first switch to flow+color, we will not have a color model to load
		start_epoch = super(ExperimentGlobalLocalTransforms, self).load_models(models_dir, load_epoch,
			stop_on_missing=stop_on_missing)
		return start_epoch


	# TODO: streamline generators since we no longer need to get same-class pairs
	def create_generators(self, batch_size):
		self.batch_size = batch_size

		if 'adni' in self.data_params['dataset_name'] and not self.data_params['load_vols']:
			# we don't preload vols into memory, so we need to load and then generate pairs
			self.pair_train_gen = transform_utils.gen_paired_batch_from_gens(
				source_gen=self.dataset.gen_vols_batch(
					['labeled_train'], load_segs=False,
				),
				target_gen=self.dataset.gen_vols_batch(
					['unlabeled_train'], load_segs=False,
				),
				batch_size=batch_size,
				same_label=False,  # all the brains are teh "same label"
				return_labels=True,  # we don't use these, but we do this to make the return count consistent
			)
			self.pair_test_gen = transform_utils.gen_paired_batch_from_gens(
				source_gen=self.dataset.gen_vols_batch(
					['labeled_train'], load_segs=False,
				),
				target_gen=self.dataset.gen_vols_batch(
					['labeled_valid'], load_segs=False,
				),
				batch_size=batch_size,
				same_label=False,  # all the brains are teh "same label"
				return_labels=True,  # we don't use these, but we do this to make the return count consistent
			)

		# returns inputs, targets, and labels
		self.train_gen_verbose = self._generate_vae_batch(self.batch_size, self.pair_train_gen)

		# returns only (inputs, targets) for keras fit_generator
		self.train_gen = self._generate_inputs_targets(self.train_gen_verbose, mode='train')
		
		self.test_gen_verbose = self._generate_vae_batch(self.batch_size, self.pair_test_gen, mode='test')
		self.valid_gen = self._generate_inputs_targets(self.test_gen_verbose, mode='valid')

		self.X_test, self.I_oh_test, _ = next(self.pair_test_gen)


		if 'canon' in self.source_name:
			# the autoencoder should get to see at least one example of each digit
			self.autoencoder_gen = data_utils.gen_batch(
				np.concatenate([self.X_source_train, self.X_source_test], axis=0),
				np.concatenate([self.Y_source_train, self.Y_source_test], axis=0),
				batch_size=self.batch_size
			)
		else:
			self.autoencoder_gen = data_utils.gen_batch(
				self.X_source_train,
				self.Y_source_train,
				batch_size=self.batch_size
			)
		# make a new generator just for sampling, in case we are using fit_generator and it is already
		# using the other test_gen
		self.sample_tgt_valid_gen = self.dataset.gen_vols_batch(dataset_splits=['labeled_valid'],
											  batch_size=batch_size, load_segs=False, randomize=True)

		self.sample_tgt_train_gen = self.dataset.gen_vols_batch(dataset_splits=['unlabeled_train', 'labeled_train'],
											  batch_size=batch_size, load_segs=False, randomize=True)

		self.sample_tgt_train_valid_gen = self.dataset.gen_vols_batch(dataset_splits=['unlabeled_train', 'labeled_train', 'labeled_valid'],
											  batch_size=batch_size, load_segs=False, randomize=True)

		# create source aux input here in case we need it later
		if self.arch_params['input_segmentations_to_color']:
			if self.arch_params['input_oh']:
				# first channel will be segs in label form
				Y_source_onehot = classification_utils.labels_to_onehot(
					self.Y_source_train[..., [0]], label_mapping=self.label_mapping)

				# a little unintuitive, but in this case the last channel of Y_source_train will be contours
				Y_source_contours = self.Y_source_train[..., [-1]]	
				self.source_aux_inputs = np.concatenate([Y_source_onehot, Y_source_contours], axis=-1)
			else:
				self.source_aux_inputs = self.Y_source_train


	def _generate_vae_batch(self, batch_size, gen=None, mode='train'):
		# TODO: this only works if we have only one source example
		if self.arch_params['input_oh']:
			src_oh = classification_utils.labels_to_onehot(self.Y_source_train[..., 0], label_mapping=self.label_mapping)
			print(src_oh.shape)
			print(self.Y_source_train.shape)
			src_labels = np.concatenate([src_oh, self.Y_source_train[..., [-1]]], axis=-1)
		else:
			src_labels = self.Y_source_train

		while True:
			start = time.time()
			X_srctgt_stack, _, tgt_labels = next(gen)
			if self.do_profile:
				self.profiler_logger.info('Generating pair took {}'.format(time.time() - start))

			if not isinstance(X_srctgt_stack, list):
				# if we stacked them
				I = np.take(X_srctgt_stack, range(self.n_chans), axis=-1).copy()
				J = np.take(X_srctgt_stack, range(self.n_chans, 2 * self.n_chans), axis=-1).copy()
			else:
				I = X_srctgt_stack[0]
				J = X_srctgt_stack[1]

			# save thsee so we can use them to make images
			if mode == 'train':
				self.I_train = I
				self.J_train = J
			else:
				self.I_test = I
				self.J_test = J

			if ('vm' in self.arch_params['model_arch']  or 'voxelmorph' in self.arch_params['model_arch']) \
					and self.started_seq_training == False:
				train_targets = [J, J]
			else:
				train_targets = self.vae.get_train_targets(I, J, batch_size)

			if self.arch_params['condition_on_class'] or (self.arch_params['warp_labels'] and self.started_seq_training):
				yield [I, J, src_labels], train_targets, src_labels, tgt_labels
			else:
				yield [I, J], train_targets, src_labels, tgt_labels


	def _generate_inputs_targets(self, gen, mode='train'):
		while True:
			inputs, targets, I_oh, J_oh = next(gen)
			if mode=='train':
				self.IJ_train_batch = inputs
				self.I_train_oh = I_oh
				self.J_train_oh = J_oh
			else:
				self.IJ_test_batch = inputs
				self.I_test_oh = I_oh
				self.J_test_oh = J_oh

			yield (inputs, targets)

	def make_train_results_im(self):
		preds = self.trainer_model.predict(self.IJ_train_batch)
		if 'adni' not in self.source_name:  # hacky, but we don't load the labels for adni
			I_labels = classification_utils.onehot_to_labels(self.I_train_oh, label_mapping=self.label_mapping)
			J_labels = classification_utils.onehot_to_labels(self.J_train_oh, label_mapping=self.label_mapping)
		else:
			I_labels = []
			J_labels = []
		# TODO: put logic of order of outputs in model class...
		ims = [self.IJ_train_batch[0], self.IJ_train_batch[1]]
		labels = [I_labels, J_labels]
		do_normalize = [False, False]
		
		if self.arch_params['input_segmentations_to_color']and self.arch_params['input_oh']:
			ims += [self.IJ_train_batch[-1][..., 16]]
			labels += ['aux_oh']
			do_normalize += [True]


		if 'bidir' in self.arch_params['flow_arch'] \
				and self.started_seq_training:
			
			if self.arch_params['warp_labels']:
				ims += preds[:5] + [preds[-1]]
				labels += ['pred_flow', 'warped', 'warped_mask', 'pred_color_delta', 'pred_color_transformed', 'pred_out']
				do_normalize += [False, False, True, True, False, False]
			else:
				ims += preds[:4] + [preds[-1]]
				labels += ['pred_flow', 'warped', 'pred_color_delta', 'pred_color_transformed', 'pred_out']
				do_normalize += [False, False, True, True, False]
	
		elif 'voxelmorph' in self.arch_params['flow_arch'] and 'voxelmorph' in self.arch_params['color_arch'] \
				and 'flow' in self.arch_params['model_arch'] and 'color' in self.arch_params['model_arch'] \
				and self.started_seq_training:

			ims += preds
			if self.arch_params['warp_labels'] and not 'bidir' in self.arch_params['model_arch']:
				labels += ['pred_flow', 'warped', 'warped_mask', 'pred_color', 'pred_out']
				do_normalize += [False, False, True, True, False]
			elif self.arch_params['warp_labels'] and 'bidir' in self.arch_params['model_arch']:
				labels += ['pred_flow', 'warped', 'warped_mask', 'pred_color_delta', 'pred_color_transformed', 'pred_out']
				do_normalize += [False, False, True, True, False, False]
			else:
				labels += ['pred_flow', 'warped', 'pred_color', 'pred_out']
				do_normalize += [False, False, True, False]

		elif 'voxelmorph' in self.arch_params['flow_arch'] and 'voxelmorph' in self.arch_params['color_arch'] \
				and 'flow' in self.arch_params['model_arch'] and 'color' in self.arch_params['model_arch'] \
				and 'bidir' not in self.arch_params['flow_arch'] \
				and not self.started_seq_training:
			ims += preds
			labels += ['pred_flow', 'warped']
			do_normalize += [False, False]
		elif 'voxelmorph' in self.arch_params['flow_arch'] and 'voxelmorph' in self.arch_params['color_arch'] \
				and 'flow' in self.arch_params['model_arch'] and 'color' in self.arch_params['model_arch'] \
				and 'bidir' in self.arch_params['flow_arch'] \
				and not self.started_seq_training:
			ims += preds
			labels += ['warped', 'warped_back', 'pred_flow', 'pred_flow_back']
			do_normalize += [False, False, False, False]
		elif 'voxelmorph' in self.arch_params['flow_arch'] and not self.started_seq_training:
			flow_pred = preds[0]
			transformed_pred = preds[1]
			ims += [flow_pred, transformed_pred]
			labels += ['pred_flow', 'pred_out']
			do_normalize += [False, False]

		return self._make_results_im(ims, labels, do_normalize)

	#def _generate_samples_from_prior(tgt_mode='valid')

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

		zs = []
		if 'bidir' in self.arch_params['model_arch']:
			flow_tgt, _ = next(sample_from_gen)
			zs += [flow_tgt]

			X_colortgt, _ = next(sample_from_gen)
			zs += [X_colortgt]

			# a bit hacky, but we need to give the source segs
			if self.arch_params['input_segmentations_to_color']:
				if labels_onehot is None:
					aux_input = self.source_aux_inputs
				else:
					aux_input = labels_onehot
				if verbose:
					self.logger.debug('Adding aux labels as input of shape {}'.format(aux_input.shape))
				zs += [aux_input]
			if verbose:
				self.logger.debug('Sampled prior shapes: {}'.format([z.shape for z in zs]))		
		
			return zs

		if 'voxelmorph' in self.arch_params['flow_arch'] and 'voxelmorph' in self.arch_params['color_arch']:
			# if none of our models are VAEs, just return the target image
			tgt, _ = next(sample_from_gen)
			return [tgt]

		if 'flow' in self.arch_params['model_arch'] and 'color' in self.arch_params['model_arch'] and self.started_seq_training \
				and 'voxelmorph' not in self.arch_params['model_arch']:
			return [np.random.randn(batch_size, self.global_transform_latent_dim), 
				np.random.randn(batch_size, self.global_transform_latent_dim)]


	def make_test_results_im(self, epoch_num=None):
		I = self.IJ_test_batch[0]
		if 'adni' not in self.source_name:
			I_labels = classification_utils.onehot_to_labels(self.I_test_oh, label_mapping=self.label_mapping)
			J_labels = classification_utils.onehot_to_labels(self.J_test_oh, label_mapping=self.label_mapping)
		else:
			I_labels = []
			J_labels = []

		if self.X_source_test.shape[0] == 1:
			sampled_zs = self._sample_from_prior(I=None, labels_onehot=None)
		else:
			sampled_zs = self._sample_from_prior(I, labels_onehot=self.I_test_oh)

		preds = self.tester_model.predict([I] + sampled_zs)
		ims = [I]
		labels = [I_labels]
		do_normalize = [False]

		if 'flow' in self.arch_params['model_arch'] and 'color' in self.arch_params['model_arch'] \
				and self.started_seq_training:
			flow_pred = preds[0]
			warped_pred = preds[1]
			color_pred = preds[2]
			transformed_pred = preds[-1]
			ims += [flow_pred, warped_pred, color_pred, transformed_pred]
			labels += ['pred_flow', 'pred_warped', 'pred_color', 'pred_out']
			do_normalize += [False, False, True, False]
		else:
			flow_pred = preds[0]
			transformed_pred = preds[1]
			recon_pred = preds[-1]
			ims += [flow_pred, transformed_pred]
			labels += ['pred_flow', 'pred_out']
			do_normalize += [False, False]
		return self._make_results_im(ims, labels, do_normalize) 


	def _make_eval_gen(self, batch_size=1):
		pair_eval_gen = transform_utils.gen_stackedpair_batch(
			self.X_source_train, self.Y_source_train,
			self.X_target_train, self.Y_target_train,
			batch_size=batch_size, same_label=True,
			exclude_identity=False,
			aug_source_params=None,
			aug_target_params=None,
			return_labels=True,
			gen_target_from_source=False)
		self.eval_gen = self._generate_vae_batch(batch_size, pair_eval_gen)


	def eval(self):
		return 0


	def get_n_train(self):
		if 'adni' in self.data_params['dataset_name'] and not self.data_params['load_vols']:
			# a bit hacky, but we don't load all the volx in this case
			return min(100, len(self.dataset.files_unlabeled_train))
		else:
			return min(100, max(self.X_source_train.shape[0], self.X_target_train.shape[0]))


	def get_n_test(self):
		if 'adni' in self.data_params['dataset_name'] and not self.data_params['load_vols']:
			# a bit hacky, but we don't load all the volx in this case
			return min(100, len(self.dataset.files_labeled_valid))
		else:
			return min(100, max(self.X_source_test.shape[0], self.X_target_test.shape[0]))


	def train_discriminator(self):
		return [], []


	def save_exp_info(self, exp_dir, figures_dir, logs_dir, models_dir):
		super(ExperimentGlobalLocalTransforms, self).save_exp_info(exp_dir, figures_dir, logs_dir, models_dir)


	def _start_sequential_training(self, epoch):
		start = time.time()
		if epoch >= self.arch_params['pretrain_flow'] and self.arch_params['pretrain_flow'] > 0 \
				and self.started_seq_training == False:
			# make sure we freeze the flow models
			if 'freeze_flow' in self.arch_params.keys() and self.arch_params['freeze_flow']:
				for fm in self.flow_models:
					fm.trainable = False
					for l in fm.layers:
						l.trainable = False

			if 'bidir' in self.arch_params['flow_arch']:# and not 'voxelmorph' in self.arch_params['color_arch']:
				# bidirectional flow with cvae color
				#self.save_models(self.models_dir, epoch - 1)  # we never use these models anyway
				self.color_lambda = 1.
				self.started_seq_training = True

				# do this to stop fit_generator so that we can start it again with the new trainer
				self.trainer_model.stop_training = True

				color_model = self.unet_color
				if self.arch_params['pretrain_flow'] > 0:
					color_model.trainable = True
					for l in color_model.layers:
						l.trainable = True


				self.vae = SequentialTransformEncoders.SequentialUnetsBidir(
					img_shape=self.img_shape,
					flow_fwd_model=self.flow_fwd_model,
					unet_flow_bck=self.flow_bck_model,
					color_model=color_model,
					color_transform_type='B_RGB',  # we don't support other color types right now
					color_lambda=self.color_lambda,
					do_warp_labels=self.arch_params['warp_labels'],
					predict_color_in_tgt_space=self.arch_params['color_transform_in_tgt_space'],
					color_loss_in_tgt_space='src' not in self.arch_params['recon_loss_I'],
					input_segmentations_to_color=self.arch_params['input_segmentations_to_color'],
					labels_shape=self.input_labels_shape,
				)

				self.vae.create_modules()
				self.vae.create_train_wrapper()

				self.trainer_model = self.vae.trainer_model
				self.tester_model = self.vae.tester_model
				self.logger.debug('Switching trainer model to SequentialBidir')
				self.trainer_model.summary(print_fn=self.logger.debug, line_length=120)

				self.models = self.vae.get_models()
			else:
			#elif 'bidir' in self.arch_params['flow_arch'] or 'voxelmorph' in self.arch_params['color_arch']:
				# TODO: check this if statement?

				#self.save_models(self.models_dir, epoch - 1)
				self.color_lambda = 1.
				self.started_seq_training = True

				if self.arch_params['pretrain_flow'] > 0:
					self.unet_color.trainable = True

				# do this to stop fit_generator so that we can start it again with the new trainer
				self.trainer_model.stop_training = True

				if 'bidir' in self.arch_params['model_arch']:
					self.vae = SequentialTransformEncoders.SequentialUnetsBidir(
						img_shape=self.img_shape,
						flow_fwd_model=self.flow_fwd_model,
						unet_flow_bck=self.flow_bck_model,
						color_model=self.unet_color,
						color_transform_type='B_RGB',  # we don't support other color types right now
						color_lambda=self.color_lambda,
						do_warp_labels=self.arch_params['warp_labels'],
						predict_color_in_tgt_space=self.arch_params['color_transform_in_tgt_space'],
						color_loss_in_tgt_space='src' not in self.arch_params['recon_loss_I'],
						input_segmentations_to_color=self.arch_params['input_segmentations_to_color'],
						labels_shape=self.input_labels_shape,
					)
					self.logger.debug('Switching trainer model to SequentialBidir')
				else:
					self.vae = SequentialTransformEncoders.SequentialUnets(
						img_shape=self.img_shape,
						unet_flow=self.unet_flow,
						unet_color=self.unet_color,
						color_transform_type='B_RGB',  # we don't support other color types right now
						color_lambda=self.color_lambda,
						do_warp_labels=self.arch_params['warp_labels']
					)
					self.logger.debug('Switching trainer model to SequentialUnets')

				self.vae.create_modules()
				self.vae.create_train_wrapper()

				self.trainer_model = self.vae.trainer_model
				self.tester_model = self.vae.tester_model
				self.trainer_model.summary(print_fn=self.logger.debug, line_length=120)

				self.models = self.vae.get_models()
	
			self.started_seq_training = True
			self.logger.debug('Starting sequential training, epoch {}'.format(epoch))

			self.compile_models()
			super(ExperimentGlobalLocalTransforms, self)._print_models()

		if self.do_profile:
			self.profiler_logger.info('Updating epoch count took {}'.format(time.time() - start))


	def update_epoch_count(self, epoch):
		self.epoch_count += 1
		self._start_sequential_training(epoch)
		return 0


	def train_joint(self):
		X, Y, X_oh, Y_oh = next(self.train_gen_verbose)
		self.IJ_train_batch = X
		self.I_train_oh = X_oh
		self.J_train_oh = Y_oh
		start = time.time()
		loss_vals = self.trainer_model.train_on_batch(
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
			X, Y, X_oh, Y_oh = next(self.test_gen_verbose)
			self.IJ_test_batch = X
			self.I_test_oh = X_oh
			self.J_test_oh = Y_oh

			loss_names = ['test_' + ln for ln in self.loss_names]
			test_loss = np.asarray(
				self.trainer_model.evaluate(
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
