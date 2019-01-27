import tensorflow as tf

import basic_networks
import numpy as np
import keras
from keras.layers import Conv2D, Conv3D
from keras.layers import Input, Concatenate, Dense, Flatten, Lambda, Reshape
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
import brainstorm_networks

import sys
sys.path.append('../neuron')
import neuron.layers as nrn_layers
'''''''''''''''''''''
VTE transform encoder
	- takes a stacked pair I+J and encodes it into z
'''''''''''''''''''''
def VM_VAE_model( img_shape, latent_dim = 10, 
		model_name = 'VM_encoder',
		enc_params = None ): 
	
	x_src = Input(img_shape, name='input_src')
	x_tgt = Input(img_shape, name='input_tgt')
	x_stacked = Concatenate(axis=-1)([x_src, x_tgt])

	'''	
	x_transform_enc = basic_networks.encoder( 
			x = x_stacked, 
			img_shape = img_shape[:-1] + (img_shape[-1]*2,),
			conv_chans = enc_params['enc_chans'],
			min_h = None, min_c = None,
			n_convs_per_stage = enc_params['n_convs_per_stage'] )
	x_enc = LeakyReLU(0.2)(x_transform_enc)
	if not enc_params['fully_conv']:
		x_enc = Flatten()(x_transform_enc)
		preflatten_shape = basic_networks.get_encoded_shape(img_shape=img_shape, conv_chans=enc_params['enc_chans'])
		x_enc = Dense(latent_dim, name='autoencoder_dense')(x_enc)

	# decode into channels that we can then turn into mean and var
	transform_shape = img_shape[:-1] + (2*2,)
	x_flow = basic_networks.decoder(x_enc,
	                                transform_shape,
	                                prefix='flow_dec',
	                                include_dropout=False,
	                                conv_chans=list(reversed(enc_params['enc_chans'])),
	                                n_convs_per_stage=enc_params['n_convs_per_stage'])
	'''
	x_flow = basic_networks.unet2D(x_stacked, img_shape[:-1] + (img_shape[-1] * 2,), out_im_chans=4, nf_enc=enc_params['enc_chans'])

	x_flow_mean = Conv2D(2, 
			kernel_size=3, padding='same', 
			kernel_initializer=keras.initializers.RandomNormal(mean=0.0,stddev=1e-5), name='vae_flow_mean')(x_flow)
	x_flow_logvar = Conv2D(2, kernel_size=3, padding='same', 
			kernel_initializer=keras.initializers.RandomNormal(mean=0.0,stddev=1e-10),
			bias_initializer=keras.initializers.Constant(value=-10),
			name='vae_flow_logvar')(x_flow)
	x_flow_params = Concatenate()([x_flow_mean, x_flow_logvar])
	x_flow = Lambda(brainstorm_networks.sampling, name='Lambda_sample', output_shape=img_shape[:-1] + (2,))([x_flow_mean, x_flow_logvar])

	# apply flow transform
	warped_out = DenseSpatialTransformer(name='spatial_transformer')([x_src, x_flow])
	return Model( inputs=[x_src, x_tgt], outputs = [x_flow_params, x_flow, warped_out], name=model_name )

def bidir_wrapper(img_shape, fwd_model, bck_model, model_name='bidir_wrapper'):
	input_src = Input(img_shape)
	input_tgt = Input(img_shape)

	fwd_model.name = 'vm2_cc_fwd'
	bck_model.name = 'vm2_cc_bck'
	transformed_fwd, flow_fwd = fwd_model([input_src, input_tgt])
	transformed_bck, flow_bck = bck_model([input_tgt, input_src])
	return Model(inputs=[input_src, input_tgt], outputs=[transformed_fwd, transformed_bck, flow_fwd, flow_bck],
		name=model_name)


def voxelmorph_wrapper(img_shape, voxelmorph_arch='vm2_guha'):
	# just reverse the output order to be consistent with ours
	from keras.models import Model
	from keras.layers import Input, Lambda, Reshape

	sys.path.append('../voxelmorph')
	import src.networks as vm_networks

	sys.path.append('../voxelmorph-sandbox')
	import voxelmorph.networks as vms_networks

	if 'diffeo' in voxelmorph_arch:
		nf_dec = [32, 32, 32, 32, 16, 3]

		vm_model = vms_networks.vmnet(
			(160, 192, 224),
			[16, 32, 32, 32],
			nf_dec,
			diffeo=True,
			interp=False
		)
	else:
		vm_model = vms_networks.vmnet(
			vol_size=img_shape[:-1],
			enc_nf=[16, 32, 32, 32],
			dec_nf=[32, 32, 32, 32, 32, 16, 16, 3],
		)

	input_src = Input(img_shape)
	input_tgt = Input(img_shape)

	transformed, flow = vm_model([input_src, input_tgt])
	flow = Lambda(lambda x: tf.gather(x, [1, 0, 2], axis=-1))(flow)
	transformed = Reshape(img_shape, name='spatial_transformer')(transformed)
	unet_flow = Model(
		inputs=[input_src, input_tgt],
		outputs=[flow, transformed],
		name='{}_wrapper'.format(voxelmorph_arch)
	)
	return unet_flow


def VM2_model(img_shape,
				  model_name='VM2',
				  enc_params=None):
	x_src = Input(img_shape, name='input_src')
	x_tgt = Input(img_shape, name='input_tgt')

	x_stacked = Concatenate(axis=-1)([x_src, x_tgt])

	n_dims = len(img_shape) - 1

	if n_dims == 2:
		x_flow = basic_networks.unet2D(x_stacked, img_shape, 2,
									   nf_enc=enc_params['nf_enc'],
									   n_convs_per_stage=enc_params['n_convs_per_stage'],
									   include_residual=False, do_last_conv=False)
		conv_fn = Conv2D
	else:
		x_flow = basic_networks.unet3D(x_stacked, img_shape, 3,
									   nf_enc=enc_params['nf_enc'],
									   n_convs_per_stage=enc_params['n_convs_per_stage'],
										use_maxpool=enc_params['use_maxpool'],
									   include_residual=False, do_last_conv=False)
		conv_fn = Conv3D


	x_flow = conv_fn(32, kernel_size=3, padding='same')(x_flow)
	x_flow = LeakyReLU(0.2)(x_flow)
	x_flow = conv_fn(16, kernel_size=3, padding='same')(x_flow)
	x_flow = LeakyReLU(0.2)(x_flow)
	x_flow = conv_fn(16, kernel_size=3, padding='same')(x_flow)
	x_flow = LeakyReLU(0.2)(x_flow)
	x_flow = conv_fn(3, kernel_size=3, padding='same')(x_flow)

	warped_out = nrn_layers.SpatialTransformer(name='spatial_transformer')([x_src, x_flow])
	return Model(inputs=[x_src, x_tgt], outputs=[x_flow, warped_out], name=model_name)


def VM_unet_model(img_shape,
				  model_name='VM_unet',
				  enc_params=None):
	x_src = Input(img_shape, name='input_src')
	x_tgt = Input(img_shape, name='input_tgt')

	x_stacked = Concatenate(axis=-1)([x_src, x_tgt])

	n_dims = len(img_shape) - 1

	if n_dims == 2:
		x_flow = basic_networks.unet2D(x_stacked, img_shape, 2,
									   nf_enc=enc_params['nf_enc'],
									   n_convs_per_stage=enc_params['n_convs_per_stage'],
									   include_residual=False)
	else:
		x_flow = basic_networks.unet3D(x_stacked, img_shape, 3,
									   nf_enc=enc_params['nf_enc'],
									   n_convs_per_stage=enc_params['n_convs_per_stage'],
									   include_residual=False)

	warped_out = nrn_layers.SpatialTransformer(name='spatial_transformer')([x_src, x_flow])
	return Model(inputs=[x_src, x_tgt], outputs=[x_flow, warped_out], name=model_name)
