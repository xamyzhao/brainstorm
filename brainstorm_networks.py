import sys

import cv2
import keras.backend as K
import numpy as np
import tensorflow as tf

# TODO: move sampling functions elsewhere (VAE utils?)
from keras import backend as K
from keras.layers import Input, Lambda, MaxPooling2D, UpSampling2D, Reshape, MaxPooling3D, UpSampling3D, Conv2D, Conv3D, LeakyReLU
from keras.engine import Layer
from keras.models import Model

sys.path.append('../evolving_wilds')
from cnn_utils import image_utils

sys.path.append('../voxelmorph-sandbox/util')
#from spatial_transforms import Dense2DSpatialTransformer, Dense3DSpatialTransformer

sys.path.append('../neuron')
from neuron.utils import volshape_to_ndgrid
from neuron.layers import SpatialTransformer

from keras.layers import Add, Concatenate
import basic_networks
def color_delta_unet_model(img_shape,
						   n_output_chans,
						   model_name='color_delta_unet',
						   enc_params=None,
						   include_aux_input=False,
						   aux_input_shape=None
						   ):
	x_src = Input(img_shape, name='input_src')
	x_tgt = Input(img_shape, name='input_tgt')

	if include_aux_input:
		if aux_input_shape is None:
			aux_input_shape = img_shape
		x_seg = Input(aux_input_shape, name='input_src_aux')
		inputs = [x_src, x_tgt, x_seg]
		unet_input_shape = img_shape[:-1] + (img_shape[-1] * 2 + aux_input_shape[-1],)
	else:
		inputs = [x_src, x_tgt]
		unet_input_shape = img_shape[:-1] + (img_shape[-1] * 2,)
	x_stacked = Concatenate(axis=-1)(inputs)

	n_dims = len(img_shape) - 1

	if n_dims == 2:
		color_delta = basic_networks.unet2D(x_stacked, unet_input_shape, n_output_chans,
									   nf_enc=enc_params['nf_enc'],
										nf_dec=enc_params['nf_dec'],
									   n_convs_per_stage=enc_params['n_convs_per_stage'],
									   include_residual=False)
		conv_fn = Conv2D
	else:
		color_delta = basic_networks.unet3D(x_stacked, unet_input_shape, n_output_chans,
									   nf_enc=enc_params['nf_enc'],
										nf_dec=enc_params['nf_dec'],
									   n_convs_per_stage=enc_params['n_convs_per_stage'],
									   include_residual=False)
		conv_fn = Conv3D

	'''
	for nf in enc_params['nf_dec'][len(enc_params['nf_enc']):]:
		color_delta = conv_fn(32, kernel_size=3, padding='same')(color_delta)
		color_delta = LeakyReLU(0.2)(color_delta)
	'''

	color_delta = conv_fn(n_output_chans, kernel_size=3, padding='same')(color_delta)

	transformed_out = Add(name='add_color_delta')([x_src, color_delta])
	return Model(inputs=inputs, outputs=[color_delta, transformed_out], name=model_name)


def interp_upsampling(V):
	""" 
	upsample a field by a factor of 2
	TODO: should switch this to use neuron.utils.interpn()
	"""
	V = tf.reshape(V, [-1] + V.get_shape().as_list()[1:])
	grid = volshape_to_ndgrid([f*2 for f in V.get_shape().as_list()[1:-1]])
	grid = [tf.cast(f, 'float32') for f in grid]
	grid = [tf.expand_dims(f/2 - f, 0) for f in grid]
	offset = tf.stack(grid, len(grid) + 1)

	V = SpatialTransformer(interp_method='linear')([V, offset])
	return V



def sampling(args):
	epsilon_std = 1.0
	z_mean, z_logvar = args
	epsilon = K.random_normal(shape=K.shape(z_mean), mean=0.,
														stddev=epsilon_std)
	return z_mean + K.exp(z_logvar / 2.) * epsilon


def sampling_sigma1(z_mean):
	epsilon_std = 1.0
	epsilon = K.random_normal(shape=K.shape(z_mean), mean=0.,
														stddev=epsilon_std)
	return z_mean + epsilon


def affine_warp_model(img_shape, pad_val=1.):
	img_in = Input(img_shape, name='input_img')
	H_in = Input((2,3), name='input_transform')
	img = Lambda(lambda x:affineWarp(x[0], x[1], pad_val), name='lambda_transform_img')([img_in, H_in])
	return Model(inputs=[img_in, H_in], outputs=img, name='affine_warp_model')


def downsample_model(img_shape, scale_factor=0.5, do_blur=True):
	n_dims = len(img_shape) - 1
	img_in = Input(img_shape, name='input_img')
	blurred_downsampled_img = Blur_Downsample(n_dims=n_dims, n_chans=img_shape[-1], do_blur=do_blur)(img_in)
	return Model(inputs=img_in, outputs=blurred_downsampled_img, name='blur_downsample_model')


# TODO: move this to some voxelmorph-specific file?
def vm_halfres_wrapper(full_img_shape, model, upsample_outputs=None, scale_outputs=None):
	n_dims = len(full_img_shape) - 1
	src_img_in = Input(full_img_shape, name='input_img_src')
	tgt_img_in = Input(full_img_shape, name='input_img_tgt')

	src_downsampled = Blur_Downsample(n_dims=n_dims, n_chans=full_img_shape[-1], do_blur=True)(src_img_in)
	tgt_downsampled = Blur_Downsample(n_dims=n_dims, n_chans=full_img_shape[-1], do_blur=True)(tgt_img_in)

	#src_downsampled = Lambda(lambda x:x*1.)(src_downsampled)
	#tgt_downsampled = Lambda(lambda x:x*1.)(tgt_downsampled)
	preds = model([src_downsampled, tgt_downsampled])
#	preds = model([src_img_in, tgt_img_in])

	if upsample_outputs is None:
		upsample_outputs = [False] * len(preds)
	if scale_outputs is None:
		scale_outputs = [1] * len(preds)

	upsampled_preds = []
	for pi, p in enumerate(preds):
		if upsample_outputs[pi]:
			upsampled_preds.append(UpsampleInterp()(preds[pi]))
		else:
			upsampled_preds.append(preds[pi])

		if scale_outputs[pi] > 1:
			upsampled_preds[pi] = Lambda(lambda x: x * scale_outputs[pi])(upsampled_preds[pi])

	return Model(inputs=[src_img_in, tgt_img_in], outputs=upsampled_preds, name='halfres_wrapper_{}'.format(model.name))

def warp_model(img_shape, interp_mode='linear', indexing='ij'):
	n_dims = len(img_shape) - 1
	img_in = Input(img_shape, name='input_img')
	flow_in = Input(img_shape[:-1] + (n_dims,), name='input_flow')
	img_warped = SpatialTransformer(interp_mode, indexing=indexing, name='densespatialtransformer_img')([img_in, flow_in])
	'''
	if n_dims == 3:
		img_warped = Dense3DSpatialTransformer(interp_mode, name='densespatialtransformer_img')([img_in, flow_in])
	else:
		img_warped = Dense2DSpatialTransformer(interp_mode, name='densespatialtransformer_img')([img_in, flow_in])
	'''
	return Model(inputs=[img_in, flow_in], outputs=img_warped, name='warp_model')


def randflow_ronneberger_model(img_shape,
				   model,
				   model_name='randflow_ronneberger_model',
				   flow_sigma=None,
				   flow_amp=None,
				   blur_sigma=5,
				   interp_mode='linear',
					indexing='xy',
				   ):
	n_dims = len(img_shape) - 1

	x_in = Input(img_shape, name='img_input_randwarp')
	#flow_placeholder = Input(img_shape[:-1] + (n_dims,), name='flow_input_placeholder')

	if n_dims == 3:
		n_pools = 5

		flow = MaxPooling3D(2)(x_in)
		for i in range(n_pools-1):
			flow = MaxPooling3D(2)(flow)
		# reduce flow by a factor of 64 until we have roughly 3x3x3
		flow_shape = tuple([int(s/(2**n_pools)) for s in img_shape[:-1]] + [n_dims])
		print('Smallest flow shape: {}'.format(flow_shape))
	else:
		#flow = flow_placeholder
		flow = x_in
		flow_shape = img_shape[:-1] + (n_dims,)
	# random flow field
	if flow_amp is None:
		# sigmas and blurring are hand-tuned to be similar to gaussian with stddev = 10, with smooth upsampling
		flow = RandFlow(name='randflow', img_shape=flow_shape, blur_sigma=0., flow_sigma=flow_sigma * 8)(flow)

	if n_dims == 3:
		print(flow_shape)
		print(flow.get_shape())
		flow = Reshape(flow_shape)(flow)
		flow_shape = flow_shape[:-1]
		for i in range(n_pools):
			flow_shape = [fs * 2 for fs in flow_shape]			
			flow = Lambda(interp_upsampling, output_shape=tuple(flow_shape) + (n_dims,))(flow)
			if i > 0 and i < 4:
				print(flow_shape)
				flow = BlurFlow(img_shape=tuple(flow_shape) + (n_dims,), blur_sigma=5,
					)(flow)#min(7, flow_shape[0]/4.))(flow)

		'''
		flow = Lambda(interp_upsampling)(flow)
		flow = Lambda(interp_upsampling)(flow)
		flow = Lambda(interp_upsampling)(flow)
		flow = Lambda(interp_upsampling)(flow)
		flow = Lambda(interp_upsampling)(flow), output_shape=img_shape[:-1] + (n_dims,))(flow)
		'''
		flow = basic_networks._pad_or_crop_to_shape(flow, flow_shape, img_shape)
		print('Cropped flow shape {}'.format(flow.get_shape()))
		#flow = UpSampling3D(2)(flow)
		print(img_shape[:-1] + (n_dims,))
		flow = BlurFlow(img_shape[:-1] + (n_dims,), blur_sigma=3)(flow)
		flow = Reshape(img_shape[:-1] + (n_dims,), name='randflow_out')(flow)
#		x_warped = Dense3DSpatialTransformer(interp_method=interp_mode, name='densespatialtransformer_img')([x_in, flow])
	else:
		flow = Reshape(img_shape[:-1] + (n_dims,), name='randflow_out')(flow)
#		x_warped = Dense2DSpatialTransformer(interp_method=interp_mode, name='densespatialtransformer_img')([x_in, flow])
	x_warped = SpatialTransformer(indexing=indexing, interp_method=interp_mode, name='densespatialtransformer_img')([x_in, flow])

	if model is not None:
		model_outputs = model(x_warped)
		if not isinstance(model_outputs, list):
			model_outputs = [model_outputs]
	else:
		model_outputs = [x_warped, flow]
	return Model(inputs=[x_in], outputs=model_outputs, name=model_name)

def randflow_model(img_shape,
				   model,
				   model_name='randflow_model',
				   flow_sigma=None,
				   flow_amp=None,
				   blur_sigma=5,
				   interp_mode='linear',
					indexing='xy',
				   ):
	n_dims = len(img_shape) - 1

	x_in = Input(img_shape, name='img_input_randwarp')
	#flow_placeholder = Input(img_shape[:-1] + (n_dims,), name='flow_input_placeholder')


	if n_dims == 3:
		flow = MaxPooling3D(2)(x_in)
		flow = MaxPooling3D(2)(flow)
		blur_sigma = int(np.ceil(blur_sigma / 4.))
		flow_shape = tuple([int(s/4) for s in img_shape[:-1]] + [n_dims])
	else:
		#flow = flow_placeholder
		flow = x_in
		flow_shape = img_shape[:-1] + (n_dims,)
	# random flow field
	if flow_amp is None:
		flow = RandFlow(name='randflow', img_shape=flow_shape, blur_sigma=blur_sigma, flow_sigma=flow_sigma)(flow)
	elif flow_sigma is None:
		flow = RandFlow_Uniform(name='randflow', img_shape=flow_shape, blur_sigma=blur_sigma, flow_amp=flow_amp)(flow)

	if n_dims == 3:
		flow = Reshape(flow_shape)(flow)
		# upsample with linear interpolation using adrian's function
		#flow = UpSampling3D(2)(flow)
		flow = Lambda(interp_upsampling)(flow)
		flow = Lambda(interp_upsampling, output_shape=img_shape[:-1] + (n_dims,))(flow)
		#flow = UpSampling3D(2)(flow)
		#flow = BlurFlow(name='blurflow', img_shape=img_shape, blur_sigma=3)(flow)
		flow = Reshape(img_shape[:-1] + (n_dims,), name='randflow_out')(flow)
#		x_warped = Dense3DSpatialTransformer(interp_method=interp_mode, name='densespatialtransformer_img')([x_in, flow])
	else:
		flow = Reshape(img_shape[:-1] + (n_dims,), name='randflow_out')(flow)
#		x_warped = Dense2DSpatialTransformer(interp_method=interp_mode, name='densespatialtransformer_img')([x_in, flow])
	x_warped = SpatialTransformer(interp_method=interp_mode, name='densespatialtransformer_img', indexing=indexing)([x_in, flow])


	if model is not None:
		model_outputs = model(x_warped)
		if not isinstance(model_outputs, list):
			model_outputs = [model_outputs]
	else:
		model_outputs = [x_warped, flow]
	return Model(inputs=[x_in], outputs=model_outputs, name=model_name)


def affineWarp(im, theta, constant_vals):
	num_batch = tf.shape(im)[0]
	height = tf.shape(im)[1]
	width = tf.shape(im)[2]

	orig_shape = tf.shape(im)

	# if there are time and chans dimensions at the end, squeeze them into chans
	im = tf.reshape(im, [num_batch, height, width, -1])
	channels = tf.shape(im)[3]

	x_t,y_t = _meshgrid(height, width)
	x_t_flat = tf.reshape(x_t, (1,-1))
	y_t_flat = tf.reshape(y_t, (1,-1))
	ones = tf.ones_like(x_t_flat)
	grid = tf.concat(axis=0, values=[x_t_flat,y_t_flat,ones])
	grid = tf.expand_dims(grid,0)
	grid = tf.reshape(grid,[-1])
	grid = tf.tile(grid, tf.stack([num_batch]))
	grid = tf.reshape(grid, tf.stack([num_batch,3,-1]))

	T_g = tf.matmul(theta, grid)
	x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
	y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])

	# adjust coordinates so that everything happens about the center of the image
	x_s_flat = tf.reshape(x_s, [-1]) + tf.scalar_mul(0.5, tf.cast(width, tf.float32))
	y_s_flat = tf.reshape(y_s, [-1]) + tf.scalar_mul(0.5, tf.cast(height, tf.float32))
	im_interp = interpolate([im, x_s_flat, y_s_flat], constant_vals)
	im_interp = tf.reshape(im_interp, orig_shape)
	return im_interp


def _repeat(x, n_repeats):
	rep = tf.transpose(
		tf.expand_dims(tf.ones(shape=tf.stack([n_repeats,])), 1),[1,0])
	rep = tf.cast(rep, dtype='int32')
	x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
	return tf.reshape(x,[-1])


def _meshgrid(height, width):
	w = tf.cast(width, tf.float32)
	h = tf.cast(height, tf.float32)
	x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
		tf.transpose(tf.expand_dims(
			tf.linspace(
				-tf.scalar_mul(0.5, w),
				tf.scalar_mul(0.5, w)-1.0, width),
		1), [1, 0]))
	y_t = tf.matmul(tf.expand_dims(
			tf.linspace(
				-tf.scalar_mul(0.5, h),
				tf.scalar_mul(0.5, h)-1.0, height),
		1),
		tf.ones(shape=tf.stack([1, width])))
	return x_t,y_t


def interpolate(inputs, constant_vals=0.):
	im = inputs[0]
	x = inputs[1]
	y = inputs[2]
	orig_im_shape = tf.shape(im)

	im = tf.pad(im, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT", constant_values=constant_vals)

	num_batch = tf.shape(im)[0]
	height = tf.shape(im)[1]
	width = tf.shape(im)[2]
	channels = tf.shape(im)[3]


	x = tf.reshape(x, [num_batch, -1])
	numel_xy = tf.size(x[0, :])
	x = tf.reshape(x, [-1])
	y = tf.reshape(y, [-1])

	x = tf.cast(x, 'float32') + 1
	y = tf.cast(y, 'float32') + 1

	max_x = tf.cast(width - 1, 'int32')
	max_y = tf.cast(height - 1, 'int32')

	x0 = tf.cast(tf.floor(x), 'int32')
	x1 = x0 + 1
	y0 = tf.cast(tf.floor(y), 'int32')
	y1 = y0 + 1

	x0 = tf.clip_by_value(x0, 0, max_x)
	x1 = tf.clip_by_value(x1, 0, max_x)
	y0 = tf.clip_by_value(y0, 0, max_y)
	y1 = tf.clip_by_value(y1, 0, max_y)

	dim2 = width
	dim1 = width * height
	base = _repeat(tf.range(num_batch) * dim1, numel_xy) #(height - 2) * (width - 2))

	base_y0 = base + y0 * dim2
	base_y1 = base + y1 * dim2

	idx_a = base_y0 + x0
	idx_b = base_y1 + x0
	idx_c = base_y0 + x1
	idx_d = base_y1 + x1

	# use indices to lookup pixels in the flat image and restore
	# channels dim
	im_flat = tf.reshape(im, tf.stack([-1, channels]))
	im_flat = tf.cast(im_flat, 'float32')

	Ia = tf.gather(im_flat, idx_a)
	Ib = tf.gather(im_flat, idx_b)
	Ic = tf.gather(im_flat, idx_c)
	Id = tf.gather(im_flat, idx_d)

	# and finally calculate interpolated values
	x0_f = tf.cast(x0, 'float32')
	x1_f = tf.cast(x1, 'float32')
	y0_f = tf.cast(y0, 'float32')
	y1_f = tf.cast(y1, 'float32')

	dx = x1_f - x
	dy = y1_f - y

	wa = tf.expand_dims((dx * dy), 1)
	wb = tf.expand_dims((dx * (1 - dy)), 1)
	wc = tf.expand_dims(((1 - dx) * dy), 1)
	wd = tf.expand_dims(((1 - dx) * (1 - dy)), 1)

	output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
	#output = tf.reshape(output, tf.stack([-1, height - 2, width - 2, channels]))
	output = tf.reshape(output, [num_batch, numel_xy, channels])
	output = tf.reshape(output, orig_im_shape)
	return output


class UpsampleInterp(Layer):
	def __init__(self, **kwargs):
		super(UpsampleInterp, self).__init__(**kwargs)
	def build(self, input_shape):
		self.built = True

	def call(self, inputs):
		return interp_upsampling(inputs)

	def compute_output_shape(self, input_shape):
		return tuple([input_shape[0]] + [int(s * 2) for s in input_shape[1:]] )


class Blur_Downsample(Layer):
	def __init__(self, n_chans, n_dims, do_blur=True, **kwargs):
		super(Blur_Downsample, self).__init__(**kwargs)
		scale_factor = 0.5 # we only support halving right now

		if do_blur:
			# according to scikit-image.transform.rescale documentation
			blur_sigma = (1. - scale_factor) / 2

			blur_kernel = image_utils.create_gaussian_kernel(blur_sigma, n_dims=n_dims, n_sigmas_per_side=4)
			if n_dims==2:
				blur_kernel = np.tile(np.reshape(blur_kernel, blur_kernel.shape + (1,1)), tuple([1]*n_dims) + (n_dims, 1))
			else:
				blur_kernel = np.reshape(blur_kernel, blur_kernel.shape + (1,1))
			self.blur_kernel = tf.constant(blur_kernel, dtype=tf.float32)
		else:
			self.blur_kernel = tf.constant(np.ones([1] * n_dims + [1, 1]), dtype=tf.float32)
		self.n_dims = n_dims
		self.n_chans = n_chans

	def build(self, input_shape):
		self.built = True
	
	def call(self, inputs):	
		if self.n_dims == 2:
			blurred = tf.nn.depthwise_conv2d(inputs, self.blur_kernel, 
											   padding='SAME', strides=[1, 2, 2, 1])
		elif self.n_dims == 3:
			chans_list = tf.unstack(inputs, num=self.n_chans, axis=-1)
			blurred_chans = []
			for c in range(self.n_chans):
				blurred_chan = tf.nn.conv3d(tf.expand_dims(chans_list[c], axis=-1), self.blur_kernel,
										 strides=[1, 2, 2, 2, 1], padding='SAME')
				blurred_chans.append(blurred_chan[:, :, :, :, 0])
			blurred = tf.stack(blurred_chans, axis=-1)
			#blurred = tf.nn.conv3d(inputs, self.blur_kernel, strides=[1, 2, 2, 2, 1], padding='SAME')
		return blurred


	def compute_output_shape(self, input_shape):
		return tuple([input_shape[0]] + [s/2 for s in input_shape[1:-1]] + [self.n_chans])


class RandFlow_Uniform(Layer):
	def __init__(self, img_shape, blur_sigma, flow_amp, **kwargs):
		super(RandFlow_Uniform, self).__init__(**kwargs)
		n_dims = len(img_shape) - 1

		self.flow_shape = img_shape[:-1] + (n_dims,)

		blur_kernel = image_utils.create_gaussian_kernel(blur_sigma, n_dims=n_dims, n_sigmas_per_side=4)
		# TODO: make this work for 3D
		if n_dims==2:
			blur_kernel = np.tile(np.reshape(blur_kernel, blur_kernel.shape + (1,1)), tuple([1]*n_dims) + (n_dims, 1))
		else:
			blur_kernel = np.reshape(blur_kernel, blur_kernel.shape + (1,1))
		self.blur_kernel = tf.constant(blur_kernel, dtype=tf.float32)
		self.flow_amp = flow_amp
		self.n_dims = n_dims

	def build(self, input_shape):
		self.built = True


	def call(self, inputs):
		if self.n_dims == 2:
			rand_flow = K.random_uniform(
				shape=tf.convert_to_tensor(
					[tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], self.n_dims]),
				minval=-self.flow_amp,
				maxval=self.flow_amp, dtype='float32')
			rand_flow = tf.nn.depthwise_conv2d(rand_flow, self.blur_kernel, strides=[1] * (self.n_dims + 2),
											   padding='SAME')
		elif self.n_dims == 3:
			rand_flow = K.random_uniform(
				shape=tf.convert_to_tensor(
					[tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3], self.n_dims]),
				minval=-self.flow_amp,
				maxval=self.flow_amp, dtype='float32')

			# blur it here, then again later?
			rand_flow_list = tf.unstack(rand_flow, num=self.n_dims, axis=-1)
			flow_chans = []
			for c in range(self.n_dims):
				flow_chan = tf.nn.conv3d(tf.expand_dims(rand_flow_list[c], axis=-1), self.blur_kernel,
										 strides=[1] * (self.n_dims + 2), padding='SAME')
				flow_chans.append(flow_chan[:, :, :, :, 0])
			rand_flow = tf.stack(flow_chans, axis=-1)
		rand_flow = tf.reshape(rand_flow, [-1] + list(self.flow_shape))
		return rand_flow


	def compute_output_shape(self, input_shape):
		return tuple(input_shape[:-1] + (self.n_dims,))


class BlurFlow(Layer):
	def __init__(self, img_shape, blur_sigma, **kwargs):
		super(BlurFlow, self).__init__(**kwargs)
		n_dims = len(img_shape) - 1

		self.flow_shape = tuple(img_shape[:-1]) + (n_dims,)

		blur_kernel = image_utils.create_gaussian_kernel(blur_sigma, n_dims=n_dims, n_sigmas_per_side=2)
		# TODO: make this work for 3D
		if n_dims==2:
			blur_kernel = np.tile(np.reshape(blur_kernel, blur_kernel.shape + (1,1)), tuple([1]*n_dims) + (n_dims, 1))
		else:
			blur_kernel = np.reshape(blur_kernel, blur_kernel.shape + (1,1))
		self.blur_kernel = tf.constant(blur_kernel, dtype=tf.float32)
		self.n_dims = n_dims

	
	def build(self, input_shape):
		self.built = True


	def call(self, inputs):
		# squeeze chanenls into batch so we can do a single conv
		flow_flat = tf.transpose(inputs, [0, 4, 1, 2, 3])
		flow_flat = tf.reshape(flow_flat, [-1] + list(self.flow_shape[:-1]))
		# convolve with blurring filter
		flow_blurred = tf.nn.conv3d(tf.expand_dims(flow_flat, axis=-1), self.blur_kernel,
									 strides=[1] * (self.n_dims + 2), padding='SAME')
		# get rid of extra channels
		flow_blurred = flow_blurred[:, :, :, :, 0]

		flow_out = tf.reshape(flow_blurred, [-1, self.n_dims] + list(self.flow_shape[:-1]))
		flow_out = tf.transpose(flow_out, [0, 2, 3, 4, 1])

		'''
		rand_flow_list = tf.unstack(inputs, num=self.n_dims, axis=-1)
		flow_chans = []
		for c in range(self.n_dims):
			flow_chan = tf.nn.conv3d(tf.expand_dims(rand_flow_list[c], axis=-1), self.blur_kernel,
									 strides=[1] * (self.n_dims + 2), padding='SAME')
			flow_chans.append(flow_chan[:, :, :, :, 0])
		rand_flow = tf.stack(flow_chans, axis=-1)
		'''
		return flow_out


class RandFlow(Layer):
	def __init__(self, img_shape, blur_sigma, flow_sigma, normalize_max=False, **kwargs):
		super(RandFlow, self).__init__(**kwargs)
		n_dims = len(img_shape) - 1

		self.flow_shape = img_shape[:-1] + (n_dims,)

		if blur_sigma > 0:
			blur_kernel = image_utils.create_gaussian_kernel(blur_sigma, n_dims=n_dims)
			# TODO: make this work for 3D
			if n_dims==2:
				blur_kernel = np.tile(np.reshape(blur_kernel, blur_kernel.shape + (1,1)), tuple([1]*n_dims) + (n_dims, 1))
			else:
				blur_kernel = np.reshape(blur_kernel, blur_kernel.shape + (1,1))
			self.blur_kernel = tf.constant(blur_kernel, dtype=tf.float32)
		else:
			self.blur_kernel = None
		self.flow_sigma = flow_sigma
		self.normalize_max = normalize_max
		self.n_dims = n_dims
		print('Randflow dims: {}'.format(self.n_dims))

	def build(self, input_shape):
		self.built = True

	def call(self, inputs):
		if self.n_dims == 2:
			rand_flow = K.random_normal(shape=tf.convert_to_tensor([tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], self.n_dims]), mean=0., stddev=1., dtype='float32')
			rand_flow = tf.nn.depthwise_conv2d(rand_flow, self.blur_kernel, strides=[1] * (self.n_dims+2), padding='SAME')
		elif self.n_dims == 3:
			rand_flow = K.random_normal(
				shape=tf.convert_to_tensor(
					[tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3], self.n_dims]),
					mean=0., stddev=1., dtype='float32')
			if self.blur_kernel is not None:
				rand_flow_list = tf.unstack(rand_flow, num=3, axis=-1)
				flow_chans = []
				for c in range(self.n_dims):
					flow_chan = tf.nn.conv3d(tf.expand_dims(rand_flow_list[c], axis=-1), self.blur_kernel, strides=[1] * (self.n_dims+2), padding='SAME')
					flow_chans.append(flow_chan[:,:,:,:,0])
				rand_flow = tf.stack(flow_chans, axis=-1)

#		rand_flow = K.cast(rand_flow / tf.reduce_max(tf.abs(rand_flow)) * self.flow_sigma, dtype='float32')
		if self.normalize_max:
			rand_flow = K.cast(tf.add_n([rand_flow * 0, rand_flow / tf.reduce_max(tf.abs(rand_flow)) * self.flow_sigma]), dtype='float32')
		else:
			rand_flow = K.cast(rand_flow * self.flow_sigma, dtype='float32')
		return rand_flow

	def compute_output_shape(self, input_shape):
		return tuple(input_shape[:-1] + (self.n_dims,))


class DilateAndBlur(Layer):
	def __init__(self, img_shape, dilate_kernel_size, blur_sigma, flow_sigma, **kwargs):
		super(DilateAndBlur, self).__init__(**kwargs)
		n_dims = len(img_shape) - 1

		self.flow_shape = img_shape[:-1] + (n_dims,)

		dilate_kernel = np.reshape(
				cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size)),
				(dilate_kernel_size, dilate_kernel_size, 1, 1))
		self.dilate_kernel = tf.constant(dilate_kernel, dtype=tf.float32)

		blur_kernel = image_utils.create_gaussian_kernel(blur_sigma, n_dims=n_dims)
		blur_kernel = np.reshape(blur_kernel, blur_kernel.shape + (1, 1))
		blur_kernel = blur_kernel / np.max(blur_kernel)  # normalize by max instead of by sum

		self.blur_kernel = tf.constant(blur_kernel, dtype=tf.float32)
		self.flow_sigma = flow_sigma


	def build(self, input_shape):
		self.built = True

	def call(self, inputs):
#		errormap = inputs - tf.reduce_min(inputs)
#		errormap = inputs / (1e-5 + tf.reduce_max(errormap))
		errormap = inputs[0]
		dilated_errormap = tf.nn.conv2d(errormap, self.dilate_kernel, strides=[1, 1, 1, 1], padding='SAME')
		blurred_errormap = tf.nn.conv2d(dilated_errormap, self.blur_kernel, strides=[1, 1, 1, 1], padding='SAME')
		blurred_errormap = K.cast(blurred_errormap / (1e-5 + tf.reduce_max(blurred_errormap)) * self.flow_sigma, dtype='float32')
		min_map = tf.tile(inputs[1][:,tf.newaxis, tf.newaxis,:],
				tf.concat([
						[1], tf.gather(tf.shape(blurred_errormap), [1,2,3])
				], 0))
		blurred_errormap = tf.maximum(min_map, blurred_errormap)
		return blurred_errormap
