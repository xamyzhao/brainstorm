from keras.layers import Layer
from keras.layers import BatchNormalization, Input, Flatten, Dense, Reshape, Lambda, Activation, Dropout
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose, Cropping2D, ZeroPadding2D, MaxPooling2D
from keras.layers.convolutional import Conv3D, UpSampling3D, Conv3DTranspose, Cropping3D, ZeroPadding3D, MaxPooling3D
from keras.layers.merge import Add, Concatenate, Multiply
from keras.layers.pooling import MaxPooling2D
from keras import regularizers, initializers
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
import keras.backend as K
import numpy as np
import tensorflow as tf

def autoencoder(img_shape, latent_dim=10,
                conv_chans = None,
                n_convs_per_stage = None,
				fully_conv=False,
                name_prefix = ''):
	x_input = Input(img_shape)

	x_enc = encoder(x_input, img_shape, conv_chans = conv_chans,
	            n_convs_per_stage = n_convs_per_stage,
	            prefix = 'autoencoder_enc')
	preflatten_shape = get_encoded_shape(img_shape=img_shape, conv_chans=conv_chans)
	print('Autoencoder preflatten shape {}'.format(preflatten_shape))

	if fully_conv:
		x_enc = Flatten()(x_enc)
		x_enc = Dense(latent_dim, name='autoencoder_latent')(x_enc)
		x = Dense(np.prod(preflatten_shape))(x_enc)
	else:
		x = Reshape(preflatten_shape, name='autoencoder_latent')(x_enc)

	x = Reshape(preflatten_shape)(x)
	y = decoder(x, img_shape, conv_chans=conv_chans,
	            n_convs_per_stage=n_convs_per_stage,
	            prefix = 'autoencoder_dec',
				min_h = preflatten_shape[0])
	return Model(inputs=[x_input], outputs=[x_enc, y],
	             name=name_prefix + '_autoencoder')


def myConv(nf, n_dims, prefix=None, suffix=None, ks=3, strides=1):
	# wrapper for 2D and 3D conv
	if n_dims == 2:
		if not isinstance(strides, tuple):
			strides = (strides, strides)
		return Conv2D(nf, kernel_size=ks, padding='same', strides=strides,
		              name='_'.join([
			              str(part) for part in [prefix, 'conv2D', suffix]  # include prefix and suffix if they exist
			              if part is not None and len(str(part)) > 0]))
	elif n_dims == 3:
		if not isinstance(strides, tuple):
			strides = (strides, strides, strides)
		return Conv3D(nf, kernel_size=ks, padding='same', strides=strides,
		              name='_'.join([
			              str(part) for part in [prefix, 'conv3D', suffix]  # include prefix and suffix if they exist
			              if part is not None and len(str(part)) > 0]))

def myPool(n_dims, prefix, suffix):
	if n_dims == 2:
		return MaxPooling2D(padding='same',
		                    name='_'.join([
			                    str(part) for part in [prefix, 'maxpool2D', suffix]  # include prefix and suffix if they exist
			                    if part is not None and len(str(part)) > 0]))
	elif n_dims == 3:
		return MaxPooling3D(padding='same',
		                    name='_'.join([
			                    str(part) for part in [prefix, 'maxpool3D', suffix]  # include prefix and suffix if they exist
			                    if part is not None and len(str(part)) > 0]))
'''''''''
Basic encoder/decoders
- chans per stage specified
- additional conv with no activation at end to desired shape
'''''''''

def encoder(x, img_shape,
            conv_chans=None,
            n_convs_per_stage=1,
            min_h=5, min_c=None,
            prefix='',
            ks=3,
            return_skips=False, use_residuals=False, use_maxpool=False, use_batchnorm=False):
	skip_layers = []
	concat_skip_sizes = []
	n_dims = len(img_shape) - 1  # assume img_shape includes spatial dims, followed by channels

	if conv_chans is None:
		n_convs = int(np.floor(np.log2(img_shape[0] / min_h)))
		conv_chans = [min_c * 2] * (n_convs - 1) + [min_c]
	elif not type(conv_chans) == list:
		n_convs = int(np.floor(np.log2(img_shape[0] / min_h)))
		conv_chans = [conv_chans] * (n_convs - 1) + [min_c]

	for i in range(len(conv_chans)):
		if n_convs_per_stage is not None and n_convs_per_stage > 1 or use_maxpool and n_convs_per_stage is not None:
			for ci in range(n_convs_per_stage):
				x = myConv(nf=conv_chans[i], ks=ks, strides=1, n_dims=n_dims,
				           prefix='{}_enc'.format(prefix),
				           suffix='{}_{}'.format(i, ci + 1))(x)

				if ci == 0 and use_residuals:
					residual_input = x
				elif ci == n_convs_per_stage - 1 and use_residuals:
					x = Add(name='{}_enc_{}_add_residual'.format(prefix, i))([residual_input, x])

				if use_batchnorm:
					x = BatchNormalization()(x)
				x = LeakyReLU(0.2, name='{}_enc_leakyrelu_{}_{}'.format(prefix, i, ci + 1))(x)

		if return_skips:
			skip_layers.append(x)
			concat_skip_sizes.append(np.asarray(x.get_shape().as_list()[1:-1]))

		if use_maxpool:
			x = myPool(n_dims=n_dims, prefix=prefix, suffix=i)(x)
		else:
			x = myConv(conv_chans[i], ks=ks, strides=2, n_dims=n_dims,
			           prefix='{}_enc'.format(prefix), suffix=i)(x)

		# don't activate right after a maxpool, it makes no sense
		if not use_maxpool and i < len(conv_chans) - 1:  # no activation on last convolution
			x = LeakyReLU(0.2, name='{}_enc_leakyrelu_{}'.format(prefix, i))(x)

	if min_c is not None and min_c > 0:
		# if the last number of channels is specified, convolve to that
		if n_convs_per_stage is not None and n_convs_per_stage > 1:
			for ci in range(n_convs_per_stage):
				x = myConv(min_c, ks=ks, n_dims=n_dims, strides=1,
				           prefix='{}_enc'.format(prefix), suffix='last_{}'.format(ci + 1))(x)

				if ci == 0 and use_residuals:
					residual_input = x
				elif ci == n_convs_per_stage - 1 and use_residuals:
					x = Add(name='{}_enc_{}_add_residual'.format(prefix, 'last'))([residual_input, x])
				x = LeakyReLU(0.2, name='{}_enc_leakyrelu_last'.format(prefix))(x)

		x = myConv(min_c, ks=ks, strides=1, n_dims=n_dims,
		           prefix='{}_enc'.format(prefix),
		           suffix='_last')(x)

		if return_skips:
			skip_layers.append(x)
			concat_skip_sizes.append(np.asarray(x.get_shape().as_list()[1:-1]))

	if return_skips:
		return x, skip_layers, concat_skip_sizes
	else:
		return x


def encoder_model(img_shape,
				conv_chans = None,
				n_convs_per_stage = 1,
				min_h = 5, min_c = None,
				prefix = '',
				ks = 3,
			 return_skips=False, use_residuals=False, use_maxpool=False):
	x = Input(img_shape, name='{}_enc_input'.format(prefix))
	y = encoder(x, img_shape=img_shape,
				conv_chans=conv_chans,
				n_convs_per_stage=n_convs_per_stage,
				prefix=prefix, ks=ks,
				return_skips=return_skips,
				use_residuals=use_residuals,
				use_maxpool=use_maxpool
				)
	return Model(inputs=x, outputs=y, name='{}_encoder_model'.format(prefix))


'''''''''
Basic encoder/decoders
- chans per stage specified
- additional conv with no activation at end to desired shape
'''''''''
def encoder3D(x, img_shape,
			conv_chans=None,
			n_convs_per_stage=1,
			min_h=5, min_c=None,
			prefix='vte',
			ks=3,
			return_skips=False, use_residuals=False, use_maxpool=False,
			max_time_downsample=None):
	skip_layers = []
	concat_skip_sizes = []

	if max_time_downsample is None:
		# do not attempt to downsample beyond 1 
		max_time_downsample = int(np.floor(np.log2(img_shape[-2]))) - 1
		print('Max downsamples in time: {}'.format(max_time_downsample))

	if conv_chans is None:
		n_convs = int(np.floor(np.log2(img_shape[0] / min_h)))
		conv_chans = [min_c * 2] * (n_convs - 1) + [min_c]
	elif not type(conv_chans) == list:
		n_convs = int(np.floor(np.log2(img_shape[0] / min_h)))
		conv_chans = [conv_chans] * (n_convs - 1) + [min_c]

	for i in range(len(conv_chans)):
		if n_convs_per_stage is not None and n_convs_per_stage > 1 or use_maxpool and n_convs_per_stage is not None:
			for ci in range(n_convs_per_stage):
				x = Conv3D(conv_chans[i], kernel_size=ks, padding='same',
						   name='{}_enc_conv3D_{}_{}'.format(prefix, i, ci + 1))(x)
				if ci == 0 and use_residuals:
					residual_input = x
				elif ci == n_convs_per_stage - 1 and use_residuals:
					x = Add(name='{}_enc_{}_add_residual'.format(prefix, i))([residual_input, x])

				x = LeakyReLU(0.2, name='{}_enc_leakyrelu_{}_{}'.format(prefix, i, ci + 1))(x)

		if return_skips:
			skip_layers.append(x)
			concat_skip_sizes.append(x.get_shape().as_list()[1:-1])

		# only downsample if we are below the max number of downsamples in time
		if i < max_time_downsample:
			strides = (2, 2, 2)
		else:
			strides = (2, 2, 1)

		if use_maxpool:
			x = MaxPooling3D(pool_size=strides, 
							 name='{}_enc_maxpool_{}'.format(prefix, i))(x)
		else:
			x = Conv3D(conv_chans[i], kernel_size=ks, strides=strides, padding='same',
					   name='{}_enc_conv3D_{}'.format(prefix, i))(x)

		if i < len(conv_chans) - 1:  # no activation on last convolution
			x = LeakyReLU(0.2, name='{}_enc_leakyrelu_{}'.format(prefix, i))(x)

	if min_c is not None and min_c > 0:
		if n_convs_per_stage is not None and n_convs_per_stage > 1:
			for ci in range(n_convs_per_stage):
				x = Conv3D(min_c, kernel_size=ks, padding='same',
						   name='{}_enc_conv3D_last_{}'.format(prefix, ci + 1))(x)
				if ci == 0 and use_residuals:
					residual_input = x
				elif ci == n_convs_per_stage - 1 and use_residuals:
					x = Add(name='{}_enc_{}_add_residual'.format(prefix, 'last'))([residual_input, x])
				x = LeakyReLU(0.2, name='{}_enc_leakyrelu_last'.format(prefix))(x)
		x = Conv3D(min_c, kernel_size=ks, strides=(1, 1, 1), padding='same',
				   name='{}_enc_conv3D_last'.format(prefix))(x)
		if return_skips:
			skip_layers.append(x)
			concat_skip_sizes.append(x.get_shape().as_list()[1:-1])

	if return_skips:
		return x, skip_layers, concat_skip_sizes
	else:
		return x

def myConvTranspose(nf, n_dims, prefix=None, suffix=None, ks=3, strides=1):
	# wrapper for 2D and 3D conv
	if n_dims == 2:
		if not isinstance(strides, tuple):
			strides = (strides, strides)
		return Conv2DTranspose(nf, kernel_size=ks, padding='same', strides=strides,
		              name='_'.join([
			              str(part) for part in [prefix, 'conv2Dtrans', suffix]  # include prefix and suffix if they exist
			              if part is not None and len(str(part)) > 0]))
	elif n_dims == 3:
		if not isinstance(strides, tuple):
			strides = (strides, strides, strides)
		return Conv3DTranspose(nf, kernel_size=ks, padding='same', strides=strides,
		              name='_'.join([
			              str(part) for part in [prefix, 'conv3Dtrans', suffix]  # include prefix and suffix if they exist
			              if part is not None and len(str(part)) > 0]))

def myUpsample(n_dims, size=2, prefix=None, suffix=None):
	if n_dims == 2:
		if not isinstance(size, tuple):
			size = (size, size)

		return UpSampling2D(size=size,
		                    name='_'.join([
			                    str(part) for part in [prefix, 'upsamp2D', suffix]  # include prefix and suffix if they exist
			                    if part is not None and len(str(part)) > 0]))
	elif n_dims == 3:
		if not isinstance(size, tuple):
			size = (size, size, size)

		return UpSampling3D(size=size,
		                    name='_'.join([
			                    str(part) for part in [prefix, 'upsamp3D', suffix]  # include prefix and suffix if they exist
			                    if part is not None and len(str(part)) > 0]))


def decoder(x,
            img_shape,
            encoded_shape,
            conv_chans=None,
            min_h=5, min_c=4,
            prefix='vte_dec',
            n_convs_per_stage=None,
            include_dropout=False,
            ks=3,
            include_skips=None,
            use_residuals=False,
            use_upsample=False,
            target_vol_sizes=None,
            ):
	n_dims = len(img_shape) - 1

	if conv_chans is None:
		n_convs = int(np.floor(np.log2(img_shape[0] / min_h)))
		conv_chans = [min_c * 2] * n_convs
	elif not type(conv_chans) == list:
		n_convs = int(np.floor(np.log2(img_shape[0] / min_h)))
		conv_chans = [conv_chans] * n_convs
	elif type(conv_chans) == list:
		n_convs = len(conv_chans)

	# compute default sizes that we want on the way up, mainly in case we have more convs than stages
	# and we upsample past the output size
	if n_dims == 2:
		# just upsample by a factor of 2 and then crop the final volume to the desired volume
		default_target_vol_sizes = np.asarray(
			[(int(encoded_shape[0] * 2. ** (i + 1)), int(encoded_shape[1] * 2. ** (i + 1)))
			 for i in range(n_convs - 1)] + [img_shape[:2]])
	elif n_dims == 3:
		print(img_shape)
		print(encoded_shape)
		# just upsample by a factor of 2 and then crop the final volume to the desired volume
		default_target_vol_sizes = np.asarray(
			[(
				min(img_shape[0], int(encoded_shape[0] * 2. ** (i + 1))), 
				min(img_shape[1], int(encoded_shape[1] * 2. ** (i + 1))), 
				min(img_shape[2], int(encoded_shape[2] * 2. ** (i + 1)))) 
			for i in range(n_convs - 1)] + [img_shape[:3]])

	# automatically stop when we reach the desired image shape
	# TODO: is this desirable behavior? 
	for vi, vs in enumerate(default_target_vol_sizes):
		if np.all(vs >= img_shape[:-1]):
			default_target_vol_sizes[vi] = img_shape[:-1]
	print('Automatically computed target output sizes: {}'.format(default_target_vol_sizes))

	if target_vol_sizes is None:
		target_vol_sizes = default_target_vol_sizes
	else:
		print('Target encoder vols to match: {}'.format(target_vol_sizes))
		# fill in any Nones that we might have in our target_vol_sizes
		filled_target_vol_sizes = default_target_vol_sizes[:]
		for i in range(n_convs - 1):
			if i < len(target_vol_sizes) and target_vol_sizes[i] is not None:
				filled_target_vol_sizes[i] = target_vol_sizes[i]
		target_vol_sizes = filled_target_vol_sizes
	# target_vol_sizes = np.asarray(list(reversed([(int(np.ceil(img_shape[0]/2.0**i)), int(np.ceil(img_shape[1]/2.0**i))) \
	# 	for i in range(0, len(conv_chans) + 1)])))

	# curr_h = encoding_vol_sizes[-1] * 2
	print('Target encoder vols to match: {}'.format(target_vol_sizes))
	curr_shape = np.asarray(encoded_shape[:n_dims])
	for i in range(n_convs):
		print(x.get_shape())
		if np.all(curr_shape == img_shape[:len(curr_shape)]):
			x = myConv(conv_chans[i], n_dims=n_dims, ks=ks, strides=1, prefix=prefix, suffix=i)(x)
			x = LeakyReLU(0.2, name='{}_leakyrelu_{}'.format(prefix, i))(x)  # changed 5/15/2018, will break old models
		else:
			if not use_upsample:
				# if we have convolutional filters left at the end, just apply them at full resolution
				x = myConvTranspose(conv_chans[i], n_dims=n_dims,
									ks=ks, strides=2,
									prefix=prefix, suffix=i,
									)(x)
				x = LeakyReLU(0.2, name='{}_leakyrelu_{}'.format(prefix, i))(x)  # changed 5/15/2018, will break old models
			else:
				x = myUpsample(size=2, n_dims=n_dims, prefix=prefix, suffix=i)(x)
			curr_shape *= 2

			# if we want to concatenate with something from the encoder...
			if i < len(target_vol_sizes) and target_vol_sizes[i] is not None:
				x = _pad_or_crop_to_shape(x, curr_shape, target_vol_sizes[i])
				curr_shape = np.asarray(target_vol_sizes[i])  # we will upsample first thing next stage

			if include_skips is not None and i < len(include_skips) and include_skips[i] is not None:
				x = Concatenate(axis=-1)([x, include_skips[i]])

		for ci in range(n_convs_per_stage):
			x = myConv(conv_chans[i],
			           ks=ks,
			           strides=1,
			           n_dims=n_dims,
			           prefix=prefix,
			           suffix='{}_{}'.format(i, ci + 1))(x)
			# if we want residuals, store them here
			if ci == 0 and use_residuals:
				residual_input = x
			elif ci == n_convs_per_stage - 1 and use_residuals:
				x = Add(name='{}_{}_add_residual'.format(prefix, i))([residual_input, x])
			x = LeakyReLU(0.2,
			              name='{}_leakyrelu_{}_{}'.format(prefix, i, ci + 1))(x)

		if include_dropout and i < 2:
			x = Dropout(0.3)(x)

			

	# last stage of convolutions, no more upsampling
	x = myConv(img_shape[-1], ks=ks, n_dims=n_dims,
	           strides=1,
	           prefix=prefix,
	           suffix='final',
	           )(x)
	# TODO: is this what we want? probably dont want any repeated convs with 3 channels or whatever...
	#for ci in range(n_convs_per_stage - 1):
	#	x = LeakyReLU(0.2, name='{}_leakyrelu_final_{}'.format(prefix, ci + 1))(x)
	#	x = myConv(img_shape[-1], ks=ks, strides=1, n_dims=n_dims, prefix=prefix, suffix='final_{}'.format(ci + 1))(x)

	return x


def get_encoded_shape( img_shape, min_c = None, conv_chans = None, n_convs = None):
	if n_convs is None:
		n_convs = len(conv_chans)
		min_c = conv_chans[-1]
	encoded_shape = tuple([int(np.ceil(s/2. ** n_convs)) for s in img_shape[:-1]] + [min_c])
	#encoded_shape = (int(np.ceil(img_shape[0]/2.**n_convs)), int(np.ceil(img_shape[1]/2.**n_convs)), min_c)
	print('Encoded shape for img {} with {} convs is {}'.format(img_shape, n_convs, encoded_shape))
	return encoded_shape

def bousmalis_residual_morechans( x, bn=False, n_chans = 128):
	x0 = x
	x = Conv2D( n_chans, kernel_size = 1, padding='same', strides=(1,1) )(x)
	if bn:
		x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Conv2D( n_chans, kernel_size = 1, padding='same', strides=(1,1))(x)
	if bn:
		x = BatchNormalization()(x)
	x = Add()([x0, x])
	return x

def bousmalis_residual( x, bn=False ):
	x0 = x
	n_chans = 64
	x = Conv2D( n_chans, kernel_size = 1, padding='same', strides=(1,1) )(x)
	if bn:
		x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Conv2D( n_chans, kernel_size = 1, padding='same', strides=(1,1))(x)
	if bn:
		x = BatchNormalization()(x)
	x = Add()([x0, x])
#	x = Activation('relu')(x) #TODO: check if this should be here
	return x

def residual_EWM_encoded( x, w0, w1, bn = False, name_prefix = 'residual_elementwisemult' ):
	x0 = x
#	w0 = Lambda( lambda x:x[:,:,:,:3] )(w)
#	w1 = Lambda( lambda x:x[:,:,:,3:6] )(w)
	x = Multiply( name='{}_multiply_0'.format(name_prefix) )([x,w0])
	if bn:
		x = BatchNormalization()(x)
#	x = Activation('relu')(x)
	x = LeakyReLU(0.2)(x)
	x = Multiply( name='{}_multiply_1'.format(name_prefix) )([x,w1])	
	if bn:
		x = BatchNormalization()(x)
	x = Add( name='{}_add_input'.format(name_prefix) )([x0,x])
#	x = LeakyReLU(0.2)(x) # add this yourself later
	return x	

def conv2d_backend( x_in ):
#	x = x_in[0]
#	w = x_in[1]
	x,w = x_in
	print(w.get_shape())
	return K.conv2d(x,w)

def residual_encoded( x, w0, w1, bn = False ):
	x0 = x
#	w0 = Lambda( lambda x:x[:,:,:,:3] )(w)
#	w1 = Lambda( lambda x:x[:,:,:,3:6] )(w)
#	x = Multiply()([x,w0])
	x = Lambda( conv2d_backend )( [x,w0] )	
	if bn:
		x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Lambda( conv2d_backend )( [x,w1] )	
	if bn:
		x = BatchNormalization()(x)
	x = Add()([x0,x])
	return x	


def _pad_or_crop_to_shape(x, in_shape, tgt_shape):
	if len(in_shape) == 2:
		'''
		in_shape, tgt_shape are both 2x1 numpy arrays
		'''
		print('Padding input from {} to {}'.format(in_shape, tgt_shape))
		im_diff = in_shape - tgt_shape
		if im_diff[0] < 0:
			pad_amt = (int(np.ceil(abs(im_diff[0])/2.0)), int(np.floor(abs(im_diff[0])/2.0)))
			x = ZeroPadding2D( (pad_amt, (0,0)) )(x)
		if im_diff[1] < 0:
			pad_amt = (int(np.ceil(abs(im_diff[1])/2.0)), int(np.floor(abs(im_diff[1])/2.0)))
			x = ZeroPadding2D( ((0,0), pad_amt) )(x)

		if im_diff[0] > 0:
			crop_amt = (int(np.ceil(im_diff[0]/2.0)), int(np.floor(im_diff[0]/2.0))) 
			x = Cropping2D( (crop_amt, (0,0)) )(x)
		if im_diff[1] > 0:
			crop_amt = (int(np.ceil(im_diff[1]/2.0)), int(np.floor(im_diff[1]/2.0))) 
			x = Cropping2D( ((0,0),crop_amt) )(x)
		return x
	else:
		return _pad_or_crop_to_shape_3D(x, in_shape, tgt_shape)

def _pad_or_crop_to_shape_3D(x, in_shape, tgt_shape):
	'''
	in_shape, tgt_shape are both 2x1 numpy arrays
	'''
	im_diff = np.asarray(in_shape[:3]) - np.asarray(tgt_shape[:3])
	print(im_diff)
	if im_diff[0] < 0:
		pad_amt = (int(np.ceil(abs(im_diff[0])/2.0)), int(np.floor(abs(im_diff[0])/2.0)))
		x = ZeroPadding3D((
				pad_amt, 
				(0,0), 
				(0,0)
		))(x)
	if im_diff[1] < 0:
		pad_amt = (int(np.ceil(abs(im_diff[1])/2.0)), int(np.floor(abs(im_diff[1])/2.0)))
		x = ZeroPadding3D(((0,0), pad_amt, (0,0)) )(x)
	if im_diff[2] < 0:
		pad_amt = (int(np.ceil(abs(im_diff[2])/2.0)), int(np.floor(abs(im_diff[2])/2.0)))
		x = ZeroPadding3D(((0,0), (0,0), pad_amt))(x)

	if im_diff[0] > 0:
		crop_amt = (int(np.ceil(im_diff[0]/2.0)), int(np.floor(im_diff[0]/2.0))) 
		x = Cropping3D((crop_amt, (0,0), (0,0)))(x)
	if im_diff[1] > 0:
		crop_amt = (int(np.ceil(im_diff[1]/2.0)), int(np.floor(im_diff[1]/2.0))) 
		x = Cropping3D(((0,0), crop_amt, (0,0)))(x)
	if im_diff[2] > 0:
		crop_amt = (int(np.ceil(im_diff[2]/2.0)), int(np.floor(im_diff[2]/2.0))) 
		x = Cropping3D(((0,0), (0,0), crop_amt))(x)
	return x


def unet2D(x_in,
           img_shape, out_im_chans,
           nf_enc=[64, 64, 128, 128, 256, 256, 512],
           nf_dec=None,
           regularizer=None, initializer=None, layer_prefix='unet',
           n_convs_per_stage=1,
           include_residual=False,
           use_maxpool=False,
           concat_at_stages=None,
			do_last_conv=True,
	):
	ks = 3

	encoding_im_sizes = np.asarray([
		(int(np.ceil(img_shape[0] / 2.0 ** i)), int(np.ceil(img_shape[1] / 2.0 ** i))) \
		for i in range(0, len(nf_enc) + 1)])

	reg_params = {}
	if regularizer == 'l1':
		reg = regularizers.l1(1e-6)
	else:
		reg = None

	if initializer == 'zeros':
		reg_params['kernel_initializer'] = initializers.Zeros()

	x = x_in
	# TODO: fix this later
	# start with the input channels
	nf_enc = [img_shape[-1]] + nf_enc
	encodings = []
	for i in range(len(nf_enc)):
		if not use_maxpool and i > 0:
			x = LeakyReLU(0.2)(x)

		for j in range(n_convs_per_stage):
			if nf_enc[i] is not None:  # in case we dont want to convolve at the first resolution
				x = Conv2D(nf_enc[i],
						   kernel_regularizer=reg, kernel_size=ks,
						   strides=(1, 1), padding='same',
						   name='{}_enc_conv2D_{}_{}'.format(layer_prefix, i, j + 1))(x)

			if concat_at_stages and concat_at_stages[i] is not None:
				x = Concatenate(axis=-1)([x, concat_at_stages[i]])

			if j == 0 and include_residual:
				residual_input = x
			elif j == n_convs_per_stage - 1 and include_residual:
				x = Add()([residual_input, x])
			x = LeakyReLU(0.2)(x)

		encodings.append(x)
		if i < len(nf_enc) - 1:
			if use_maxpool:
				x = MaxPooling2D(pool_size=(2, 2), padding='same',
				                 name='{}_enc_maxpool_{}'.format(layer_prefix, i))(x)
			else:
				x = Conv2D(nf_enc[i], kernel_size=ks, strides=(2, 2), padding='same',
				           name='{}_enc_conv2D_{}'.format(layer_prefix, i))(x)

	if nf_dec is None:
		nf_dec = list(reversed(nf_enc[1:]))

	decoding_im_sizes = [encoding_im_sizes[-1] * 2]
	for i in range(len(nf_dec)):
		x = UpSampling2D(size=(2, 2), name='{}_dec_upsamp_{}'.format(layer_prefix, i))(x)
		curr_shape = x.get_shape().as_list()[1:-1]
		x = _pad_or_crop_to_shape(x, curr_shape, encoding_im_sizes[-i - 2])

		#decoding_im_sizes.append(encoding_im_sizes[-i - 2] * 2)  # the next deconv layer will produce this image height
		x = Concatenate()([x, encodings[-i - 2]])
		x = LeakyReLU(0.2)(x)

		residual_input = x

		for j in range(n_convs_per_stage):
			x = Conv2D(nf_dec[i],
			           kernel_regularizer=reg,
			           kernel_size=ks, strides=(1, 1), padding='same',
			           name='{}_dec_conv2D_{}_{}'.format(layer_prefix, i, j))(x)
			if j == 0 and include_residual:
				residual_input = x
			elif j == n_convs_per_stage - 1 and include_residual:
				x = Add()([residual_input, x])
			x = LeakyReLU(0.2)(x)

	x = Concatenate()([x, encodings[0]])

	for j in range(n_convs_per_stage - 1):
		x = Conv2D(out_im_chans,
		           kernel_regularizer=reg,
		           kernel_size=ks, strides=(1, 1), padding='same',
		           name='{}_dec_conv2D_last_{}'.format(layer_prefix, j))(x)
		x = LeakyReLU(0.2)(x)
	if do_last_conv:
		y = Conv2D(out_im_chans, kernel_size=1, padding='same', kernel_regularizer=reg,
	           name='{}_dec_conv2D_final'.format(layer_prefix))(x)  # add your own activation after this model
	else:
		y = x

	# add your own activation after this model
	return y


def segnet2D(x_in,
		   img_shape, out_im_chans,
		   nf_enc=[64, 64, 128, 128, 256, 256, 512],
		   nf_dec=None,
		   regularizer=None, initializer=None, layer_prefix='segnet',
		   n_convs_per_stage=1,
		   include_residual=False,
		   concat_at_stages=None):
	ks = 3

	encoding_im_sizes = np.asarray([(int(np.ceil(img_shape[0]/2.0**i)), int(np.ceil(img_shape[1]/2.0**i))) \
									for i in range(0, len(nf_enc) + 1)])


	reg_params = {}
	if regularizer=='l1':
		reg = regularizers.l1(1e-6)
	else:
		reg = None

	if initializer=='zeros':
		reg_params['kernel_initializer'] = initializers.Zeros()

	x = x_in
	# start with the input channels
	encoder_pool_idxs = []

	for i in range(len(nf_enc)):
		for j in range(n_convs_per_stage):
			x = Conv2D(nf_enc[i],
					   kernel_regularizer=reg, kernel_size=ks,
					   strides=(1,1), padding='same',
					   name='{}_enc_conv2D_{}_{}'.format(layer_prefix, i, j+1))(x)

			if concat_at_stages and concat_at_stages[i] is not None:
				x = Concatenate(axis=-1)([x, concat_at_stages[i]])

			if j==0 and include_residual:
				residual_input = x
			elif j==n_convs_per_stage-1 and include_residual:
				x = Add()([residual_input, x])
			x = LeakyReLU(0.2)(x)

		x, pool_idxs = MaxPoolingWithArgmax2D(pool_size=(ks, ks), padding='same', name='{}_enc_maxpool_{}'.format(layer_prefix, i))(x)
		encoder_pool_idxs.append(pool_idxs)


	if nf_dec is None:
		nf_dec = list(reversed(nf_enc[1:]))

	decoding_im_sizes = [ encoding_im_sizes[-1]*2 ]
	for i in range(len(nf_dec)):
		x = MaxUnpooling2D()([x, encoder_pool_idxs[-1-i]])
		x = _pad_or_crop_to_shape(x, decoding_im_sizes[-1], encoding_im_sizes[-i-2] )


		decoding_im_sizes.append( encoding_im_sizes[-i-2] * 2 ) # the next deconv layer will produce this image height

		residual_input = x

		for j in range(n_convs_per_stage):
			x = Conv2D(nf_dec[i],
					   kernel_regularizer=reg,
					   kernel_size=ks, strides=(1,1), padding='same',
					   name='{}_dec_conv2D_{}_{}'.format(layer_prefix, i, j))(x)
			if j==0 and include_residual:
				residual_input = x
			elif j==n_convs_per_stage-1 and include_residual:
				x = Add()([residual_input, x])
			x = LeakyReLU(0.2)(x)

		if i < len(nf_dec) - 1:
			# an extra conv compared to unet, so that the unpool op gets the right number of filters
			x = Conv2D(nf_dec[i + 1],
					   kernel_regularizer=reg,
					   kernel_size=ks, strides=(1, 1), padding='same',
					   name='{}_dec_conv2D_{}_extra'.format(layer_prefix, i))(x)
			x = LeakyReLU(0.2)(x)

	y = Conv2D( out_im_chans, kernel_size=1, padding='same', kernel_regularizer=reg,
			name='{}_dec_conv2D_last_last'.format(layer_prefix))(x)		# add your own activation after this model
	# add your own activation after this model
	return y

def unet3D(x_in,
		   img_shape, out_im_chans,
		   nf_enc=[64, 64, 128, 128, 256, 256, 512],
		   nf_dec=None,
		   regularizer=None, initializer=None, layer_prefix='unet',
		   n_convs_per_stage=1,
		   include_residual=False, use_maxpool=True,
		   max_time_downsample=None,
           n_tasks=1,
		   use_dropout=False,
		   do_unpool=False,
			do_last_conv=True,
		):
	ks = 3
	if max_time_downsample is None:
		max_time_downsample = len(nf_enc)  # downsample in time all the way down

		encoding_im_sizes = np.asarray([(
					int(np.ceil(img_shape[0] / 2.0 ** i)), 
					int(np.ceil(img_shape[1] / 2.0 ** i)),
					int(np.ceil(img_shape[2] / 2.0 ** i)),
				) for i in range(0, len(nf_enc) + 1)])
	else:
		encoding_im_sizes = np.asarray([(
					int(np.ceil(img_shape[0] / 2.0 ** i)), 
					int(np.ceil(img_shape[1] / 2.0 ** i)),
					max(int(np.ceil(img_shape[2] / 2.0 ** (max_time_downsample))), int(np.ceil(img_shape[2] / 2.0 ** i))),
				) for i in range(0, len(nf_enc) + 1)])

	reg_params = {}
	if regularizer == 'l1':
		reg = regularizers.l1(1e-6)
	else:
		reg = None

	if initializer == 'zeros':
		reg_params['kernel_initializer'] = initializers.Zeros()

	x = x_in

	encodings = []
	encoding_im_sizes = []
	for i in range(len(nf_enc)):
		if not use_maxpool and i > 0:
			x = LeakyReLU(0.2)(x)

		for j in range(n_convs_per_stage):
			if nf_enc[i] is not None:  # in case we dont want to convovle at max resolution
				x = Conv3D(
					nf_enc[i],
					kernel_regularizer=reg, kernel_size=ks,
					strides=(1, 1, 1), padding='same',
					name='{}_enc_conv3D_{}_{}'.format(layer_prefix, i, j + 1))(x)
			#if use_dropout:
			#	x = Dropout(0.2)(x)

			if j == 0 and include_residual:
				residual_input = x
			elif j == n_convs_per_stage - 1 and include_residual:
				x = Add()([residual_input, x])

			x = LeakyReLU(0.2)(x)

		encodings.append(x)
		encoding_im_sizes.append(np.asarray(x.get_shape().as_list()[1:-1]))

		# only downsample if we haven't reached the max
		if i >= max_time_downsample:
			ds = (2, 2, 1)
		else:
			ds = (2, 2, 2)

		if i < len(nf_enc) - 1:
			if use_maxpool:
				x = MaxPooling3D(pool_size=ds, padding='same', name='{}_enc_maxpool_{}'.format(layer_prefix, i))(x)
				#x, pool_idxs = Lambda(lambda x:._max_pool_3d_with_argmax(x, ksize=ks, strides=(2, 2, 2), padding='same'), name='{}_enc_maxpool3dwithargmax_{}'.format(layer_prefix, i))(x)
			else:
				x = Conv3D(nf_enc[i], kernel_size=ks, strides=ds,  padding='same', name='{}_enc_conv3D_{}'.format(layer_prefix, i))(x)

	if nf_dec is None:
		nf_dec = list(reversed(nf_enc[1:]))

	decoder_outputs = []
	x_encoded = x
	print(encoding_im_sizes)
	print(nf_dec)
	for ti in range(n_tasks):
		decoding_im_sizes = []
		x = x_encoded
		for i in range(len(nf_dec)):
			curr_shape = x.get_shape().as_list()[1:-1]

			print('Current shape {}, img shape {}'.format(x.get_shape().as_list(), img_shape))
			# only do upsample if we are not yet at max resolution
			if np.any(curr_shape < list(img_shape[:len(curr_shape)])):
				# TODO: fix this for time
				'''
				if i < len(nf_dec) - max_time_downsample + 1 \
						 or curr_shape[-1] >= encoding_im_sizes[-i-2][-1]:  # if we are already at the correct time scale
					us = (2, 2, 1)
				else:
				'''
				us = (2, 2, 2)
				#decoding_im_sizes.append(encoding_im_sizes[-i-1] * np.asarray(us))

				x = UpSampling3D(size=us, name='{}_dec{}_upsamp_{}'.format(layer_prefix, ti, i))(x)

			# just concatenate the final layer here
			if i <= len(encodings) - 2:
				x = _pad_or_crop_to_shape_3D(x, np.asarray(x.get_shape().as_list()[1:-1]), encoding_im_sizes[-i-2])
				x = Concatenate(axis=-1)([x, encodings[-i-2]])
				#x = LeakyReLU(0.2)(x)
			residual_input = x

			for j in range(n_convs_per_stage):
				x = Conv3D(nf_dec[i],
						   kernel_regularizer=reg,
						   kernel_size=ks, strides=(1, 1, 1), padding='same',
						   name='{}_dec{}_conv3D_{}_{}'.format(layer_prefix, ti, i, j))(x)
				if use_dropout and i < 2:
					x = Dropout(0.2)(x)
				if j == 0 and include_residual:
					residual_input = x
				elif j == n_convs_per_stage - 1 and include_residual:
					x = Add()([residual_input, x])
				x = LeakyReLU(0.2)(x)
		'''
		x = UpSampling3D(size=(2, 2, 2), name='{}_dec{}_upsamp_last'.format(layer_prefix, ti))(x)
		decoding_im_sizes.append(encoding_im_sizes[1] * 2)
		x = _pad_or_crop_to_shape_3D(x, decoding_im_sizes[-1], encoding_im_sizes[0])
		'''
		#x = Concatenate(axis=-1)([x, encodings[0]])

		'''
		for j in range(n_convs_per_stage - 1):
			x = Conv3D(out_im_chans,
					   kernel_regularizer=reg,
					   kernel_size=ks, strides=(1, 1, 1), padding='same',
					   name='{}_dec{}_conv3D_last_{}'.format(layer_prefix, ti, j))(x)
			x = LeakyReLU(0.2)(x)
		'''

		if do_last_conv:
			y = Conv3D(out_im_chans, kernel_size=1, padding='same', kernel_regularizer=reg,
					   name='{}_dec{}_conv3D_final'.format(layer_prefix, ti))(x)  # add your own activation after this model
		else:
			y = x
		decoder_outputs.append(y)
	# add your own activation after this model
	if n_tasks == 1:
		return y
	else:
		return decoder_outputs

# copied from https://github.com/rayanelleuch/tensorflow/blob/b46d50583d8f4893f1b1d629d0ac9cb2cff580af/tensorflow/contrib/layers/python/layers/layers.py#L2291-L2327
def unpool_2d(pool, 
              ind, 
              stride=[1, 2, 2, 1], 
			  ):
	"""Adds a 2D unpooling op.
	https://arxiv.org/abs/1505.04366
	Unpooling layer after max_pool_with_argmax.
	   Args:
		   pool:        max pooled output tensor
		   ind:         argmax indices
		   stride:      stride is the same as for the pool
	   Return:
		   unpool:    unpooling tensor
	"""
	input_shape = tf.shape(pool)
	output_shape = [input_shape[0], input_shape[1] * stride[1], input_shape[2] * stride[2], input_shape[3]]

	flat_input_size = tf.reduce_prod(input_shape)
	flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

	pool_ = tf.reshape(pool, [flat_input_size])
	batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype), 
									  shape=[input_shape[0], 1, 1, 1])
	b = tf.ones_like(ind) * batch_range
	b1 = tf.reshape(b, [flat_input_size, 1])
	ind_ = tf.reshape(ind, [flat_input_size, 1])
	ind_ = tf.concat([b1, ind_], 1)

	ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
	ret = tf.reshape(ret, output_shape)
	'''
	set_input_shape = pool.get_shape()
	set_output_shape = [set_input_shape[0], set_input_shape[1] * stride[1], set_input_shape[2] * stride[2], set_input_shape[3]]
	ret.set_shape(set_output_shape)
	'''

	return ret


# adapted from https://github.com/rayanelleuch/tensorflow/blob/b46d50583d8f4893f1b1d629d0ac9cb2cff580af/tensorflow/contrib/layers/python/layers/layers.py#L2291-L2327
def unpool_3d(pool, 
              ind, 
              stride=[1, 2, 2, 2, 1], 
              scope='unpool_3d'):
  """Adds a 2D unpooling op.
  https://arxiv.org/abs/1505.04366
  Unpooling layer after max_pool_with_argmax.
       Args:
           pool:        max pooled output tensor
           ind:         argmax indices
           stride:      stride is the same as for the pool
       Return:
           unpool:    unpooling tensor
  """
  with tf.variable_scope(scope):
    input_shape = tf.shape(pool)
    output_shape = [input_shape[0], input_shape[1] * stride[1], input_shape[2] * stride[2], input_shape[3] * stride[3], input_shape[4]]

    flat_input_size = tf.reduce_prod(input_shape)
    flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3] * output_shape[4]]

    pool_ = tf.reshape(pool, [flat_input_size])
    batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype), 
                                      shape=[input_shape[0], 1, 1, 1, 1])
    b = tf.ones_like(ind) * batch_range
    b1 = tf.reshape(b, [flat_input_size, 1])
    ind_ = tf.reshape(ind, [flat_input_size, 1])
    ind_ = tf.concat([b1, ind_], 1)

    ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
    ret = tf.reshape(ret, output_shape)

    set_input_shape = pool.get_shape()
    set_output_shape = [set_input_shape[0], set_input_shape[1] * stride[1], set_input_shape[2] * stride[2], set_input_shape[3] * stride[3], set_input_shape[4]]
    ret.set_shape(set_output_shape)
    return ret


# copied from https://github.com/PavlosMelissinos/enet-keras/blob/master/src/models/layers/pooling.py
class MaxPoolingWithArgmax2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='same', **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        if K.backend() == 'tensorflow':
            ksize = [1, pool_size[0], pool_size[1], 1]
            padding = padding.upper()
            strides = [1, strides[0], strides[1], 1]
            output, argmax = K.tf.nn.max_pool_with_argmax(inputs, ksize=ksize, strides=strides, padding=padding)
        else:
            errmsg = '{} backend is not supported for layer {}'.format(K.backend(), type(self).__name__)
            raise NotImplementedError(errmsg)
        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [int(np.ceil(dim / float(ratio[idx]))) if dim is not None else None for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


class MaxUnpooling2D(Layer):
	def __init__(self, size=(2, 2), **kwargs):
		super(MaxUnpooling2D, self).__init__(**kwargs)
		self.size = size

	def call(self, inputs, output_shape=None):
		"""
		Seen on https://github.com/tensorflow/tensorflow/issues/2169
		Replace with unpool op when/if issue merged
		Add theano backend
		"""
		pool, ind = inputs[0], inputs[1]
		ind = K.cast(ind, tf.int64)

		with tf.variable_scope(self.name):
			input_shape = tf.shape(pool)
			output_shape = [input_shape[0], input_shape[1] * self.size[0], input_shape[2] * self.size[1], input_shape[3]]

			flat_input_size = tf.reduce_prod(input_shape)
			flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

			pool_ = tf.reshape(pool, [flat_input_size])
			batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype), 
											  shape=[input_shape[0], 1, 1, 1])
			b = tf.ones_like(ind) * batch_range
			b1 = tf.reshape(b, [flat_input_size, 1])
			ind_ = tf.reshape(ind, [flat_input_size, 1])
			ind_ = tf.concat([b1, ind_], 1)

			ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
			ret = tf.reshape(ret, output_shape)
			'''
			set_input_shape = pool.get_shape()
			set_output_shape = [set_input_shape[0], set_input_shape[1] * self.size[0], set_input_shape[2] * self.size[1], set_input_shape[3]]
			ret.set_shape(set_output_shape)
			'''
			return ret
		'''
		with K.tf.variable_scope(self.name):
			mask = K.cast(mask, 'int32')
			mask_flat = K.reshape(mask, [-1])
			input_shape = K.tf.shape(updates, out_type='int32')
			#  calculation new shape
			if output_shape is None:
				output_shape = (input_shape[0], input_shape[1] * self.size[0], input_shape[2] * self.size[1], input_shape[3])
			self.output_shape1 = output_shape

			# calculation indices for batch, height, width and feature maps
			one_like_mask = K.ones_like(mask_flat, dtype='int32')
			batch_shape = K.concatenate([[input_shape[0]], [1], [1], [1]], axis=0)
			batch_range = K.reshape(K.tf.range(output_shape[0], dtype='int32'), shape=batch_shape)
			b = one_like_mask * batch_range
			y = mask_flat // (output_shape[2] * output_shape[3])
			x = (mask_flat // output_shape[3]) % output_shape[2]
			feature_range = K.tf.range(output_shape[3], dtype='int32')
			f = one_like_mask * feature_range
			
			# transpose indices & reshape update values to one dimension
			updates_size = K.tf.size(updates)
			indices = K.transpose(K.reshape(K.stack([b, y, x, f]), [-1, updates_size]))
			values = K.reshape(updates, [updates_size])
			ret = K.tf.scatter_nd(indices, values, output_shape)
			return ret
		'''
	def compute_output_shape(self, input_shape):
		print(input_shape)
		mask_shape = input_shape[0]
		return mask_shape[0], mask_shape[1] * self.size[0], mask_shape[2] * self.size[1], mask_shape[3]
