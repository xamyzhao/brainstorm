import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Add, Concatenate
from tensorflow.keras.layers import Input, Lambda, Reshape, Activation
from tensorflow.keras.layers import Conv2D, Cropping2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from tensorflow.keras.layers import Conv3D, Cropping3D, MaxPooling3D, UpSampling3D, ZeroPadding3D
from tensorflow.keras.layers import LeakyReLU

from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model

from ext.neuron.neuron.utils import volshape_to_ndgrid
from ext.neuron.neuron.layers import SpatialTransformer


##############################################################################
# Basic networks
##############################################################################


def unet2D(x_in,
           img_shape, out_im_chans,
           nf_enc=[64, 64, 128, 128, 256, 256, 512],
           nf_dec=None,
           layer_prefix='unet',
           n_convs_per_stage=1,
        ):
    ks = 3
    x = x_in

    encodings = []
    encoding_vol_sizes = []
    for i in range(len(nf_enc)):
        for j in range(n_convs_per_stage):
            x = Conv2D(
                nf_enc[i],
                kernel_size=ks,
                strides=(1, 1), padding='same',
                name='{}_enc_conv2D_{}_{}'.format(layer_prefix, i, j + 1))(x)
            x = LeakyReLU(0.2)(x)

        encodings.append(x)
        encoding_vol_sizes.append(np.asarray(x.get_shape().as_list()[1:-1]))

        if i < len(nf_enc) - 1:
            x = MaxPooling2D(pool_size=(2, 2), padding='same', name='{}_enc_maxpool_{}'.format(layer_prefix, i))(x)

    if nf_dec is None:
        nf_dec = list(reversed(nf_enc[1:]))

    for i in range(len(nf_dec)):
        curr_shape = x.get_shape().as_list()[1:-1]

        # only do upsample if we are not yet at max resolution
        if np.any(curr_shape < list(img_shape[:len(curr_shape)])):
            x = UpSampling2D(size=(2, 2), name='{}_dec_upsamp_{}'.format(layer_prefix, i))(x)

        # just concatenate the final layer here
        if i <= len(encodings) - 2:
            x = _pad_or_crop_to_shape_2D(x, np.asarray(x.get_shape().as_list()[1:-1]), encoding_vol_sizes[-i-2])
            x = Concatenate(axis=-1)([x, encodings[-i-2]])

        for j in range(n_convs_per_stage):
            x = Conv2D(nf_dec[i],
                       kernel_size=ks, padding='same',
                       name='{}_dec_conv2D_{}_{}'.format(layer_prefix, i, j))(x)
            x = LeakyReLU(0.2)(x)


    y = Conv2D(out_im_chans, kernel_size=1, padding='same',
               name='{}_dec_conv2D_final'.format(layer_prefix))(x)  # add your own activation after this model

    # add your own activation after this model
    return y


def unet3D(x_in,
           img_shape, out_im_chans,
           nf_enc=[64, 64, 128, 128, 256, 256, 512],
           nf_dec=None,
           layer_prefix='unet',
           n_convs_per_stage=1,
        ):
    ks = 3
    x = x_in

    encodings = []
    encoding_vol_sizes = []
    for i in range(len(nf_enc)):
        for j in range(n_convs_per_stage):
            x = Conv3D(
                nf_enc[i],
                kernel_size=ks,
                strides=(1, 1, 1), padding='same',
                name='{}_enc_conv3D_{}_{}'.format(layer_prefix, i, j + 1))(x)
            x = LeakyReLU(0.2)(x)

        encodings.append(x)
        encoding_vol_sizes.append(np.asarray(x.get_shape().as_list()[1:-1]))

        if i < len(nf_enc) - 1:
            x = MaxPooling3D(pool_size=(2, 2, 2), padding='same', name='{}_enc_maxpool_{}'.format(layer_prefix, i))(x)

    if nf_dec is None:
        nf_dec = list(reversed(nf_enc[1:]))

    for i in range(len(nf_dec)):
        curr_shape = x.get_shape().as_list()[1:-1]

        # only do upsample if we are not yet at max resolution
        if np.any(curr_shape < list(img_shape[:len(curr_shape)])):
            us = (2, 2, 2)
            x = UpSampling3D(size=us, name='{}_dec_upsamp_{}'.format(layer_prefix, i))(x)

        # just concatenate the final layer here
        if i <= len(encodings) - 2:
            x = _pad_or_crop_to_shape_3D(x, np.asarray(x.get_shape().as_list()[1:-1]), encoding_vol_sizes[-i-2])
            x = Concatenate(axis=-1)([x, encodings[-i-2]])

        for j in range(n_convs_per_stage):
            x = Conv3D(nf_dec[i],
                       kernel_size=ks, strides=(1, 1, 1), padding='same',
                       name='{}_dec_conv3D_{}_{}'.format(layer_prefix, i, j))(x)
            x = LeakyReLU(0.2)(x)


    y = Conv3D(out_im_chans, kernel_size=1, padding='same',
               name='{}_dec_conv3D_final'.format(layer_prefix))(x)  # add your own activation after this model

    # add your own activation after this model
    return y


def _pad_or_crop_to_shape_2D(x, in_shape, tgt_shape):
    '''
    in_shape, tgt_shape are both 2x1 numpy arrays
    '''
    in_shape = np.asarray(in_shape)
    tgt_shape = np.asarray(tgt_shape)
    print('Padding input from {} to {}'.format(in_shape, tgt_shape))
    im_diff = in_shape - tgt_shape
    if im_diff[0] < 0:
        pad_amt = (int(np.ceil(abs(im_diff[0]) / 2.0)), int(np.floor(abs(im_diff[0]) / 2.0)))
        x = ZeroPadding2D((pad_amt, (0, 0)))(x)
    if im_diff[1] < 0:
        pad_amt = (int(np.ceil(abs(im_diff[1]) / 2.0)), int(np.floor(abs(im_diff[1]) / 2.0)))
        x = ZeroPadding2D(((0, 0), pad_amt))(x)

    if im_diff[0] > 0:
        crop_amt = (int(np.ceil(im_diff[0] / 2.0)), int(np.floor(im_diff[0] / 2.0)))
        x = Cropping2D((crop_amt, (0, 0)))(x)
    if im_diff[1] > 0:
        crop_amt = (int(np.ceil(im_diff[1] / 2.0)), int(np.floor(im_diff[1] / 2.0)))
        x = Cropping2D(((0, 0), crop_amt))(x)
    return x


def _pad_or_crop_to_shape_3D(x, in_shape, tgt_shape):
    '''
    in_shape, tgt_shape are both 2x1 numpy arrays
    '''
    im_diff = np.asarray(in_shape[:3]) - np.asarray(tgt_shape[:3])

    if im_diff[0] < 0:
        pad_amt = (int(np.ceil(abs(im_diff[0]) / 2.0)), int(np.floor(abs(im_diff[0]) / 2.0)))
        x = ZeroPadding3D((
            pad_amt,
            (0, 0),
            (0, 0)
        ))(x)
    if im_diff[1] < 0:
        pad_amt = (int(np.ceil(abs(im_diff[1]) / 2.0)), int(np.floor(abs(im_diff[1]) / 2.0)))
        x = ZeroPadding3D(((0, 0), pad_amt, (0, 0)))(x)
    if im_diff[2] < 0:
        pad_amt = (int(np.ceil(abs(im_diff[2]) / 2.0)), int(np.floor(abs(im_diff[2]) / 2.0)))
        x = ZeroPadding3D(((0, 0), (0, 0), pad_amt))(x)

    if im_diff[0] > 0:
        crop_amt = (int(np.ceil(im_diff[0] / 2.0)), int(np.floor(im_diff[0] / 2.0)))
        x = Cropping3D((crop_amt, (0, 0), (0, 0)))(x)
    if im_diff[1] > 0:
        crop_amt = (int(np.ceil(im_diff[1] / 2.0)), int(np.floor(im_diff[1] / 2.0)))
        x = Cropping3D(((0, 0), crop_amt, (0, 0)))(x)
    if im_diff[2] > 0:
        crop_amt = (int(np.ceil(im_diff[2] / 2.0)), int(np.floor(im_diff[2] / 2.0)))
        x = Cropping3D(((0, 0), (0, 0), crop_amt))(x)
    return x

##############################################################################
# Spatial transform model
##############################################################################
def cvpr2018_net(vol_size, enc_nf, dec_nf, indexing='ij', name="voxelmorph"):
    """
    From https://github.com/voxelmorph/voxelmorph.

    unet architecture for voxelmorph models presented in the CVPR 2018 paper.
    You may need to modify this code (e.g., number of layers) to suit your project needs.

    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4.
           e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6 (like voxelmorph-1) or 1x7 (voxelmorph-2)
    :return: the keras model
    """
    import tensorflow.keras.layers as KL

    ndims = len(vol_size)
    assert ndims==3, "ndims should be 3. found: %d" % ndims

    src = Input(vol_size + (1,), name='input_src')
    tgt = Input(vol_size + (1,), name='input_tgt')

    input_stack = Concatenate(name='concat_inputs')([src, tgt])

    # get the core model
    x = unet3D(input_stack, img_shape=vol_size, out_im_chans=ndims, nf_enc=enc_nf, nf_dec=dec_nf)

    # transform the results into a flow field.
    Conv = getattr(KL, 'Conv%dD' % ndims)
    flow = Conv(ndims, kernel_size=3, padding='same', name='flow',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x)

    # warp the source with the flow
    y = SpatialTransformer(interp_method='linear', indexing=indexing)([src, flow])
    # prepare model
    model = Model(inputs=[src, tgt], outputs=[y, flow], name=name)
    return model


##############################################################################
# Appearance transform model
##############################################################################
def color_delta_unet_model(img_shape,
                           n_output_chans,
                           model_name='color_delta_unet',
                           enc_params=None,
                           include_aux_input=False,
                           aux_input_shape=None,
                           do_warp_to_target_space=False
                           ):
    x_src = Input(img_shape, name='input_src')
    x_tgt = Input(img_shape, name='input_tgt')
    inputs = [x_src, x_tgt]

    if aux_input_shape is None:
        aux_input_shape = img_shape

    x_seg = Input(aux_input_shape, name='input_src_aux')
    inputs += [x_seg]

    if do_warp_to_target_space: # warp transformed vol to target space in the end
        n_dims = len(img_shape) - 1
        flow_srctotgt = Input(img_shape[:-1] + (n_dims,), name='input_flow')
        inputs += [flow_srctotgt]

    if include_aux_input:
        unet_inputs = [x_src, x_tgt, x_seg]
        unet_input_shape = img_shape[:-1] + (img_shape[-1] * 2 + aux_input_shape[-1],)
    else:
        unet_inputs = [x_src, x_tgt]
        unet_input_shape = img_shape[:-1] + (img_shape[-1] * 2,)
    x_stacked = Concatenate(axis=-1)(unet_inputs)

    n_dims = len(img_shape) - 1

    if n_dims == 2:
        color_delta = unet2D(x_stacked, unet_input_shape, n_output_chans,
                             nf_enc=enc_params['nf_enc'],
                             nf_dec=enc_params['nf_dec'],
                             n_convs_per_stage=enc_params['n_convs_per_stage'],
                             )
        conv_fn = Conv2D
    else:
        color_delta = unet3D(x_stacked, unet_input_shape, n_output_chans,
                             nf_enc=enc_params['nf_enc'],
                             nf_dec=enc_params['nf_dec'],
                             n_convs_per_stage=enc_params['n_convs_per_stage'],
                             )
        conv_fn = Conv3D

    # last conv to get the output shape that we want
    color_delta = conv_fn(n_output_chans, kernel_size=3, padding='same', name='color_delta')(color_delta)

    transformed_out = Add(name='add_color_delta')([x_src, color_delta])
    if do_warp_to_target_space:
        transformed_out = SpatialTransformer(indexing='xy')([transformed_out, flow_srctotgt])

    # hacky, but do a reshape so keras doesnt complain about returning an input
    x_seg = Reshape(aux_input_shape, name='aux')(x_seg)

    return Model(inputs=inputs, outputs=[transformed_out, color_delta, x_seg], name=model_name)


##############################################################################
# Model for warping volumes/segmentations on the GPU
##############################################################################
def warp_model(img_shape, interp_mode='linear', indexing='ij'):
    n_dims = len(img_shape) - 1
    img_in = Input(img_shape, name='input_img')
    flow_in = Input(img_shape[:-1] + (n_dims,), name='input_flow')

    img_warped = SpatialTransformer(
        interp_mode, indexing=indexing, name='densespatialtransformer_img')([img_in, flow_in])

    return Model(inputs=[img_in, flow_in], outputs=img_warped, name='warp_model')


##############################################################################
# Models and utils for generating random flow fields on the GPU
##############################################################################
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

    if n_dims == 3:
        flow = MaxPooling3D(2)(x_in)
        flow = MaxPooling3D(2)(flow)
        blur_sigma = int(np.ceil(blur_sigma / 4.))
        flow_shape = tuple([int(s/4) for s in img_shape[:-1]] + [n_dims])
    else:
        flow = x_in
        flow_shape = img_shape[:-1] + (n_dims,)

    # random flow field
    if flow_amp is None:
        flow = RandFlow(name='randflow', img_shape=flow_shape, blur_sigma=blur_sigma, flow_sigma=flow_sigma)(flow)
    elif flow_sigma is None:
        flow = RandFlow_Uniform(name='randflow', img_shape=flow_shape, blur_sigma=blur_sigma, flow_amp=flow_amp)(flow)

    if n_dims == 3:
        flow = Reshape(flow_shape)(flow)
        # upsample with linear interpolation
        flow = Lambda(interp_upsampling)(flow)
        flow = Lambda(interp_upsampling, output_shape=img_shape[:-1] + (n_dims,))(flow)
        flow = Reshape(img_shape[:-1] + (n_dims,), name='randflow_out')(flow)
    else:
        flow = Reshape(img_shape[:-1] + (n_dims,), name='randflow_out')(flow)

    x_warped = SpatialTransformer(interp_method=interp_mode, name='densespatialtransformer_img', indexing=indexing)(
        [x_in, flow])


    if model is not None:
        model_outputs = model(x_warped)
        if not isinstance(model_outputs, list):
            model_outputs = [model_outputs]
    else:
        model_outputs = [x_warped, flow]
    return Model(inputs=[x_in], outputs=model_outputs, name=model_name)


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


def create_gaussian_kernel(sigma, n_sigmas_per_side=8, n_dims=2):
    t = np.linspace(-sigma * n_sigmas_per_side / 2, sigma * n_sigmas_per_side / 2, int(sigma * n_sigmas_per_side + 1))
    gauss_kernel_1d = np.exp(-0.5 * (t / sigma) ** 2)

    if n_dims == 2:
        gauss_kernel_2d = gauss_kernel_1d[:, np.newaxis] * gauss_kernel_1d[np.newaxis, :]
    else:
        gauss_kernel_2d = gauss_kernel_1d[:, np.newaxis, np.newaxis] * gauss_kernel_1d[np.newaxis, np.newaxis,
                                                                       :] * gauss_kernel_1d[np.newaxis, :, np.newaxis]
    gauss_kernel_2d = gauss_kernel_2d / np.sum(gauss_kernel_2d)
    return gauss_kernel_2d


class UpsampleInterp(Layer):
    def __init__(self, **kwargs):
        super(UpsampleInterp, self).__init__(**kwargs)
    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        return interp_upsampling(inputs)

    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0]] + [int(s * 2) for s in input_shape[1:]] )


class RandFlow_Uniform(Layer):
    def __init__(self, img_shape, blur_sigma, flow_amp, **kwargs):
        super(RandFlow_Uniform, self).__init__(**kwargs)
        n_dims = len(img_shape) - 1

        self.flow_shape = img_shape[:-1] + (n_dims,)

        blur_kernel = create_gaussian_kernel(blur_sigma, n_dims=n_dims, n_sigmas_per_side=4)
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

        blur_kernel = create_gaussian_kernel(blur_sigma, n_dims=n_dims, n_sigmas_per_side=2)
        # TODO: make this work for 3D
        if n_dims == 2:
            blur_kernel = np.tile(np.reshape(blur_kernel, blur_kernel.shape + (1, 1)),
                                  tuple([1] * n_dims) + (n_dims, 1))
        else:
            blur_kernel = np.reshape(blur_kernel, blur_kernel.shape + (1, 1))
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

        return flow_out


class RandFlow(Layer):
    def __init__(self, img_shape, blur_sigma, flow_sigma, normalize_max=False, **kwargs):
        super(RandFlow, self).__init__(**kwargs)
        n_dims = len(img_shape) - 1

        self.flow_shape = img_shape[:-1] + (n_dims,)

        if blur_sigma > 0:
            blur_kernel = create_gaussian_kernel(blur_sigma, n_dims=n_dims)
            # TODO: make this work for 3D
            if n_dims == 2:
                blur_kernel = np.tile(np.reshape(blur_kernel, blur_kernel.shape + (1, 1)),
                                      tuple([1] * n_dims) + (n_dims, 1))
            else:
                blur_kernel = np.reshape(blur_kernel, blur_kernel.shape + (1, 1))
            self.blur_kernel = tf.constant(blur_kernel, dtype=tf.float32)
        else:
            self.blur_kernel = None
        self.flow_sigma = flow_sigma
        self.normalize_max = normalize_max
        self.n_dims = n_dims

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        if self.n_dims == 2:
            rand_flow = K.random_normal(
                shape=tf.convert_to_tensor(
                    [tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], self.n_dims]),
                mean=0., stddev=1., dtype='float32')
            rand_flow = tf.nn.depthwise_conv2d(rand_flow, self.blur_kernel, strides=[1] * (self.n_dims + 2),
                                               padding='SAME')
        elif self.n_dims == 3:
            rand_flow = K.random_normal(
                shape=tf.convert_to_tensor(
                    [tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3], self.n_dims]),
                mean=0., stddev=1., dtype='float32')
            if self.blur_kernel is not None:
                rand_flow_list = tf.unstack(rand_flow, num=3, axis=-1)
                flow_chans = []
                for c in range(self.n_dims):
                    flow_chan = tf.nn.conv3d(tf.expand_dims(rand_flow_list[c], axis=-1),
                                             self.blur_kernel, strides=[1] * (self.n_dims + 2), padding='SAME')
                    flow_chans.append(flow_chan[:, :, :, :, 0])
                rand_flow = tf.stack(flow_chans, axis=-1)

        if self.normalize_max:
            rand_flow = K.cast(
                tf.add_n([rand_flow * 0, rand_flow / tf.reduce_max(tf.abs(rand_flow)) * self.flow_sigma]),
                dtype='float32')
        else:
            rand_flow = K.cast(rand_flow * self.flow_sigma, dtype='float32')
        return rand_flow

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1] + (self.n_dims,))


##############################################################################
# Spatial transform model wrapper
##############################################################################
def bidir_wrapper(img_shape, fwd_model, bck_model, model_name='bidir_wrapper'):
    input_src = Input(img_shape)
    input_tgt = Input(img_shape)

    fwd_model.name = 'vm2_cc_fwd'
    bck_model.name = 'vm2_cc_bck'
    transformed_fwd, flow_fwd = fwd_model([input_src, input_tgt])
    transformed_bck, flow_bck = bck_model([input_tgt, input_src])
    return Model(inputs=[input_src, input_tgt], outputs=[transformed_fwd, transformed_bck, flow_fwd, flow_bck],
        name=model_name)


##############################################################################
# Segmentation model
##############################################################################
def segmenter_unet(img_shape, n_labels, params, model_name='segmenter_unet', activation='softmax'):
    n_dims = len(img_shape) - 1
    x_in = Input(img_shape, name='img_input')

    if 'nf_dec' not in params.keys():
        params['nf_dec'] = list(reversed(params['nf_enc']))

    if n_dims == 2:
        x = unet2D(x_in, img_shape, n_labels,
                   nf_enc=params['nf_enc'],
                   nf_dec=params['nf_dec'],
                   n_convs_per_stage=params['n_convs_per_stage']
                   )
    elif n_dims == 3:
        x = unet3D(x_in, img_shape, n_labels,
                   nf_enc=params['nf_enc'],
                   nf_dec=params['nf_dec'],
                   n_convs_per_stage=params['n_convs_per_stage'],
                   )

    if activation is not None:
        seg = Activation(activation)(x)
    else:
        seg = x

    return Model(inputs=[x_in], outputs=seg, name=model_name)
