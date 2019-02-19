import sys

import cv2
import keras.backend as K
import numpy as np
import tensorflow as tf

# TODO: move sampling functions elsewhere (VAE utils?)
from keras import backend as K, Input, Model
from keras.layers import Input, Lambda, MaxPooling2D, UpSampling2D, Reshape, MaxPooling3D, UpSampling3D, Conv2D, Conv3D, \
    LeakyReLU, Activation
from keras.engine import Layer
from keras.models import Model

sys.path.append('../evolving_wilds')
from cnn_utils import basic_networks, image_utils

sys.path.append('../neuron')
from neuron.utils import volshape_to_ndgrid
from neuron.layers import SpatialTransformer

from keras.layers import Add, Concatenate

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
    if aux_input_shape is None:
        aux_input_shape = img_shape

    x_seg = Input(aux_input_shape, name='input_src_aux')
    inputs = [x_src, x_tgt, x_seg]

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

    # last conv to get the output shape that we want
    color_delta = conv_fn(n_output_chans, kernel_size=3, padding='same', name='color_delta')(color_delta)

    transformed_out = Add(name='add_color_delta')([x_src, color_delta])
    if do_warp_to_target_space:
        transformed_out = SpatialTransformer(indexing='xy')([transformed_out, flow_srctotgt])

    # kind of silly, but do a reshape so keras doesnt complain about returning an input
    x_seg = Reshape(aux_input_shape, name='aux')(x_seg)

    return Model(inputs=inputs, outputs=[color_delta, transformed_out, x_seg], name=model_name)

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
            blur_kernel = image_utils.create_gaussian_kernel(blur_sigma, n_dims=n_dims)
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
        print('Randflow dims: {}'.format(self.n_dims))

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

##############################################################################
# Spatial transform model wrappers
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


def voxelmorph_wrapper(img_shape, voxelmorph_arch='vm2_guha'):
    # just reverse the output order to be consistent with ours
    from keras.models import Model
    from keras.layers import Input, Lambda, Reshape

    sys.path.append('../voxelmorph')

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

##############################################################################
# Segmentation model
##############################################################################
def segmenter_unet(img_shape, n_labels, params, model_name='segmenter_unet', activation='softmax'):
    n_dims = len(img_shape) - 1
    x_in = Input(img_shape, name='img_input')

    if 'nf_dec' not in params.keys():
        params['nf_dec'] = list(reversed(params['nf_enc']))

    if n_dims == 2:
        x = basic_networks.unet2D(x_in, img_shape, n_labels,
                                  nf_enc=params['nf_enc'],
                                  nf_dec=params['nf_dec'],
                                  n_convs_per_stage=params['n_convs_per_stage'],
                                  use_maxpool=params['use_maxpool'],
                                  use_residuals=params['use_residuals'])
    elif n_dims == 3:
        x = basic_networks.unet3D(x_in, img_shape, n_labels,
                                  nf_enc=params['nf_enc'],
                                  nf_dec=params['nf_dec'],
                                  n_convs_per_stage=params['n_convs_per_stage'],
                                  use_maxpool=params['use_maxpool'],
                                  use_residuals=params['use_residuals'])

    if activation is not None:
        seg = Activation(activation)(x)
    else:
        seg = x

    return Model(inputs=[x_in], outputs=seg, name=model_name)
