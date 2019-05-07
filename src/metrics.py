import numpy as np
import tensorflow as tf


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
            loss += tf.reduce_mean(
                tf.abs(
                    dCdx * tf.gather(segments_mask, tf.range(1, tf.shape(y_pred)[d+1]), axis=d+1)
                ))
        return loss

class SpatialIntensitySmoothness(object):
    def __init__(self, n_dims,
                 use_true_gradients=True, pred_image_output=None,
                 lambda_i=1.
                 ):
        self.n_dims = n_dims
        self.use_true_gradients = use_true_gradients
        self.pred_image_output = pred_image_output
        self.lambda_i = lambda_i

    def compute_loss(self, y_true, y_pred):
        loss = 0

        for d in range(self.n_dims):
            # we use x to indicate the current spatial dimension, not just the first
            dCdx = tf.gather(y_pred, tf.range(1, tf.shape(y_pred)[d + 1]), axis=d + 1) \
                   - tf.gather(y_pred, tf.range(0, tf.shape(y_pred)[d + 1] - 1), axis=d + 1)

            if self.use_true_gradients:
                dIdx = tf.abs(tf.gather(y_true, tf.range(1, tf.shape(y_true)[d + 1]), axis=d + 1) \
                              - tf.gather(y_true, tf.range(0, tf.shape(y_true)[d + 1] - 1), axis=d + 1))

            else:
                dIdx = self.lambda_i * tf.abs(
                    tf.gather(self.pred_image_output, tf.range(1, tf.shape(y_true)[d + 1]), axis=d + 1) \
                    - tf.gather(self.pred_image_output, tf.range(0, tf.shape(y_true)[d + 1] - 1), axis=d + 1))

            # average across spatial dims and color channels
            loss += tf.reduce_mean(tf.abs(dCdx * tf.exp(-dIdx)))
        return loss


def gradient_loss_l2(n_dims):
    def compute_loss(_, y_pred):
        loss = 0.
        for d in range(n_dims):
            # we use x to indicate the current spatial dimension, not just the first
            dIdx = tf.abs(tf.gather(y_pred, tf.range(1, tf.shape(y_pred)[d + 1]), axis=d + 1) \
                          - tf.gather(y_pred, tf.range(0, tf.shape(y_pred)[d + 1] - 1), axis=d + 1))

            # average across spatial dims and color channels
            loss += tf.reduce_mean(dIdx * dIdx)
        return loss / float(n_dims)

    return compute_loss


class NCC():
    """
    local (over window) normalized cross correlation
    """

    def __init__(self, win=None, eps=1e-5):
        self.win = win
        self.eps = eps

    def ncc(self, I, J):
        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(I.get_shape().as_list()) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        if self.win is None:
            self.win = [9] * ndims

        # get convolution function
        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        # compute filters
        sum_filt = tf.ones([*self.win, 1, 1])
        strides = [1] * (ndims + 2)
        padding = 'SAME'

        # compute local sums via convolution
        I_sum = conv_fn(I, sum_filt, strides, padding)
        J_sum = conv_fn(J, sum_filt, strides, padding)
        I2_sum = conv_fn(I2, sum_filt, strides, padding)
        J2_sum = conv_fn(J2, sum_filt, strides, padding)
        IJ_sum = conv_fn(IJ, sum_filt, strides, padding)

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return tf.reduce_mean(cc)

    def loss(self, I, J):
        return - self.ncc(I, J)
