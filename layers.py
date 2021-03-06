from builtins import range
import numpy as np


def linear_forward(x, w, b):
    """
    Computes the forward pass for an linear (fully-connected) layer.
    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.
    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)
    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the linear forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # print("x" + str(x.shape))
    # print("w" + str(w.shape))
    # print("b" + str(b.shape))

    x_reshaped = x.reshape(x.shape[0], -1)
    out = x_reshaped.dot(w) + b

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def linear_backward(dout, cache):
    """
    Computes the backward pass for an linear layer.
    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Bias, of shape (M,)
    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################

    x_reshaped = np.reshape(x, (x.shape[0], np.prod(x[0].shape)))
    dw = x_reshaped.T.dot(dout)
    dx = w.dot(dout.T).T.reshape(x.shape)
    db = np.sum(dout, axis=0)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).
    Input:
    - x: Inputs, of any shape
    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None

    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).
    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout
    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache

    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = (x > 0) * (dout)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def conv_forward(x, w, b, conv_param):
    """
    An implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 


    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################

    stride = conv_param['stride']
    pad = conv_param['pad']

    np.set_printoptions(suppress=True)
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

    H_p = 1 + (x.shape[2] + 2 * pad - w.shape[2]) // stride
    W_p = 1 + (x.shape[3] + 2 * pad - w.shape[3]) // stride

    out = np.zeros((x.shape[0], w.shape[0], H_p, W_p))

    for batch_index in range(x.shape[0]):
        for filter_index in range(w.shape[0]):
            height_out = 0
            for height_index in [x * stride for x in (list(range(out.shape[2])))]:
                width_out = 0
                for width_index in [x * stride for x in (list(range(out.shape[3])))]:
                    x_values = x_pad[batch_index, :, height_index:height_index + w.shape[2],
                               width_index:width_index + w.shape[3]]
                    w_values = w[filter_index, :, 0:w.shape[2], 0:w.shape[3]]
                    mult_values = x_values * w_values
                    sum_values = np.sum(mult_values) + b[filter_index]
                    out[batch_index, filter_index, height_out, width_out] = sum_values
                    width_out += 1
                height_out += 1

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward(dout, cache):
    """
    An implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache

    pad = conv_param['pad']
    stride = conv_param['stride']
    x_with_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)

    """ 
    Code with more loops but more easily understood

    # - b: Biases, of shape(F, )
    db = np.zeros_like(b)
    for filter_index in range(0, w.shape[0]): # one bias per filter
        db[filter_index] = np.sum(dout[:, filter_index, :, :])

    dw = np.zeros_like(w)
    for filter_index in range(0, w.shape[0]):
        for channel_index in range(0, w.shape[1]):
            for height_index in range(0, w.shape[2]):
                for width_index in range(0, w.shape[3]):
                    # no index for batch because we want the sum of the gradients for every batch
                    dw[filter_index, channel_index, height_index, width_index] = np.sum(
                        dout[:, filter_index, :, :] * x_with_pad[:, channel_index, height_index:height_index + dout.shape[2] * stride:stride, width_index:width_index + dout.shape[3] * stride:stride])

    dx = np.zeros_like(x)
    for batch_index in range(x.shape[0]):
        for height_index in range(x.shape[2]):
            for width_index in range(x.shape[3]):
                for filter_index in range(w.shape[0]):
                    for out_height_index in range(dout.shape[2]):
                        for out_width_index in range(dout.shape[3]):
                            mask1 = np.zeros_like(w[filter_index, :, :, :])
                            mask2 = np.zeros_like(w[filter_index, :, :, :])
                            if w.shape[2] > (height_index + pad - out_height_index * é) >= 0:
                                mask1[:, height_index + pad - out_height_index * stride, :] = 1.0
                            if w.shape[3] > (width_index + pad - out_width_index * stride) >= 0:
                                mask2[:, :, width_index + pad - out_width_index * stride] = 1.0

                            w_masked = np.sum(w[filter_index, :, :, :] * mask1 * mask2, axis=(1, 2))
                            dx[batch_index, :, height_index, width_index] += dout[batch_index, filter_index, out_height_index, out_width_index] * w_masked
    
    """

    db = np.zeros_like(b)
    dw = np.zeros_like(w)
    dx = np.zeros_like(x)
    for batch_index in range(x.shape[0]):
        for height_index in range(x.shape[2]):
            for width_index in range(x.shape[3]):
                for filter_index in range(w.shape[0]):
                    db[filter_index] = np.sum(dout[:, filter_index, :, :])
                    for channel_index in range(0, w.shape[1]):
                        for height_index_w in range(0, w.shape[2]):
                            for width_index_w in range(0, w.shape[3]):
                                dw[filter_index, channel_index, height_index_w, width_index_w] = np.sum(
                                    dout[:, filter_index, :, :] * x_with_pad[:, channel_index,
                                                                  height_index_w:height_index_w + dout.shape[
                                                                      2] * stride:stride,
                                                                  width_index_w:width_index_w + dout.shape[
                                                                      3] * stride:stride])
                    for out_height_index in range(dout.shape[2]):
                        for out_width_index in range(dout.shape[3]):
                            mask1 = np.zeros_like(w[filter_index, :, :, :])
                            mask2 = np.zeros_like(w[filter_index, :, :, :])
                            if w.shape[2] > (height_index + pad - out_height_index * stride) >= 0:
                                mask1[:, height_index + pad - out_height_index * stride, :] = 1.0
                            if w.shape[3] > (width_index + pad - out_width_index * stride) >= 0:
                                mask2[:, :, width_index + pad - out_width_index * stride] = 1.0

                            w_masked = np.sum(w[filter_index, :, :, :] * mask1 * mask2, axis=(1, 2))
                            dx[batch_index, :, height_index, width_index] += dout[
                                                                                 batch_index, filter_index, out_height_index, out_width_index] * w_masked

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward(x, pool_param):
    """
    An implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    H_p = 1 + (x.shape[2] - pool_height) // stride
    W_p = 1 + (x.shape[3] - pool_width) // stride

    out = np.zeros((x.shape[0], x.shape[1], H_p, W_p))
    for batch_index in range(x.shape[0]):
        for channel_index in range(x.shape[1]):
            height_out = 0
            for height_index in [x * stride for x in (list(range(out.shape[2])))]:
                width_out = 0
                for width_index in [x * stride for x in (list(range(out.shape[3])))]:
                    x_values = x[batch_index, channel_index, height_index:height_index + pool_height,
                               width_index:width_index + pool_width]

                    out[batch_index, channel_index, height_out, width_out] = np.amax(x_values)
                    width_out += 1
                height_out += 1

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward(dout, cache):
    """
    An implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    dx = np.zeros_like(x)

    for batch_index in range(x.shape[0]):
        for channel_index in range(x.shape[1]):
            height_dout = 0
            for height_index in [x * stride for x in (list(range(dout.shape[2])))]:
                width_dout = 0
                for width_index in [x * stride for x in (list(range(dout.shape[3])))]:
                    x_values = x[batch_index, channel_index, height_index:height_index + pool_height,
                               width_index:width_index + pool_width]
                    dx[batch_index, channel_index, height_index:height_index + pool_height,
                    width_index:width_index + pool_width] += dout[batch_index, channel_index, height_dout, width_dout] * (
                                                                         x_values == np.max(x_values))
                    width_dout += 1
                height_dout += 1

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx


def linear_relu_forward(x, w, b):
    """
    Convenience layer that performs an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = linear_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def linear_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = linear_backward(da, fc_cache)
    return dx, dw, db
