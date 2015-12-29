import theano
import theano.tensor as tt
import lasagne as lnn


# this is taken from pylearn2 (https://github.com/lisa-lab/pylearn2)
def _log_sum_exp(x=None, axis=None):
    """
    A numerically stable expression for
    `T.log(T.exp(x).sum(axis=axis))`
    Parameters
    ----------
    x : theano.gof.Variable
        A tensor we want to compute the log sum exp of
    axis : int, optional
        Axis along which to sum
    Returns
    -------
    log_sum_exp : theano.gof.Variable
        The log sum exp of `A`
    """
    x_max = tt.max(x, axis=axis, keepdims=True)
    y = (
        tt.log(tt.sum(tt.exp(x - x_max), axis=axis, keepdims=True)) +
        x_max
    )

    if axis is None:
        return y.dimshuffle(())
    else:
        if type(axis) is int:
            axis = [axis]
        return y.dimshuffle([i for i in range(y.ndim) if
                             i % y.ndim not in axis])


def neg_log_likelihood(crf, target, mask=None):
    # get output and compute partition function
    x = lnn.layers.get_output(crf.input_layers[0])
    log_z = lnn.layers.get_output(crf, mode='partition')

    # noinspection PyPep8Naming
    def seq_step(y_prev, y_cur, x_cur, lp, A, W, c):
        return lp + c.dot(y_cur.T) + (y_prev.dot(A) * y_cur).sum(axis=1) + \
               (x_cur.dot(W) * y_cur).sum(axis=1)

    # noinspection PyPep8Naming
    def seq_step_masked(y_prev, y_cur, x_cur, mask_i, lp, A, W, c):
        lp_cur = c.dot(y_cur.T) + (y_prev.dot(A) * y_cur).sum(axis=1) + \
            (x_cur.dot(W) * y_cur).sum(axis=1)
        return lp + lp_cur * mask_i[0]

    # make time first dimension
    y = target.dimshuffle(1, 0, 2)
    x = x.dimshuffle(1, 0, 2)

    # create sequences - since we use x[0] already
    # for computing the initial value, we start from x[1]
    sequences = [dict(input=y, taps=[-1, 0]), x[1:]]
    if mask is not None:
        sequences.append(mask.dimshuffle(1, 0, 'x')[1:])

    # sum out all possibilities of y_0
    # assumes that:
    #  - for masked values the last valid y value is repeated!
    #  - assumes y_1 is never masked
    # this should work in the most common case where you mask at the
    # end of a sequence.
    # tricky: y_1 corresponds to y[0], while y_0 is a
    # non-existing 'virtual state'
    init_lp = \
        _log_sum_exp(crf.pi + crf.A.dot(y[0].T).T, axis=1) + \
        y[0].dot(crf.c) + (x[0].dot(crf.W) * y[0]).sum(axis=1) + \
        y[-1].dot(crf.tau) - log_z

    # process the sequence
    seq_lp, _ = theano.scan(
        fn=seq_step if mask is None else seq_step_masked,
        outputs_info=init_lp,
        sequences=sequences,
        non_sequences=[crf.A, crf.W, crf.c])

    # negate log likelihood because we are minimizing
    return -seq_lp[-1]
