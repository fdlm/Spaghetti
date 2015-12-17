import theano
import lasagne as lnn


def neg_log_likelihood(crf, target, mask=None):
    mask = [mask] if mask is not None else []

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
        return lp + lp_cur * mask_i

    # make time first dimension
    y = target.dimshuffle(1, 0, 2)
    x = x.dimshuffle(1, 0, 2)
    if mask:
        mask[0] = mask[0].dimshuffle(1, 0)

    # assumes that for masked values, the last y value is repeated!
    init_lp = y[0].dot(crf.pi) + y[-1].dot(crf.tau) - log_z

    seq_lp, _ = theano.scan(
        fn=seq_step if not mask else seq_step_masked,
        outputs_info=init_lp,
        # this starts with y_prev=y[0], y_cur=y[1]
        sequences=[dict(input=y, taps=[-1, 0]), x] + mask,
        non_sequences=[crf.A, crf.W, crf.c])

    # negate log likelihood because we are minimizing
    return -seq_lp[-1]
