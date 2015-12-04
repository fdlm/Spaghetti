import theano.tensor as tt
import theano
import lasagne as lnn


def neg_log_likelihood(crf, target):
    x = lnn.layers.get_output(crf.input_layers[0])
    y = target.dimshuffle(1, 0, 2)
    log_z = crf.get_output_for([x], mode='partition')
    x = x.dimshuffle(1, 0, 2)

    def seq_step(y_prev, y_cur, x_cur, lp, A, W, c):
        return lp + c.dot(y_cur.T) + (y_prev.dot(A) * y_cur).sum(axis=1) + \
               (x_cur.dot(W) * y_cur).sum(axis=1)

    init_lp = \
        y[0].dot(crf.pi) + crf.c.dot(y[0].T) + y[-1].dot(crf.tau) + \
        (x[0].dot(crf.W) * y[0]).sum(axis=1) - log_z

    seq_lp, _ = theano.scan(
        fn=seq_step,
        outputs_info=init_lp,
        # this starts with y_prev=y[0], y_cur=y[1]
        sequences=[dict(input=y, taps=[-1, 0]),
                   # the indexing is necessary so that it starts with x_cur=x[1]
                   x[1:]],
        non_sequences=[crf.A, crf.W, crf.c])

    # negate log likelihood because we are minimizing
    return -seq_lp[-1]