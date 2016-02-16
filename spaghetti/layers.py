"""
Layers that construct global connections using Conditional Random Fields.
Similar to recurrent layers, CRF layers expect the input shape to be
``(batch_size, sequence_length, num_inputs)``. The input is allowed to have
more than three dimensions in which case dimensions trailing the third
dimension are flattened.
"""
import lasagne as lnn
import theano
import theano.tensor as tt
import numpy as np

STATE_ID_DTYPE = 'uint16'


# noinspection PyPep8Naming
class CrfLayer(lnn.layers.MergeLayer):
    """
    spaghetti.layers.ViterbiLayer(incoming, pi, c, A, W, mask_input=None,
    **kwargs)

    Conditional random field layer

    TODO: describe parameters
    """

    def __init__(self, incoming, num_states, pi=lnn.init.Constant(0.),
                 tau=lnn.init.Constant(0.), c=lnn.init.Constant(0.),
                 A=lnn.init.GlorotUniform(), W=lnn.init.GlorotUniform(),
                 mask_input=None, **kwargs):

        incomings = [incoming]
        if mask_input is not None:
            incomings.append(mask_input)

        super(CrfLayer, self).__init__(incomings, **kwargs)

        self.num_states = num_states
        num_inputs = int(np.prod(self.input_shapes[0][2:]))

        self.pi = self.add_param(pi, (num_states,), name='pi')
        self.tau = self.add_param(tau, (num_states,), name='tau')
        self.c = self.add_param(c, (num_states,), name='c')
        self.A = self.add_param(A, (num_states, num_states), name='A')
        self.W = self.add_param(W, (num_inputs, num_states), name='W')

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        return input_shape[0], input_shape[1], self.num_states

    def _get_viterbi_output_for(self, sequences, num_batches):

        def vit_step(x_i, delta_p, A, W, c):
            all_trans = A + tt.shape_padright(delta_p)
            best_trans = tt.max(all_trans, axis=1)
            best_trans_id = tt.cast(tt.argmax(all_trans, axis=1),
                                    dtype=STATE_ID_DTYPE)
            return c.T + x_i.dot(W) + best_trans, best_trans_id

        def vit_step_masked(x_i, mask_i, delta_p, A, W, c, masked_bck_ptrs):
            all_trans = A + tt.shape_padright(delta_p)
            best_trans = tt.max(all_trans, axis=1)
            best_trans_id = tt.cast(tt.argmax(all_trans, axis=1),
                                    dtype=STATE_ID_DTYPE)
            delta_c = c.T + x_i.dot(W) + best_trans

            return (delta_c * mask_i + delta_p * (1 - mask_i),
                    tt.cast(best_trans_id * mask_i +
                            masked_bck_ptrs * (1 - mask_i),
                            dtype=STATE_ID_DTYPE))

        # prepare initial values
        delta_0 = tt.repeat(tt.shape_padleft(self.pi), num_batches, axis=0)

        # choose step function
        if len(sequences) == 1:
            step_fun = vit_step
            non_sequences = [self.A, self.W, self.c]
        else:
            step_fun = vit_step_masked
            # We need backtracking pointers for masked steps. They just point
            # to the state itself, effectively just copying the decoded step
            non_sequences = [self.A, self.W, self.c,
                             tt.shape_padleft(tt.arange(0, self.num_states,
                                                        dtype=STATE_ID_DTYPE))]

        # loop over the observation sequence
        ([deltas, back_ptrs], _) = theano.scan(
            fn=step_fun,
            outputs_info=[delta_0, None],
            sequences=sequences,
            non_sequences=non_sequences,
            strict=True)

        # don't forget tau for the last step
        deltas_N = deltas[-1] + self.tau

        # noinspection PyShadowingNames
        def bcktr_step(back_ptrs, next_state, num_batches):
            return back_ptrs[tt.arange(num_batches), next_state]

        # y_star is the most probable state sequence
        y_star, _ = theano.scan(
            fn=bcktr_step,
            outputs_info=tt.cast(deltas_N.argmax(axis=1),
                                 dtype=STATE_ID_DTYPE),
            sequences=back_ptrs[1:],  # don't report the initial state y_0
            non_sequences=[num_batches],
            go_backwards=True,
            strict=True)

        # add y_star_N, reverse to bring path in correct order and shape
        y_star = tt.concatenate([y_star[::-1],
                                 tt.shape_padleft(deltas[-1].argmax(axis=1))
                                 ]).T

        # create one-hot encoding of state sequence. since theano's
        # "to_one_hot" function only takes vectors and converts them to
        # matrices, we have reshape forth and back
        y_star_oh = tt.extra_ops.to_one_hot(
            y_star.flatten(),
            self.num_states).reshape((num_batches, -1, self.num_states))

        return y_star_oh

    def _get_forward_output_for(self, sequences, num_batches):

        # define loop functions for theano scan, one for unmasked input,
        # one for masked input
        def fwd_step(x_i, alpha_p, Z_p, A, W, c):
            alpha_c = tt.exp(c.T + x_i.dot(W)) * alpha_p.dot(tt.exp(A))
            return (alpha_c / tt.shape_padright(alpha_c.sum(axis=1)),
                    Z_p + tt.log(alpha_c.sum(axis=1)))

        def fwd_step_masked(x_i, mask_i, alpha_p, Z_p, A, W, c):
            alpha_c = tt.exp(c.T + x_i.dot(W)) * alpha_p.dot(tt.exp(A))
            norm = alpha_c.sum(axis=1)
            alpha_c /= tt.shape_padright(norm)

            # use .squeeze() to remove last broadcastable dimension
            return (alpha_c * mask_i + alpha_p * (1 - mask_i),
                    Z_p + tt.log(norm) * mask_i.squeeze())

        # prepare initial values
        alpha_0 = tt.repeat(tt.shape_padleft(tt.exp(self.pi)),
                            num_batches, axis=0)
        Z_0 = tt.log(alpha_0.sum(axis=1))
        alpha_0 /= tt.shape_padright(alpha_0.sum(axis=1))

        # loop over the observation sequence
        ([alphas, log_zs], upd) = theano.scan(
            fn=fwd_step if len(sequences) == 1 else fwd_step_masked,
            outputs_info=[alpha_0, Z_0],
            sequences=sequences,
            non_sequences=[self.A, self.W, self.c],
            strict=True)

        # don't forget tau for the last step, recopute the log probability
        alphas_N = alphas[-1] * tt.exp(self.tau)
        norm = alphas_N.sum(axis=1)
        log_z = log_zs[-1] + tt.log(norm)
        alphas_N /= tt.shape_padright(norm)

        # add corrected alpha_N
        alphas = tt.concatenate([alphas[:-1], tt.shape_padleft(alphas_N)])

        # bring to (num_batches, seq_len, features) shape and return
        alphas = alphas.dimshuffle(1, 0, 2)
        return alphas, log_z

    def get_output_for(self, inputs, mode='viterbi', **kwargs):
        # Retrieve the layer input
        data = inputs[0]
        # Treat all dimensions after the second as flattened feature dimensions
        if data.ndim > 3:
            data = tt.flatten(data, 3)
        # Input should be provided as (n_batch, n_time_steps, n_features)
        # but scan requires the iterable dimension to be first
        # So, we need to dimshuffle to (n_time_steps, n_batch, n_features)
        data = data.dimshuffle(1, 0, 2)
        seq_len, num_batches, _ = data.shape
        sequences = [data]

        # Retrieve the mask when it is supplied
        if len(inputs) > 1:
            mask = inputs[1]
            mask = mask.dimshuffle(1, 0, 'x')
            sequences.append(mask)

        if mode == 'viterbi':
            return self._get_viterbi_output_for(sequences, num_batches)
        elif mode == 'forward':
            return self._get_forward_output_for(sequences, num_batches)[0]
        elif mode == 'partition':
            return self._get_forward_output_for(sequences, num_batches)[1]
        else:
            raise NotImplementedError('Invalid mode "%s"'.format(mode))
