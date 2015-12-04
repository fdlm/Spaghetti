"""
Layers that construct global connections using Conditional Random Fields.
Similar to recurrent layers, CRF layers expect the input shape to be
``(batch_size, sequence_length, num_inputs)``. The input is allowed to have
more than three dimensions in which case dimensions trailing the third dimension
are flattened.
"""
import lasagne as lnn
import theano
import theano.tensor as tt
import numpy as np


class CrfLayer(lnn.layers.MergeLayer):
    """
    spaghetti.layers.ViterbiLayer(incoming, pi, c, A, W, mask_input=None,
    **kwargs)

    Conditional random field Viterbi layer

    TODO: describe parameters
    """

    def __init__(self, incoming, num_states, pi, tau, c, A, W,
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
        return input_shape[0], input_shape[1]

    def _get_viterbi_output_for(self, sequences, num_batches):

        def vit_step(x_i, delta_p, A, W, c):
            all_trans = A + tt.shape_padright(delta_p)
            best_trans = tt.max(all_trans, axis=1)
            best_trans_id = tt.argmax(all_trans, axis=1)
            return c.T + x_i.dot(W) + best_trans, best_trans_id

        if isinstance(sequences, list):
            # TODO: Implement masked input with Viterbi algorithm
            raise NotImplementedError('Viterbi for masked input does not exist'
                                      ' yet')
        else:
            step_fun = vit_step

        delta_0 = \
            tt.repeat(tt.shape_padleft(self.pi + self.c), num_batches, axis=0) \
            + sequences[0].dot(self.W)

        ([deltas, back_ptrs], _) = theano.scan(
            fn=step_fun,
            sequences=sequences[1:],  # x_0 is already considered in delta_0
            outputs_info=[delta_0, None],
            non_sequences=[self.A, self.W, self.c],
            strict=True)

        deltas_N = deltas[-1] +\
            tt.repeat(tt.shape_padleft(self.pi + self.c), num_batches, axis=0)

        # add delta_0 and deltas_N
        deltas = tt.concatenate([tt.shape_padleft(delta_0),
                                 deltas[:-1],
                                 tt.shape_padleft(deltas_N)])

        def bcktr_step(back_ptrs, next_state, num_batches):
            return back_ptrs[tt.arange(num_batches), next_state]

        # y_star is the most probable state sequence
        y_star, _ = theano.scan(
            fn=bcktr_step,
            outputs_info=deltas[-1].argmax(axis=1),
            sequences=back_ptrs,
            non_sequences=[num_batches],
            go_backwards=True,
            strict=True)

        # add y_star_N, reverse to bring path in correct order and shape
        y_star = tt.concatenate([y_star[::-1],
                                 tt.shape_padleft(deltas[-1].argmax(axis=1))
                                 ]).T
        return y_star

    def _get_forward_output_for(self, sequences, num_batches):
        # here we assume that sequences is just containing the observations
        # TODO: take care of masked input

        def fwd_step(x_i, alpha_p, Z_p, A, W, c):
            f = tt.exp(c.T + x_i.dot(W)) * alpha_p.dot(tt.exp(A))
            return (f / tt.shape_padright(f.sum(axis=1)),
                    Z_p + tt.log(f.sum(axis=1)))

        alpha_0 = tt.exp(
            tt.repeat(tt.shape_padleft(self.pi + self.c), num_batches, axis=0) +
            sequences[0].dot(self.W))

        Z_0 = tt.log(alpha_0.sum(axis=1))
        alpha_0 /= tt.shape_padright(alpha_0.sum(axis=1))

        ([alphas, log_zs], upd) = theano.scan(
            fn=fwd_step,
            outputs_info=[alpha_0, Z_0],
            sequences=sequences[1:-1],  # we used x_0 already for alpha_0,
                                        # and the last step will be calculated
                                        # outside of the loop
            non_sequences=[self.A, self.W, self.c],
            strict=True)

        alpha_N = tt.exp(self.c.T + sequences[-1].dot(self.W) + self.tau.T) *\
            alphas[-1].dot(tt.exp(self.A))

        log_z = log_zs[-1] + tt.log(alpha_N.sum(axis=1))

        alpha_N /= tt.shape_padright(alpha_N.sum(axis=1))

        # add alpha_0 and alpha_N
        alphas = tt.concatenate([tt.shape_padleft(alpha_0),
                                 alphas,
                                 tt.shape_padright(alpha_N)])
        alphas = alphas.dimshuffle(1, 0, 2)

        return alphas, log_z

    def get_output_for(self, inputs, mode='viterbi', **kwargs):
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = inputs[1] if len(inputs) > 1 else None

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = tt.flatten(input, 3)

        # Input should be provided as (n_batch, n_time_steps, n_features)
        # but scan requires the iterable dimension to be first
        # So, we need to dimshuffle to (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batches, _ = input.shape
        sequences = [input, mask] if mask is not None else input

        if mode == 'viterbi':
            return self._get_viterbi_output_for(sequences, num_batches)
        elif mode == 'forward':
            return self._get_forward_output_for(sequences, num_batches)[0]
        elif mode == 'partition':
            return self._get_forward_output_for(sequences, num_batches)[1]
        else:
            raise NotImplementedError('Invalid mode "%s"'.format(mode))
