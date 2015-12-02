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


class ViterbiLayer(lnn.layers.MergeLayer):
    """
    spaghetti.layers.ViterbiLayer(incoming, pi, c, A, W, mask_input=None,
    **kwargs)

    Conditional random field Viterbi layer

    TODO: describe parameters
    """

    def __init__(self, incoming, pi, c, A, W, mask_input=None, **kwargs):

        incomings = [incoming]
        if mask_input is not None:
            incomings.append(mask_input)

        super(ViterbiLayer, self).__init__(incomings, **kwargs)

        self.pi = pi
        self.c = c
        self.A = A
        self.W = W

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        return input_shape[0], input_shape[1]

    def get_output_for(self, inputs, **kwargs):

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
        seq_len, num_batch, _ = input.shape

        def vit_step(x_i, delta_p, A, W, c):
            all_trans = A + tt.shape_padright(delta_p)
            best_trans = tt.max(all_trans, axis=0)
            best_trans_id = tt.argmax(all_trans, axis=0)
            return c.T + x_i.T.dot(W) + best_trans, best_trans_id

        if mask is not None:
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            raise NotImplementedError('Viterbi for masked input does not exist'
                                      ' yet')
            step_fun = step_masked
        else:
            sequences = input
            step_fun = vit_step

        ([deltas, back_ptrs], _) = theano.scan(
            fn=vit_step,
            sequences=sequences,
            outputs_info=[tt.log(self.pi), None],
            non_sequences=[self.A, self.W, self.c],
            strict=True)
        # add delta_0
        deltas = tt.concatenate([tt.shape_padleft(tt.log(self.pi)), deltas])

        def bcktr_step(back_ptrs, next_state):
            return back_ptrs[next_state]

        # y_star is the most probable state sequence
        y_star, _ = theano.scan(
            fn=bcktr_step,
            outputs_info=deltas[-1].argmax(),
            sequences=back_ptrs,
            go_backwards=True,
            strict=True)

        # add y_star_N, reverse to bring path in correct order
        y_star = tt.concatenate([y_star[::-1],
                                 tt.shape_padleft(deltas[-1].argmax())])

        return y_star
