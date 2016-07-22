# Spaghetti

Spaghetti is an implementation of **Linear-Chain Conditional Random Fields**
(CRFs) as [Lasagne](https://github.com/Lasagne/Lasagne) layer. It facilitates
integrating CRFs with neural networks.

## Installation

To install Spaghetti, follow these steps. Adapt as necessary.

1. `git clone https://github.com/fdlm/Spaghetti.git`
2. `cd Spaghetti`
3. `python setup.py install`

## Examples

### Decoding with fixed parameters

```python
import numpy as np
import theano
import theano.tensor as T
import spaghetti as spg
import lasagne

# invent parameters for the CRF

eta = 0.000000000000001  # numerical stability
pi = np.log(np.array([0.6, 0.2, 0.1, 0.1], dtype=np.float32))
tau = np.log(np.ones(4, dtype=np.float32))
c = np.log(np.ones(4, dtype=np.float32))

A = np.log(np.array([[0.8, 0.2, 0.0, 0.0],
                     [0.1, 0.6, 0.3, 0.0],
                     [0.0, 0.2, 0.7, 0.1],
                     [0.0, 0.0, 0.4, 0.6]]) + eta).astype(np.float32)

W = np.log(np.array([[0.7,  0.1, 0.2, 0.3],
                     [0.15, 0.4, 0.7, 0.1],
                     [0.15, 0.5, 0.1, 0.6]]) + eta).astype(np.float32)

# create observation sequence in one-hot encoding

def to_onehot(seq, num_states=3):
    seq_oh = np.zeros(seq.shape + (num_states,), dtype=np.float32)
    seq_oh[range(len(seq)), seq] = 1.
    return seq_oh

x = to_onehot(np.array([0, 0, 1, 0, 0, 2, 1, 0, 2, 1, 0, 1, 1, 1, 0, 2,
                         0, 2, 0, 1, 1, 2, 0, 0, 0, 1]))[np.newaxis, ...]

# create simple crf model

x_var = T.ftensor3(name='x')
l_in = lasagne.layers.InputLayer(name='input', shape=(None, x.shape[1], 3),
                                 input_var=x_var)
l_crf = spg.layers.CrfLayer(incoming=l_in, num_states=4, name='crf',
                            pi=pi, tau=tau, c=c, A=A, W=W)
path = lasagne.layers.get_output(l_crf, mode='decoding')
decode = theano.function([x_var], path)

# decode the state sequence, convert it from one-hot to state id
print decode(x).argmax(axis=2)
```

### Training

```python
import spaghetti as spg
import lasagne as lnn
import numpy as np
import theano.tensor as tt
import theano

# one hot encoding of sequences

def to_onehot(seq, num_states=4):
    seq_oh = np.zeros(seq.shape + (num_states,), dtype=np.float32)
    seq_oh[range(len(seq)), seq] = 1.
    return seq_oh

x = np.stack((to_onehot(np.array([0, 0, 1, 0, 0, 2, 1, 0, 2, 1, 0, 1, 1, 1, 0, 2, 0, 2, 0, 1, 1, 2, 0, 0, 0, 1]), 3),
              to_onehot(np.array([2, 2, 2, 2, 1, 0, 2, 0, 0, 0, 1, 1, 1, 2, 0, 2, 2, 2, 0, 1, 1, 1, 1, 1, 1, 1]), 3)))

y = np.stack((to_onehot(np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 2, 2, 1, 0, 0, 0, 0])),
              to_onehot(np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2]))))

# create model
x_var = tt.ftensor3(name='x')   # observation sequence variable
y_var = tt.ftensor3(name='y')   # state sequence variable

l_in= lnn.layers.InputLayer(name='input', shape=(2, x.shape[1], 3),
                            input_var=x_var)

l_crf = spg.layers.CrfLayer(incoming=l_in, num_states=4, name='crf')

# create train function
objective = spg.objectives.neg_log_likelihood(l_crf, y_var)
params = lnn.layers.get_all_params(l_crf, trainable=True)
loss = objective.mean()
updates = lnn.updates.sgd(loss, params, learning_rate=0.01)
train = theano.function([y_var, x_var], loss, updates=updates)

for i in range(100):
    cur_loss = train(y, x)
    if i % 10 == 0:
        print cur_loss
```

## TODO

 - Add unit tests
 - Implement smoothing
