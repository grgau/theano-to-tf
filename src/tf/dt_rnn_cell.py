from tensorflow.python.ops import math_ops, array_ops, nn_ops, init_ops
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras import activations

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class DT_RNNCell(tf.nn.rnn_cell.BasicRNNCell):
  def __init__(self, num_units, activation=None, name=None, dtype=None, use_noise=None):
    super(DT_RNNCell, self).__init__(num_units=num_units, name=name, dtype=dtype)

    # Inputs must be 2-dimensional.
    self.input_spec = input_spec.InputSpec(ndim=2)
    self._num_units = num_units
    self._use_noise = use_noise

    if activation:
      self._activation = activations.get(activation)
    else:
      self._activation = math_ops.tanh

  @property
  def state_size(self):
    return self._num_units[-1]

  @property
  def output_size(self):
    return self._num_units[-1]

  @tf_utils.shape_type_conversion
  def build(self, inputs_shape):
    if inputs_shape[-1] is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % str(inputs_shape))

    input_depth = inputs_shape[-1]

    self._kernel = [self.add_variable(_WEIGHTS_VARIABLE_NAME+"_0", shape=[input_depth + self._num_units[0], self._num_units[0]], initializer=tf.keras.initializers.glorot_normal())]
    for i in range(1, len(self._num_units)):
      self._kernel.append(self.add_variable(_WEIGHTS_VARIABLE_NAME+"_"+str(i), shape=[self._num_units[i-1], self._num_units[i]], initializer=tf.keras.initializers.glorot_normal()))

    self._bias = [self.add_variable(_BIAS_VARIABLE_NAME+"_0", shape=[self._num_units[0]], initializer=init_ops.zeros_initializer(dtype=self.dtype))]
    for i in range(1, len(self._num_units)-1):
      self._bias.append(self.add_variable(_BIAS_VARIABLE_NAME+"_"+str(i), shape=[self._num_units[i]], initializer=init_ops.zeros_initializer(dtype=self.dtype)))

    self.built = True

  def call(self, inputs, state):
    # Most basic RNN: new_state = act(W * state + U * input).
    gate_inputs = array_ops.concat([inputs, state], 1)
    gate_inputs = math_ops.matmul(gate_inputs, self._kernel[0])
    new_state = self._activation(gate_inputs)

    if self._use_noise:
      new_state = new_state + tf.random.normal(tf.shape(new_state), dtype=self.dtype)

    # Deepest RNN transitions: new_state = act(... W[3] * act(W[2] * act(... act(W[1] * new_state[0] + B[1]))))
    for i in range(1, len(self._kernel)):
      new_state = math_ops.matmul(new_state, self._kernel[i])
      new_state = nn_ops.bias_add(new_state, self._bias[i-1])
      new_state = self._activation(new_state)

    if self._use_noise:
      new_state = new_state + tf.random.normal(tf.shape(new_state), dtype=self.dtype)

    return new_state, new_state

  def get_config(self):
    config = {
        "num_units": self._num_units,
        "activation": activations.serialize(self._activation)
    }

    base_config = super(DT_RNNCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


# inputTensor = tf.placeholder(tf.float32, [None, None, 855], name="inputs")
# seqLen = tf.placeholder(tf.float32, [None], name="nVisitsOfEachPatient_List")
#
# rnns = [DT_RNNCell(size) for size in [[271, 271]]]
# cell = tf.nn.rnn_cell.MultiRNNCell(rnns, state_is_tuple=True)
# outputs, states = tf.nn.dynamic_rnn(cell, inputTensor, sequence_length=seqLen, time_major=True, dtype=tf.float32)