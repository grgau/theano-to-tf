from tensorflow.python.ops import math_ops, array_ops, nn_ops, init_ops
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras import activations

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class DT_RNNCell(tf.nn.rnn_cell.BasicRNNCell):
  def __init__(self, num_units, activation=None):
    super(DT_RNNCell, self).__init__(num_units=num_units)

    # Inputs must be 2-dimensional.
    self.input_spec = input_spec.InputSpec(ndim=2)
    self._num_units = num_units

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
    # self._kernel = self.add_variable(_WEIGHTS_VARIABLE_NAME, shape=[input_depth + self._num_units[0], self._num_units[0], self._num_units[1]])
    # self._bias = self.add_variable(_BIAS_VARIABLE_NAME, shape=[self._num_units[1]], initializer=init_ops.zeros_initializer(dtype=self.dtype))

    self._kernel = [self.add_variable(_WEIGHTS_VARIABLE_NAME+"_0", shape=[input_depth + self._num_units[0], self._num_units[0]], initializer=tf.keras.initializers.glorot_normal()), self.add_variable(_WEIGHTS_VARIABLE_NAME+"_1", shape=[self._num_units[0], self._num_units[1]], initializer=tf.keras.initializers.glorot_normal())]
    self._bias = [self.add_variable(_BIAS_VARIABLE_NAME+"_0", shape=[self._num_units[0]], initializer=init_ops.zeros_initializer(dtype=self.dtype))]
    self.built = True

  def call(self, inputs, state):
    """Most basic RNN: output = new_state = act(W * input + U * state + B)."""
    gate_inputs = array_ops.concat([inputs, state], 1)
    gate_inputs = math_ops.matmul(gate_inputs, self._kernel[0])
    # gate_inputs = tf.tensordot(gate_inputs, self._kernel, axes=[[1],[0]])
    # gate_inputs = nn_ops.bias_add(gate_inputs, self._bias[0])
    # output = self._activation(gate_inputs)
    # return output[:, :, -1], output[:, :, -1]

    output = math_ops.matmul(gate_inputs, self._kernel[1])
    output = nn_ops.bias_add(output, self._bias[0])
    output = self._activation(output)

    # output = math_ops.matmul(output, self._kernel[2])
    # output = nn_ops.bias_add(output, self._bias[1])
    # output = self._activation(output)
    return output, output

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