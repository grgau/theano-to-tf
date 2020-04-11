import pickle
import argparse
from collections import OrderedDict
import random

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

global ARGS

def getNumberOfCodes(sets):
  highestCode = 0
  for set in sets:
    for pat in set:
      for adm in pat:
        for code in adm:
          if code > highestCode:
            highestCode = code
  return (highestCode + 1)


def prepareHotVectors(train_tensor, labels_tensor):
  nVisitsOfEachPatient_List = np.array([len(seq) for seq in train_tensor]) - 1
  numberOfPatients = len(train_tensor)
  maxNumberOfAdmissions = np.max(nVisitsOfEachPatient_List)

  x_hotvectors_tensorf = np.zeros((maxNumberOfAdmissions, numberOfPatients, ARGS.numberOfInputCodes)).astype(np.float64)
  y_hotvectors_tensor = np.zeros((maxNumberOfAdmissions, numberOfPatients, ARGS.numberOfInputCodes)).astype(np.float64)
  mask = np.zeros((maxNumberOfAdmissions, numberOfPatients)).astype(np.float64)

  for idx, (train_patient_matrix,label_patient_matrix) in enumerate(zip(train_tensor,labels_tensor)):
    for i_th_visit, visit_line in enumerate(train_patient_matrix[:-1]): #ignores the last admission, which is not part of the training
      for code in visit_line:
        x_hotvectors_tensorf[i_th_visit, idx, code] = 1
    for i_th_visit, visit_line in enumerate(label_patient_matrix[1:]):  #label_matrix[1:] = all but the first admission slice, not used to evaluate (this is the answer)
      for code in visit_line:
        y_hotvectors_tensor[i_th_visit, idx, code] = 1
    mask[:nVisitsOfEachPatient_List[idx], idx] = 1.

  nVisitsOfEachPatient_List = np.array(nVisitsOfEachPatient_List, dtype=np.int32)
  return x_hotvectors_tensorf, y_hotvectors_tensor, mask, nVisitsOfEachPatient_List

# weight and bais wrappers
def weight_variable(shape):
    """
    Create a weight variable with appropriate initialization
    :param name: weight name
    :param shape: weight shape
    :return: initialized weight variable
    """
    initer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W',
                           dtype=tf.float32,
                           shape=shape,
                           initializer=initer)

def bias_variable(shape):
    """
    Create a bias variable with appropriate initialization
    :param name: bias variable name
    :param shape: bias variable shape
    :return: initialized bias variable
    """
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b',
                           dtype=tf.float32,
                           initializer=initial)


def RNN(x, weights, biases, n_hidden, seq_max_len, seq_len):
    """
    :param x: inputs of shape [batch_size, max_time, input_dim]
    :param weights: matrix of fully-connected output layer weights
    :param biases: vector of fully-connected output layer biases
    :param n_hidden: number of hidden units
    :param seq_max_len: sequence maximum length
    :param seq_len: length of each sequence of shape [batch_size,]
    """
    print(x, weights, biases, n_hidden, seq_max_len, seq_len)
    cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    outputs, states = tf.nn.dynamic_rnn(cell, x, sequence_length=seq_len, dtype=tf.float32)
    print(outputs, states)

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seq_len - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)
    out = tf.matmul(outputs, weights) + biases
    return out


def load_data():
  main_trainSet = pickle.load(open(ARGS.inputFileRadical+'.train', 'rb'))
  print("-> " + str(len(main_trainSet)) + " patients at dimension 0 for file: "+ ARGS.inputFileRadical + ".train dimensions ")
  main_testSet = pickle.load(open(ARGS.inputFileRadical+'.test', 'rb'))
  print("-> " + str(len(main_testSet)) + " patients at dimension 0 for file: "+ ARGS.inputFileRadical + ".test dimensions ")
  print("Note: these files carry 3D tensor data; the above numbers refer to dimension 0, dimensions 1 and 2 have irregular sizes.")

  ARGS.numberOfInputCodes = getNumberOfCodes([main_trainSet,main_testSet])
  print('Number of diagnosis input codes: ' + str(ARGS.numberOfInputCodes))

  #uses the same data for testing, but disregarding the fist admission of each patient
  labels_trainSet = pickle.load(open(ARGS.inputFileRadical+'.train', 'rb'))
  labels_testSet = pickle.load(open(ARGS.inputFileRadical+'.test', 'rb'))

  train_sorted_index = sorted(range(len(main_trainSet)), key=lambda x: len(main_trainSet[x]))  #lambda x: len(seq[x]) --> f(x) return len(seq[x])
  main_trainSet = [main_trainSet[i] for i in train_sorted_index]
  labels_trainSet = [labels_trainSet[i] for i in train_sorted_index]

  test_sorted_index = sorted(range(len(main_testSet)), key=lambda x: len(main_testSet[x]))
  main_testSet = [main_testSet[i] for i in test_sorted_index]
  labels_testSet = [labels_testSet[i] for i in test_sorted_index]

  trainSet = [main_trainSet, labels_trainSet]
  testSet = [main_testSet, labels_testSet]

  return trainSet, testSet


def train_model():
  print('==> data loading')
  trainSet, testSet = load_data()
  previousDimSize = ARGS.numberOfInputCodes

  print("==> parameters initialization")
  input_dim = 846           # input dimension
  seq_max_len = 30         # sequence maximum length
  out_dim = 846             # output dimension
  num_hidden_units = 271   # number of hidden units
  training_steps = 10000  # Total number of training steps
  learning_rate = 0.001 # The optimization initial learning rate
  epochs = 10           # Total number of training epochs
  batch_size = 100      # Training batch size
  display_freq = 100    # Frequency of displaying the training results

  x = tf.placeholder(tf.float32, [None, None, input_dim])
  seqLen = tf.placeholder(tf.int32, [None])
  y = tf.placeholder(tf.float32, [None, None, out_dim])

  W = weight_variable(shape=[num_hidden_units, out_dim])
  b = bias_variable(shape=[out_dim])

  print("==> model building")
  pred_out = RNN(x, W, b, num_hidden_units, seq_max_len, seqLen)
  cost = tf.reduce_mean(tf.square(pred_out - y))
  train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
  init = tf.global_variables_initializer()

  print("==> training and validation")
  batchSize = ARGS.batchSize
  n_batches = int(np.ceil(float(len(trainSet[0])) / float(batchSize)))

  bestValidationCrossEntropy = 1e20
  bestValidationEpoch = 0
  bestModelFileName = ''

  iImprovementEpochs = 0
  iConsecutiveNonImprovements = 0
  epoch_counter = 0
  for epoch_counter in range(ARGS.nEpochs):
    iteration = 0
    trainCrossEntropyVector = []

    with tf.Session() as sess:
      sess.run(init)

      for index in random.sample(range(n_batches), n_batches):
        batchX = trainSet[0][index*batchSize:(index+1)*batchSize]
        batchY = trainSet[1][index*batchSize:(index + 1)*batchSize]
        xf, yf, mask, nVisitsOfEachPatient_List = prepareHotVectors(batchX, batchY)
        xf += np.random.normal(0, 0.1, xf.shape)  #add gaussian noise as a means to reduce overfitting

        feed_dict = {x: xf}
        # print the shape
        X_seq_shape = sess.run(tf.shape(xf), feed_dict=feed_dict)
        output_shape = sess.run(tf.shape(output), feed_dict=feed_dict)
        state_shape = sess.run(tf.shape(state), feed_dict=feed_dict)
        output_st_shape = sess.run(tf.shape(output_st), feed_dict=feed_dict)
        print('X_seq shape [batch_size, n_steps, n_inputs]: ', X_seq_shape)
        print('output shape [batch_size, n_neurons]: ', output_shape)
        print('state shape [batch_size, n_neurons]: ', state_shape)
        print('output_st shape [batch_size, n_steps, n_neurons]: ', output_st_shape)

        output_eval, state_eval = sess.run([output, state], feed_dict=feed_dict)
        print('Is the output of X2 equals to the state?', np.array_equal(output_eval[2], state_eval))
        # _, trainCrossEntropy = sess.run([train_op, cost], feed_dict={x: xf, y: yf, seqLen: nVisitsOfEachPatient_List})
        # trainCrossEntropyVector.append(trainCrossEntropy)
        # iteration += 1

    print("-> Epoch: %d, mean cross entropy considering %d TRAINING batches: %f" % (epoch_counter, n_batches, np.mean(trainCrossEntropyVector)))



def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('inputFileRadical', type=str, metavar='<visit_file>', help='File radical name (the software will look for .train and .test files) with pickled data organized as patient x admission x codes.')
  parser.add_argument('outFile', metavar='out_file', default='model_output', help='Any file name to store the model.')
  parser.add_argument('--maxConsecutiveNonImprovements', type=int, default=5, help='Training wiil run until reaching the maximum number of epochs without improvement before stopping the training')
  parser.add_argument('--hiddenDimSize', type=str, default='[271]', help='Number of layers and their size - for example [100,200] refers to two layers with 100 and 200 nodes.')
  parser.add_argument('--batchSize', type=int, default=100, help='Batch size.')
  parser.add_argument('--nEpochs', type=int, default=1000, help='Number of training iterations.')
  parser.add_argument('--LregularizationAlpha', type=float, default=0.001, help='Alpha regularization for L2 normalization')
  parser.add_argument('--dropoutRate', type=float, default=0.45, help='Dropout probability.')

  ARGStemp = parser.parse_args()
  hiddenDimSize = [int(strDim) for strDim in ARGStemp.hiddenDimSize[1:-1].split(',')]
  ARGStemp.hiddenDimSize = hiddenDimSize
  return ARGStemp

if __name__ == '__main__':
  global tPARAMS
  tPARAMS = OrderedDict()
  global ARGS
  ARGS = parse_arguments()

  train_model()