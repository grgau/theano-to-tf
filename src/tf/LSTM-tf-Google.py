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

def init_tensors():
  x = tf.placeholder(tf.float32, [None, None, ARGS.numberOfInputCodes])
  y = tf.placeholder(tf.float32, [None, None, ARGS.numberOfInputCodes])
  mask = tf.placeholder(tf.float32, [None, None, ARGS.numberOfInputCodes])
  return x, y, mask

def LSTMGoogle_layer(inputTensor):
  layer = {"weights_lstm": tf.Variable(tf.random_normal([ARGS.hiddenDimSize[0], ARGS.numberOfInputCodes])),
           "biases_lstm": tf.Variable(tf.random_normal([ARGS.numberOfInputCodes]))}

  # def weight_variable(name, shape):
  #     initial = tf.truncated_normal_initializer(stddev=0.01)
  #     return tf.get_variable('W_' + name,
  #                            dtype=tf.float32,
  #                            shape=shape,
  #                            initializer=initer)

  # def bias_variable(name, shape):
  #     initial = tf.constant(0., shape=shape, dtype=tf.float32)
  #     return tf.get_variable('b_' + name,
  #                            dtype=tf.float32,
  #                            initializer=initial)

  # W = weight_variable(name, shape=[in_dim, ARGS.numberOfInputCodes])
  # b = bias_variable(name, [ARGS.numberOfInputCodes])
  lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(ARGS.hiddenDimSize[0])
  outputs, states = tf.nn.dynamic_rnn(lstm_cell, inputTensor, dtype=tf.float32)

  output = tf.matmul(outputs[-1], layer["weights_lstm"]) + layer["biases_lstm"]

  return output

def FC_layer(inputTensor, name):
  def weight_variable(name, shape):
      initial = tf.truncated_normal_initializer(stddev=0.01)
      return tf.get_variable('W_' + name,
                             dtype=tf.float32,
                             shape=shape,
                             initializer=initial)

  def bias_variable(name, shape):
      initial = tf.constant(0., shape=shape, dtype=tf.float32)
      return tf.get_variable('b_' + name,
                             dtype=tf.float32,
                             initializer=initial)

  in_dim = inputTensor.get_shape()[1]
  W = weight_variable(name, shape=[in_dim, ARGS.numberOfInputCodes])
  b = bias_variable(name, [ARGS.numberOfInputCodes])
  output = tf.matmul(inputTensor, W)
  output += b
  output = tf.nn.relu(output)
  return output


def build_model():
  inputTensor, predictionTensor, maskTensor = init_tensors()
  lstm_predictions = LSTMGoogle_layer(inputTensor)
  predictions = FC_layer(lstm_predictions, "FC1")
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=predictionTensor))
  optimizer = tf.train.AdadeltaOptimizer().minimize(loss)
  return optimizer, loss, inputTensor, predictionTensor

def train_model():
  print('==> data loading')
  trainSet, testSet = load_data()
  previousDimSize = ARGS.numberOfInputCodes

  print("==> model building")
  optimizer, cost, x, y = build_model()

  print ("==> training and validation")
  batchSize = ARGS.batchSize
  n_batches = int(np.ceil(float(len(trainSet[0])) / float(batchSize)))

  bestValidationCrossEntropy = 1e20
  bestValidationEpoch = 0
  bestModelFileName = ''

  iImprovementEpochs = 0
  iConsecutiveNonImprovements = 0
  epoch_counter = 0

  init = (tf.global_variables_initializer(), tf.local_variables_initializer())
  with tf.Session() as sess:
    sess.run(init)

    for epoch_counter in range(ARGS.nEpochs):
      iteration = 0
      trainCrossEntropyVector = []
      for index in random.sample(range(n_batches), n_batches):
        batchX = trainSet[0][index*batchSize:(index+1)*batchSize]
        batchY = trainSet[1][index*batchSize:(index + 1)*batchSize]
        xf, yf, mask, nVisitsOfEachPatient_List = prepareHotVectors(batchX, batchY)
        xf += np.random.normal(0, 0.1, xf.shape)

        if xf.shape[0] == 1: # its not learning sequences of data, in high level too
          _, trainCrossEntropy = sess.run([optimizer, cost], feed_dict={x: xf, y: yf})
          trainCrossEntropyVector.append(trainCrossEntropy)

        iteration += 1
      print('-> Epoch: %d, mean cross entropy considering %d TRAINING batches: %f' % (epoch_counter, n_batches, np.mean(trainCrossEntropyVector)))


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