import pickle
import argparse
import shutil
import os
import random

import tensorflow.compat.v1 as tf
import tensorflow_addons as tfa

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

  for idx, (train_patient_matrix,label_patient_matrix) in enumerate(zip(train_tensor, labels_tensor)):
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

def performEvaluation(session, loss, x, y, mask, seqLen, test_Set):
  batchSize = ARGS.batchSize

  n_batches = int(np.ceil(float(len(test_Set[0])) / float(batchSize))) #default batch size is 100
  crossEntropySum = 0.0
  dataCount = 0.0
  #computes de crossEntropy for all the elements in the test_Set, using the batch scheme of partitioning
  with session.as_default() as sess:
    for index in range(n_batches):
      batchX = test_Set[0][index * batchSize:(index + 1) * batchSize]
      batchY = test_Set[1][index * batchSize:(index + 1) * batchSize]
      xf, yf, maskf, nVisitsOfEachPatient_List = prepareHotVectors(batchX, batchY)

      # start_token = np.full((xf.shape[0], xf.shape[1], 1), 100)
      # xf = np.concatenate([start_token, xf], axis=-1)
      # yf = np.concatenate([start_token, yf], axis=-1)
      #
      # end_token = np.full((yf.shape[0], yf.shape[1], 1), 200)
      # yf = np.concatenate([yf, end_token], axis=-1)

      feed_dict = {x: xf, y: yf, mask: maskf, seqLen: nVisitsOfEachPatient_List}

      if xf.shape[0] <= 5:
        crossEntropy = sess.run(loss, feed_dict=feed_dict)

      #accumulation by simple summation taking the batch size into account
      crossEntropySum += crossEntropy * len(batchX)
      dataCount += float(len(batchX))
      #At the end, it returns the mean cross entropy considering all the batches
  return n_batches, crossEntropySum / dataCount

def EncoderDecoderBahdanau_layer(inputTensor, targetTensor, vocab, seqLen):
  # Encoder
  with tf.variable_scope('encoder_cell'):
    lstms = [tf.nn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True) for size in ARGS.hiddenDimSize] #According to docs (https://www.tensorflow.org/api_docs/python/tf/compat/v1/nn/rnn_cell/LSTMCell), the peephole version is based on LSTM Google (2014)
    lstms = [tf.nn.rnn_cell.DropoutWrapper(lstm, state_keep_prob=ARGS.dropoutRate) for lstm in lstms]
    cell = tf.nn.rnn_cell.MultiRNNCell(lstms)
    lstm_outputs, lstm_states = tf.nn.dynamic_rnn(cell, inputTensor, sequence_length=seqLen, time_major=True, dtype=tf.float32)

  with tf.variable_scope('decoder_cell'):
    seqLen = tf.cast(seqLen, dtype=tf.int32)
    # Bahdanau Attention
    # query_with_time_axis = tf.expand_dims(lstm_states[-1].h, axis=1)
    # score = tf.keras.layers.Dense(1)(
    #   tf.nn.tanh(
    #     tf.keras.layers.Dense(ARGS.attentionDimSize)(query_with_time_axis) + \
    #     tf.keras.layers.Dense(ARGS.attentionDimSize)(tf.transpose(lstm_outputs, [1, 0, 2]))
    #   )
    # )
    # attention_weights = tf.nn.softmax(score, axis=1)
    # context_vector = attention_weights * tf.transpose(lstm_outputs, [1, 0, 2])
    # context_vector = tf.reduce_sum(context_vector, axis=1)
    #
    # context_vector = tf.expand_dims(context_vector, axis=0)
    # context_vector = tf.tile(context_vector, multiples=[tf.shape(lstm_outputs)[0], 1, 1])
    # lstm_outputs = tf.concat([context_vector, lstm_outputs], axis=-1)

    # Decoder
    lstms = [tf.nn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True) for size in ARGS.hiddenDimSize] #According to docs (https://www.tensorflow.org/api_docs/python/tf/compat/v1/nn/rnn_cell/LSTMCell), the peephole version is based on LSTM Google (2014)
    lstms = [tf.nn.rnn_cell.DropoutWrapper(lstm, state_keep_prob=ARGS.dropoutRate) for lstm in lstms]
    dec_cell = tf.nn.rnn_cell.MultiRNNCell(lstms)

    sampler = tfa.seq2seq.sampler.TrainingSampler(time_major=True)
    sampler.initialize(sequence_length=seqLen)
    decoder = tfa.seq2seq.BasicDecoder(dec_cell, sampler=sampler)

    go_token = -1.
    go_tokens = tf.fill((1, tf.shape(targetTensor)[1], ARGS.numberOfInputCodes), go_token)
    dec_input = tf.concat([go_tokens, targetTensor], axis=0)

    # maximum_iterations=41
    final_outputs, final_state, _ = tfa.seq2seq.dynamic_decode(decoder=decoder, output_time_major=True, decoder_init_input=dec_input,
                                                               decoder_init_kwargs={"initial_state": lstm_states, "sequence_length":seqLen})

    # with tf.variable_scope('decoder_cell', reuse=True):
    #   start_tokens = tf.fill((tf.shape(targetTensor)[1], 1), go_token)
    #   inference_sampler = tfa.seq2seq.sampler.InferenceSampler(
    #     sample_fn=lambda outputs: outputs,
    #     sample_shape=[271],
    #     sample_dtype=tf.float32,
    #     end_fn=lambda sample_ids: False)
    #
    #   inference_sampler.initialize(start_inputs=start_tokens)
    #   inference_decoder = tfa.seq2seq.BasicDecoder(dec_cell, sampler=inference_sampler)
    #
    #   decoder_init_input = tf.fill((100, tf.shape(targetTensor)[-1] + tf.shape(lstm_states[-1].h)[-1]), 0.0)
    #   final_outputs, final_state, _ = tfa.seq2seq.dynamic_decode(decoder=inference_decoder, output_time_major=True, decoder_init_input=decoder_init_input,
    #                                                            decoder_init_kwargs={"initial_state": lstm_states})

    if ARGS.state == "cell":
      return final_state[-1].c #lstm_states has shape (c, h) where c are the cell states and h the hidden states
    elif ARGS.state == "hidden":
      return final_state[-1].h #lstm_states has shape (c, h) where c are the cell states and h the hidden states

# def EncoderDecoderLuong_layer(inputTensor, seqLen):
#   # Encoder
#   with tf.variable_scope('encoder_cell'):
#     lstms = [tf.nn.rnn_cell.LSTMCell(size, state_is_tuple=True, initializer=tf.keras.initializers.glorot_normal()) for size in ARGS.hiddenDimSize] #According to docs (https://www.tensorflow.org/api_docs/python/tf/compat/v1/nn/rnn_cell/LSTMCell), the peephole version is based on LSTM Google (2014)
#     lstms = [tf.nn.rnn_cell.DropoutWrapper(lstm, state_keep_prob=ARGS.dropoutRate) for lstm in lstms]
#     cell = tf.nn.rnn_cell.MultiRNNCell(lstms)
#     lstm_outputs, lstm_states = tf.nn.dynamic_rnn(cell, inputTensor, sequence_length=seqLen, time_major=True, dtype=tf.float32)
#
#   with tf.variable_scope('decoder_cell'):
#     # Decoder
#     lstms = [tf.nn.rnn_cell.BasicLSTMCell(size, state_is_tuple=False) for size in ARGS.hiddenDimSize] #According to docs (https://www.tensorflow.org/api_docs/python/tf/compat/v1/nn/rnn_cell/LSTMCell), the peephole version is based on LSTM Google (2014)
#     lstms = [tf.nn.rnn_cell.DropoutWrapper(lstm, state_keep_prob=ARGS.dropoutRate) for lstm in lstms]
#     cell = tf.nn.rnn_cell.MultiRNNCell(lstms)
#     lstm_dec_outputs, lstm_dec_states = tf.nn.dynamic_rnn(cell, inputTensor, sequence_length=seqLen, time_major=True, initial_state=lstm_states, dtype=tf.float32)
#
#     # Luong Attention
#     # Dot
#     # score = tf.matmul(lstm_outputs, tf.transpose(lstm_dec_outputs, [0, 2, 1]))
#     # attention_weights = tf.nn.softmax(score, axis=0)
#     # context_vector = tf.matmul(attention_weights, lstm_outputs)
#     # output = tf.concat([lstm_dec_outputs, context_vector], axis=-1)
#
#     # General
#     # W = tf.keras.layers.Dense(lstm_dec_outputs.shape[-1], use_bias=False)(lstm_dec_outputs)
#     # score = tf.matmul(lstm_outputs, tf.transpose(W, [0, 2, 1]))
#     # attention_weights = tf.nn.softmax(score, axis=0)
#     # context_vector = tf.matmul(attention_weights, lstm_outputs)
#     # output = tf.concat([lstm_dec_outputs, context_vector], axis=-1)
#
#     # Dot with states
#     score = tf.matmul(lstm_states[-1].c, tf.transpose(lstm_dec_states[-1].c))
#     attention_weights = tf.nn.softmax(score, axis=0)
#     context_vector = tf.matmul(attention_weights, lstm_states[-1].c)
#     output = tf.concat([lstm_dec_states[-1].c, context_vector], axis=-1)
#
#     # General with states
#     # W = tf.keras.layers.Dense(lstm_dec_states[-1].c.shape[-1], use_bias=False)(lstm_dec_states[-1].c)
#     # score = tf.matmul(lstm_states[-1].c, tf.transpose(W))
#     # attention_weights = tf.nn.softmax(score, axis=0)
#     # context_vector = tf.matmul(attention_weights, lstm_states[-1].c)
#     # output = tf.concat([lstm_dec_states[-1].c, context_vector], axis=-1)
#
#     return output

def FC_layer(inputTensor):
  im_dim = inputTensor.get_shape()[-1]
  weights = tf.get_variable(name='weights',
                           shape=[im_dim, ARGS.numberOfInputCodes],
                           dtype=tf.float32,
                           initializer=tf.keras.initializers.glorot_normal())

  bias = tf.get_variable(name='bias',
                       shape=[ARGS.numberOfInputCodes],
                       dtype=tf.float32,
                       initializer=tf.zeros_initializer())
  output = tf.nn.softmax(tf.nn.leaky_relu(tf.add(tf.matmul(inputTensor, weights), bias)))
  return output, weights


def build_model():
  graph = tf.Graph()
  with graph.as_default():
    xf = tf.placeholder(tf.float32, [None, None, ARGS.numberOfInputCodes], name="inputs")
    yf = tf.placeholder(tf.float32, [None, None, ARGS.numberOfInputCodes], name="labels")
    df = tf.placeholder(tf.float32, [ARGS.numberOfInputCodes, ARGS.hiddenDimSize[-1]], name="vocab_size")
    maskf = tf.placeholder(tf.float32, [None, None], name="mask")
    seqLen = tf.placeholder(tf.float32, [None], name="nVisitsOfEachPatient_List")

    with tf.device('/gpu:0'):
      flowingTensor = EncoderDecoderBahdanau_layer(xf, yf, df, seqLen)
      flowingTensor, weights = FC_layer(flowingTensor)
      flowingTensor = tf.math.multiply(flowingTensor, maskf[:,:,None], name="predictions")

      epislon = 1e-8
      cross_entropy = -(yf * tf.log(flowingTensor + epislon) + (1. - yf) * tf.log(1. - flowingTensor + epislon))
      prediction_loss = tf.math.reduce_mean(tf.math.reduce_sum(cross_entropy, axis=[2, 0]) / seqLen)
      L2_regularized_loss = prediction_loss + tf.math.reduce_sum(ARGS.LregularizationAlpha * (weights ** 2))

      optimizer = tf.train.AdadeltaOptimizer(learning_rate=ARGS.learningRate, rho=0.95, epsilon=1e-06).minimize(L2_regularized_loss)

    # Bahdanau (855)
    # global_step = tf.Variable(0, trainable=False)
    # learning_rate = tf.train.exponential_decay(1.0, global_step, 1000, 0.68)
    # optimizer = tf.train.AdadeltaOptimizer(learning_rate, rho=0.95, epsilon=1e-06).minimize(L2_regularized_loss, global_step=global_step)

    # Bahdanau (271)
    # global_step = tf.Variable(0, trainable=False)
    # learning_rate = tf.train.exponential_decay(1.0, global_step, 100, 0.7)
    # optimizer = tf.train.AdadeltaOptimizer(learning_rate, rho=0.95, epsilon=1e-06).minimize(L2_regularized_loss, global_step=global_step)

    # Luong
    # global_step = tf.Variable(0, trainable=False)
    # learning_rate = tf.train.exponential_decay(1.0, global_step, 100, 0.9)
    # optimizer = tf.train.AdadeltaOptimizer(learning_rate, rho=0.95, epsilon=1e-06).minimize(L2_regularized_loss, global_step=global_step)

    return tf.global_variables_initializer(), graph, optimizer, L2_regularized_loss, xf, yf, maskf, seqLen, flowingTensor

def train_model():
  print("==> data loading")
  trainSet, testSet = load_data()

  print("==> model building")
  init, graph, optimizer, loss, x, y, mask, seqLen, predictions = build_model()

  print ("==> training and validation")
  batchSize = ARGS.batchSize
  n_batches = int(np.ceil(float(len(trainSet[0])) / float(batchSize)))

  bestValidationCrossEntropy = 1e20
  bestValidationEpoch = 0
  bestModelDirName = ''

  iImprovementEpochs = 0
  iConsecutiveNonImprovements = 0
  epoch_counter = 0

  with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(init)

    for epoch_counter in range(ARGS.nEpochs):
      iteration = 0
      trainCrossEntropyVector = []
      for index in random.sample(range(n_batches), n_batches):
        batchX = trainSet[0][index*batchSize:(index+1)*batchSize]
        batchY = trainSet[1][index*batchSize:(index + 1)*batchSize]
        xf, yf, maskf, nVisitsOfEachPatient_List = prepareHotVectors(batchX, batchY)
        xf += np.random.normal(0, 0.1, xf.shape)

        # start_token = np.full((xf.shape[0], xf.shape[1], 1), 100)
        # # xf = np.concatenate([start_token, xf], axis=-1)
        # yf = np.concatenate([start_token, yf], axis=-1)
        #
        # end_token = np.full((yf.shape[0], yf.shape[1], 1), 200)
        # yf = np.concatenate([yf, end_token], axis=-1)

        feed_dict = {x: xf, y: yf, mask: maskf, seqLen: nVisitsOfEachPatient_List}

        if xf.shape[0] <= 5:
          _, trainCrossEntropy = sess.run([optimizer, loss], feed_dict=feed_dict)

        trainCrossEntropyVector.append(trainCrossEntropy)
        iteration += 1

      print('-> Epoch: %d, mean cross entropy considering %d TRAINING batches: %f' % (epoch_counter, n_batches, np.mean(trainCrossEntropyVector)))
      nValidBatches, validationCrossEntropy = performEvaluation(sess, loss, x, y, mask, seqLen, testSet)
      print('      mean cross entropy considering %d VALIDATION batches: %f' % (nValidBatches, validationCrossEntropy))

      if validationCrossEntropy < bestValidationCrossEntropy:
        iImprovementEpochs += 1
        iConsecutiveNonImprovements = 0
        bestValidationCrossEntropy = validationCrossEntropy
        bestValidationEpoch = epoch_counter

        if os.path.exists(bestModelDirName):
          shutil.rmtree(bestModelDirName)
        bestModelDirName = ARGS.outFile + '.' + str(epoch_counter)

        if os.path.exists(bestModelDirName):
          shutil.rmtree(bestModelDirName)

        signature = tf.saved_model.signature_def_utils.predict_signature_def(inputs= {"inputs": x, "labels": y, "mask": mask, "seqLen": seqLen}, outputs= {"predictions": predictions})
        builder = tf.saved_model.builder.SavedModelBuilder(bestModelDirName)
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], signature_def_map={'model': signature})
        builder.save()

      else:
        print('Epoch ended without improvement.')
        iConsecutiveNonImprovements += 1
      if iConsecutiveNonImprovements > ARGS.maxConsecutiveNonImprovements: #default is 10
        break

    # Best results
    print('--------------SUMMARY--------------')
    print('The best VALIDATION cross entropy occurred at epoch %d, the value was of %f ' % (
    bestValidationEpoch, bestValidationCrossEntropy))
    print('Best model file: ' + bestModelDirName)
    print('Number of improvement epochs: ' + str(iImprovementEpochs) + ' out of ' + str(epoch_counter + 1) + ' possible improvements.')
    print('Note: the smaller the cross entropy, the better.')
    print('-----------------------------------')
    sess.close()


def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('inputFileRadical', type=str, metavar='<visit_file>', help='File radical name (the software will look for .train and .test files) with pickled data organized as patient x admission x codes.')
  parser.add_argument('outFile', metavar='out_file', default='model_output', help='Any file directory to store the model.')
  parser.add_argument('--maxConsecutiveNonImprovements', type=int, default=10, help='Training wiil run until reaching the maximum number of epochs without improvement before stopping the training')
  parser.add_argument('--hiddenDimSize', type=str, default='[1084]', help='Number of layers and their size - for example [100,200] refers to two layers with 100 and 200 nodes.')
  parser.add_argument('--state', type=str, default='cell', help='Pass cell or hidden to fully connected layer')
  # parser.add_argument('--attentionDimSize', type=int, default=3, help='Number of attention layer dense units')
  parser.add_argument('--batchSize', type=int, default=100, help='Batch size.')
  parser.add_argument('--nEpochs', type=int, default=1000, help='Number of training iterations.')
  parser.add_argument('--LregularizationAlpha', type=float, default=0.001, help='Alpha regularization for L2 normalization')
  parser.add_argument('--learningRate', type=float, default=0.5, help='Learning rate.')
  parser.add_argument('--dropoutRate', type=float, default=0.45, help='Dropout probability.')


  ARGStemp = parser.parse_args()
  hiddenDimSize = [int(strDim) for strDim in ARGStemp.hiddenDimSize[1:-1].split(',')]
  ARGStemp.hiddenDimSize = hiddenDimSize
  return ARGStemp

if __name__ == '__main__':
  global ARGS
  ARGS = parse_arguments()

  train_model()
