import os
import random
import pickle
import shutil
import argparse
import tensorflow as tf
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

  nVisitsOfEachPatient_List = np.array(nVisitsOfEachPatient_List, dtype=np.float64)
  return x_hotvectors_tensorf, y_hotvectors_tensor, mask, nVisitsOfEachPatient_List

def build_model():
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.InputLayer(input_shape=(None, ARGS.numberOfInputCodes)))

  for layer in range(len(ARGS.hiddenDimSize)):
    model.add(tf.keras.layers.LSTM(ARGS.hiddenDimSize[layer], return_sequences=True, dropout=ARGS.dropoutRate, dtype="float64"))

  model.add(tf.keras.layers.Dense(ARGS.numberOfInputCodes, activation="relu", activity_regularizer=tf.keras.regularizers.l2(ARGS.LregularizationAlpha), dtype="float64"))
  model.add(tf.keras.layers.Activation("softmax", dtype="float64"))

  model.compile(optimizer="Adadelta", loss="categorical_crossentropy")
  print(model.summary())

  # inputs  = tf.keras.Input(shape=(None, ARGS.numberOfInputCodes))
  # lstm    = tf.keras.layers.LSTM(ARGS.hiddenDimSize[0], return_sequences=True, dtype="float64", dropout=ARGS.dropoutRate)(inputs)
  # dense   = tf.keras.layers.Dense(ARGS.numberOfInputCodes, activation="relu", dtype="float64")(lstm)
  # output  = tf.keras.layers.Activation("softmax", dtype="float64")(dense)
  # model   = tf.keras.models.Model(inputs, output)
  # print(model.summary())
  # model.compile(optimizer="Adadelta", loss="categorical_crossentropy")

  return model


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


#the performance computation uses the test data and returns the cross entropy measure
def performEvaluation(test_model, test_Set):
  batchSize = ARGS.batchSize

  n_batches = int(np.ceil(float(len(test_Set[0])) / float(batchSize))) #default batch size is 100
  crossEntropySum = 0.0
  dataCount = 0.0
  #computes de crossEntropy for all the elements in the test_Set, using the batch scheme of partitioning
  for index in range(n_batches):
    batchX = test_Set[0][index * batchSize:(index + 1) * batchSize]
    batchY = test_Set[1][index * batchSize:(index + 1) * batchSize]
    xf, y, mask, nVisitsOfEachPatient_List = prepareHotVectors(batchX, batchY)
    mask = tf.convert_to_tensor(mask, dtype=bool)
    xf = tf.convert_to_tensor(xf, dtype="float64")
    y = tf.convert_to_tensor(y, dtype="float64")
    test_model.compute_mask(inputs=xf, mask=mask)
    crossEntropy = test_model.fit(x=xf, y=y, verbose=0)

    #accumulation by simple summation taking the batch size into account
    crossEntropySum += crossEntropy.history['loss'][0] * len(batchX)
    dataCount += float(len(batchX))
    #At the end, it returns the mean cross entropy considering all the batches
  return n_batches, crossEntropySum / dataCount


def train_model():
  print('==> data loading')
  trainSet, testSet = load_data()
  print('==> model building')
  model = build_model()

  print('==> training and validation')
  batchSize = ARGS.batchSize
  n_batches = int(np.ceil(float(len(trainSet[0])) / float(batchSize)))
  test_model = build_model()

  bestValidationCrossEntropy = 1e20
  bestValidationEpoch = 0
  bestModelFileName = ''

  iImprovementEpochs = 0
  iConsecutiveNonImprovements = 0
  epoch_counter = 0

  for epoch_counter in range(ARGS.nEpochs):
    iteration = 0
    trainCrossEntropyVector = []
    for index in random.sample(range(n_batches), n_batches):
      batchX = trainSet[0][index*batchSize:(index+1)*batchSize]
      batchY = trainSet[1][index*batchSize:(index+1)*batchSize]
      xf, y, mask, nVisitsOfEachPatient_List = prepareHotVectors(batchX, batchY)
      xf += np.random.normal(0, 0.1, xf.shape)  #add gaussian noise as a means to reduce overfitting
      mask = tf.convert_to_tensor(mask, dtype=bool)
      xf = tf.convert_to_tensor(xf, dtype="float64")
      y = tf.convert_to_tensor(y, dtype="float64")
      model.compute_mask(inputs=xf, mask=mask)
      trainCrossEntropy = model.fit(x=xf, y=y, verbose=0)

      trainCrossEntropyVector.append(trainCrossEntropy.history['loss'][0])
      iteration += 1

    print('-> Epoch: %d, mean cross entropy considering %d TRAINING batches: %f' % (epoch_counter, n_batches, np.mean(trainCrossEntropyVector)))
    nValidBatches, validationCrossEntropy = performEvaluation(test_model, testSet)
    print('      mean cross entropy considering %d VALIDATION batches: %f' % (nValidBatches, validationCrossEntropy))
    if validationCrossEntropy < bestValidationCrossEntropy:
      iImprovementEpochs += 1
      iConsecutiveNonImprovements = 0
      bestValidationCrossEntropy = validationCrossEntropy
      bestValidationEpoch = epoch_counter

      if os.path.exists(bestModelFileName):
        print(bestModelFileName)
        shutil.rmtree(bestModelFileName, ignore_errors=True)
      bestModelFileName = ARGS.outFile + '.' + str(epoch_counter) + '/'
      tf.keras.models.save_model(model, bestModelFileName)

    else:
      print('Epoch ended without improvement.')
      iConsecutiveNonImprovements += 1
    if iConsecutiveNonImprovements > ARGS.maxConsecutiveNonImprovements: #default is 10
      break
  #Best results
  print('--------------SUMMARY--------------')
  print('The best VALIDATION cross entropy occurred at epoch %d, the value was of %f ' % (bestValidationEpoch, bestValidationCrossEntropy))
  print('Best model file: ' + bestModelFileName)
  print('Number of improvement epochs: ' + str(iImprovementEpochs) + ' out of ' + str(epoch_counter+1) + ' possible improvements.')
  print('Note: the smaller the cross entropy, the better.')
  print('-----------------------------------')


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
  global ARGS
  ARGS = parse_arguments()

  train_model()