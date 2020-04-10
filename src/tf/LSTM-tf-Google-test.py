import pickle
import argparse
import tensorflow as tf
import numpy as np
global ARGS


def numpy_floatX(data):
  return np.asarray(data, dtype=config.floatX)


def prepareHotVectors(test_tensor):
  n_visits_of_each_patientList = np.array([len(seq) for seq in test_tensor]) - 1
  number_of_patients = len(test_tensor)
  max_number_of_visits = np.max(n_visits_of_each_patientList)

  x_hotvectors_tensorf = np.zeros((max_number_of_visits, number_of_patients, ARGS.numberOfInputCodes)).astype(np.float64)
  mask = np.zeros((max_number_of_visits, number_of_patients)).astype(np.float64)

  for idx, (train_patient_matrix) in enumerate(test_tensor):
    for i_th_visit, visit_line in enumerate(train_patient_matrix[:-1]): #ignores the last visit, which is not part of the computation
      for code in visit_line:
        x_hotvectors_tensorf[i_th_visit, idx, code] = 1
    mask[:n_visits_of_each_patientList[idx], idx] = 1.

  return x_hotvectors_tensorf, mask, n_visits_of_each_patientList


def loadModel():
  model = tf.keras.models.load_model(ARGS.modelPath, compile=True)
  # ARGS.numberOfInputCodes = model['W_f0'].shape[0]
  ARGS.numberOfInputCodes = 846
  return model


def load_data():
  testSet_x = np.array(pickle.load(open(ARGS.inputFileRadical+'.test', 'rb')))
  testSet_y = np.array(pickle.load(open(ARGS.inputFileRadical+'.test', 'rb')))

  def len_argsort(seq):
    return sorted(range(len(seq)), key=lambda x: len(seq[x]))

  sorted_index = len_argsort(testSet_x)
  testSet_x = [testSet_x[i] for i in sorted_index]
  testSet_y = [testSet_y[i] for i in sorted_index]

  testSet = [testSet_x, testSet_y]
  return testSet


def testModel():
  print('==> model loading')
  model = loadModel()

  print('==> data loading')
  testSet = load_data()

  print('==> model execution')
  nBatches = int(np.ceil(float(len(testSet[0])) / float(ARGS.batchSize)))
  predictedY_list = []
  actualY_list = []
  for batchIndex in range(nBatches):
    batchX = testSet[0][batchIndex * ARGS.batchSize: (batchIndex + 1) * ARGS.batchSize]
    batchY = testSet[1][batchIndex * ARGS.batchSize: (batchIndex + 1) * ARGS.batchSize]
    xf, mask, nVisitsOfEachPatient_List = prepareHotVectors(batchX)
    
    # if xf.shape[1] == 100: TODO: Arrumar mask e tamanho de entradas
    predicted_y = model.predict(xf)

    for ith_patient in range(predicted_y.shape[1]):
      predictedPatientSlice = predicted_y[:, ith_patient, :]
      actual_y = batchY[ith_patient][1:]
      for ith_admission in range(nVisitsOfEachPatient_List[ith_patient]):
        actualY_list.append(actual_y[ith_admission])

        ithPrediction = predictedPatientSlice[ith_admission]
        enumeratedPrediction = [temp for temp in enumerate(ithPrediction)]
        sortedPrediction_30Top = sorted(enumeratedPrediction, key=lambda x: x[1],reverse=True)[0:30]
        sortedPrediction_30Top_indexes = [temp[0] for temp in sortedPrediction_30Top]
        predictedY_list.append(sortedPrediction_30Top_indexes)

  print('==> computation of prediction results')
  recall_sum = [0.0,0.0,0.0]
  k_list = [10,20,30]
  for ith_admission in range(len(predictedY_list)):
    ithActualY = set(actualY_list[ith_admission])
    for ithK, k in enumerate(k_list):
      ithPredictedY = set(predictedY_list[ith_admission][:k])
      intersection_set = ithActualY.intersection(ithPredictedY)
      recall_sum[ithK] += len(intersection_set) / float(len(ithActualY))
  finalRecalls = []
  for ithK, k in enumerate(k_list):
    finalRecalls.append(recall_sum[ithK] / float(len(predictedY_list)))

  print('Results for Precision@10, Precision@20, and Precision@30')
  print(str(finalRecalls[0]))
  print(str(finalRecalls[1]))
  print(str(finalRecalls[2]))


def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('inputFileRadical', type=str, metavar='<visit_file>', help='File radical name (the software will look for .test file) with pickled data organized as patient x admission x codes.')
  parser.add_argument('modelPath', type=str, metavar='<model_path>', help='The path to the model directory')
  parser.add_argument('--hiddenDimSize', type=str, default='[271]', help='Number of layers and their size - for example [100,200] refers to two layers with 100 and 200 nodes.')
  parser.add_argument('--batchSize', type=int, default=100, help='Batch size.')
  ARGStemp = parser.parse_args()
  hiddenDimSize = [int(strDim) for strDim in ARGStemp.hiddenDimSize[1:-1].split(',')]
  ARGStemp.hiddenDimSize = hiddenDimSize
  return ARGStemp


if __name__ == '__main__':
  global ARGS
  ARGS = parse_arguments()
  print(ARGS)

  testModel()
