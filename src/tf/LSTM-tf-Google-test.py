import pickle
import argparse
import os
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# import tensorflow as tf
import numpy as np

global ARGS

# def strenght(inter, normalized_inter, adm_i, adm_j):
#   intersect = len(list(set(adm_i).intersection(adm_j)))
#   intersect_norm = np.float64(intersect) / np.float64(min(len(adm_i), len(adm_j)))
#   # return intersect_norm if inter == 0 else intersect / ((np.log(inter) + 1) ** intersect_norm)
#   # return intersect_norm if inter==0 or intersect_norm==0 else intersect/(inter**((normalized_inter*np.log(inter))/5*intersect_norm))
#   return intersect_norm if inter == 0 else intersect / ((np.log(inter) + 1) ** (1/5*intersect_norm))

# def generate_strenghts(dataset, interval):
#   normalized_strs = []
#   normalized_intervals = []
#
#   for inter in interval:
#     normalized_intervals.extend(inter[1:])
#
#   min_interval, max_interval = min(normalized_intervals), max(normalized_intervals)
#   for i, val in enumerate(normalized_intervals):
#     normalized_intervals[i] = 0 if (max_interval-min_interval) == 0 else (val-min_interval) / (max_interval-min_interval)
#
#   for t, inter in zip(dataset, interval):
#     for i in range(1, len(t)):
#       normalized_strs.append(strenght(inter[i], normalized_intervals[i], t[i-1], t[i]))
#
#   min_str, max_str = min(normalized_strs), max(normalized_strs)
#   for i, val in enumerate(normalized_strs):
#     normalized_strs[i] = 0 if (max_str-min_str) == 0 else (val-min_str) / (max_str-min_str)
#   return normalized_strs

# def split_by_strenght(dataset, interval, threshold):
#   str_counter = 0
#   strenghts = generate_strenghts(dataset, interval)
#   new_dataset = []
#   for t, inter in zip(dataset, interval):
#     split_count = 0
#     arrays = [[t[0]]]
#     for i in range(1, len(t)):
#       if len(arrays[split_count]) > 1 and len(t)-1 > i and strenghts[str_counter] < threshold: #len(t)-1 is for checking if last element in array, to not split
#         arrays.append([t[i]])
#         split_count += 1
#       else:
#         arrays[len(arrays) - 1].append(t[i])
#       str_counter += 1
#     new_dataset.extend(arrays)
#   return new_dataset

def prepareHotVectors(test_tensor, intervals_tensor):
  n_visits_of_each_patientList = np.array([len(seq) for seq in test_tensor]) - 1
  number_of_patients = len(test_tensor)
  max_number_of_visits = np.max(n_visits_of_each_patientList)

  x_hotvectors_tensorf = np.zeros((max_number_of_visits, number_of_patients, ARGS.numberOfInputCodes)).astype(np.float64)
  i_hotvectors_tensor = np.zeros((max_number_of_visits, number_of_patients, ARGS.numberOfInputCodes)).astype((np.float64))
  mask = np.zeros((max_number_of_visits, number_of_patients)).astype(np.float64)

  for idx, (train_patient_matrix, interval_patient_matrix) in enumerate(zip(test_tensor, intervals_tensor)):
    for i_th_visit, visit_line in enumerate(train_patient_matrix[:-1]): #ignores the last visit, which is not part of the computation
      for code in visit_line:
        x_hotvectors_tensorf[i_th_visit, idx, code] = 1
    if isinstance(interval_patient_matrix, list):
      for i_th_visit, visit_line in enumerate(interval_patient_matrix[1:]):
        i_hotvectors_tensor[i_th_visit, idx, :] = visit_line
    mask[:n_visits_of_each_patientList[idx], idx] = 1.

  return x_hotvectors_tensorf, i_hotvectors_tensor, mask, n_visits_of_each_patientList

def loadModel():
  model_path = ARGS.modelPath

  loaded_graph = tf.Graph()
  with tf.Session(graph=loaded_graph).as_default() as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_path)
    x = loaded_graph.get_tensor_by_name('inputs:0')
    predictions = loaded_graph.get_tensor_by_name('predictions:0')
    inter = loaded_graph.get_tensor_by_name('intervals:0')
    mask = loaded_graph.get_tensor_by_name('mask:0')
    seqLen = loaded_graph.get_tensor_by_name('nVisitsOfEachPatient_List:0')

    ARGS.numberOfInputCodes = x.get_shape()[-1]
    return sess, predictions, x, inter, mask, seqLen


def load_data():
  testSet_x = np.array(pickle.load(open(ARGS.inputFileRadical+'.test', 'rb')))
  testSet_y = np.array(pickle.load(open(ARGS.inputFileRadical+'.test', 'rb')))
  testIntervalSet = np.array(pickle.load(open(ARGS.inputFileRadical+'.INTERVAL.test', 'rb')))

  def len_argsort(seq):
    return sorted(range(len(seq)), key=lambda x: len(seq[x]))

  sorted_index = len_argsort(testSet_x)
  testSet_x = [testSet_x[i] for i in sorted_index]
  testSet_y = [testSet_y[i] for i in sorted_index]
  testIntervalSet = [testIntervalSet[i] for i in sorted_index]

  testSet = [testSet_x, testSet_y]
  return testSet, testIntervalSet

def testModel():
  # threshold = ARGS.strengthThreshold
  print('==> model loading')
  session, predictions, x, inter, mask, seqLen = loadModel()

  print('==> data loading')
  testSet, testIntervalSet = load_data()

  print('==> model execution')
  nBatches = int(np.ceil(float(len(testSet[0])) / float(ARGS.batchSize)))
  predictedY_list = []
  actualY_list = []

  with session as sess:
    for batchIndex in range(nBatches):
      batchX = testSet[0][batchIndex * ARGS.batchSize: (batchIndex + 1) * ARGS.batchSize]
      batchY = testSet[1][batchIndex * ARGS.batchSize: (batchIndex + 1) * ARGS.batchSize]
      batchInterval = testIntervalSet[batchIndex * ARGS.batchSize: (batchIndex + 1) * ARGS.batchSize]
      xf, interf, maskf, nVisitsOfEachPatient_List = prepareHotVectors(batchX, batchInterval)

      predicted_y = sess.run(predictions, feed_dict={x: xf, inter: interf, mask: maskf, seqLen: nVisitsOfEachPatient_List})

      # splittedTestSet_0 = split_by_strenght(batchX, batchInterval, threshold)
      # splittedTestSet_1 = split_by_strenght(batchY, batchInterval, threshold)
      #
      # testSet[0].extend(splittedTestSet_0)
      # testSet[1].extend(splittedTestSet_1)

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

  print('Results for Recall@10, Recall@20, and Recall@30')
  print(str(finalRecalls[0]))
  print(str(finalRecalls[1]))
  print(str(finalRecalls[2]))
  sess.close()


def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('inputFileRadical', type=str, metavar='<visit_file>', help='File radical name (the software will look for .test file) with pickled data organized as patient x admission x codes.')
  parser.add_argument('modelPath', type=str, metavar='<model_path>', help='The path to the model directory')
  parser.add_argument('--hiddenDimSize', type=str, default='[271]', help='Number of layers and their size - for example [100,200] refers to two layers with 100 and 200 nodes.')
  parser.add_argument('--batchSize', type=int, default=100, help='Batch size.')
  parser.add_argument('--strengthThreshold', type=float, default=0, help='Threshold to split patient visits')
  ARGStemp = parser.parse_args()
  hiddenDimSize = [int(strDim) for strDim in ARGStemp.hiddenDimSize[1:-1].split(',')]
  ARGStemp.hiddenDimSize = hiddenDimSize
  return ARGStemp


if __name__ == '__main__':
  global ARGS
  ARGS = parse_arguments()

  testModel()