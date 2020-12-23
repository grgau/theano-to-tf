#################################################################################################
# author: junio@usp.br - Jose F Rodrigues-Jr
# note: in many places, the code could be shorter, but that would just make it less comprehensible
#################################################################################################
import cPickle as pickle
from collections import OrderedDict
from itertools import count
import argparse
import theano
import theano.tensor as T
from theano import config
import numpy as np
import csv

global ARGS
global tPARAMS

def strenght(inter, adm_i, adm_j):
  intersect = len(list(set(adm_i).intersection(adm_j)))
  intersect_norm = np.float64(intersect) / np.float64(min(len(adm_i), len(adm_j)))
  # return intersect_norm if inter == 0 else intersect / ((np.log(inter) + 1) ** intersect_norm)
  return intersect_norm if inter == 0 else intersect / ((np.log(inter) + 1) ** (1/5*intersect_norm))

def generate_strenghts(dataset, interval):
	normalized_strs = []

	for t, inter in zip(dataset, interval):
		for i in range(1, len(t)):
			normalized_strs.append(strenght(inter[i], t[i-1], t[i]))

	if len(normalized_strs) > 0:
		min_str, max_str = min(normalized_strs), max(normalized_strs)
		for i, val in enumerate(normalized_strs):
			normalized_strs[i] = 0 if (max_str-min_str) == 0 else (val-min_str) / (max_str-min_str)
	return normalized_strs

def split_by_strenght(dataset, interval, threshold):
	str_counter = 0
	strenghts = generate_strenghts(dataset, interval)
	new_dataset = []
	for t, inter in zip(dataset, interval):
		split_count = 0
		arrays = [[t[0]]]
		for i in range(1, len(t)):
			if len(arrays[split_count]) > 1 and len(t)-1 > i and strenghts[str_counter] < threshold: #len(t)-1 is for checking if last element in array, to not split
				arrays.append([t[i]])
				split_count += 1
			else:
				arrays[len(arrays) - 1].append(t[i])
			str_counter += 1
		new_dataset.extend(arrays)
	return new_dataset

def numpy_floatX(data):
	return np.asarray(data, dtype=config.floatX)

def prepareHotVectors(test_tensor):
	n_visits_of_each_patientList = np.array([len(seq) for seq in test_tensor]) - 1
	number_of_patients = len(test_tensor)
	max_number_of_visits = np.max(n_visits_of_each_patientList)

	x_hotvectors_tensorf = np.zeros((max_number_of_visits, number_of_patients, ARGS.numberOfInputCodes)).astype(config.floatX)
	mask = np.zeros((max_number_of_visits, number_of_patients)).astype(config.floatX)

	for idx, (train_patient_matrix) in enumerate(test_tensor):
		for i_th_visit, visit_line in enumerate(train_patient_matrix[:-1]): #ignores the last visit, which is not part of the computation
			for code in visit_line:
				x_hotvectors_tensorf[i_th_visit, idx, code] = 1
		mask[:n_visits_of_each_patientList[idx], idx] = 1.

	return x_hotvectors_tensorf, mask, n_visits_of_each_patientList

def loadModel():
	model = np.load(ARGS.modelFile)
	tPARAMS = OrderedDict()
	for key, value in model.iteritems():
		tPARAMS[key] = theano.shared(value, name=key)
	ARGS.numberOfInputCodes = model['W_f0'].shape[0]
	return tPARAMS


def LSTM_layer(inputTensor, layerIndex, hiddenDimSize, mask=None):
	maxNumberOfVisits = inputTensor.shape[0]
	numberOfPatientsInTheBatch = inputTensor.shape[1]

	W_ix = T.dot(inputTensor,tPARAMS['W_i' + layerIndex])
	W_fx = T.dot(inputTensor,tPARAMS['W_f' + layerIndex])
	W_ox = T.dot(inputTensor,tPARAMS['W_o' + layerIndex])
	W_cx = T.dot(inputTensor,tPARAMS['W_c' + layerIndex])
	V_hx = T.dot(inputTensor,tPARAMS['V_h' + layerIndex])

	def stepFn(stepMask, w_ix, w_fx, w_ox, w_cx, v_hx, h_previous, state_previous):  # .* -> element-wise multiplication; * -> matrix multiplication
		input_gate = T.nnet.sigmoid(w_ix + T.dot(h_previous,tPARAMS['U_i' + layerIndex]) + T.dot(state_previous,tPARAMS['V_i' + layerIndex]) + tPARAMS['b_i' + layerIndex])
		forget_gate = T.nnet.sigmoid(w_fx + T.dot(h_previous,tPARAMS['U_f' + layerIndex]) + T.dot(state_previous,tPARAMS['V_f' + layerIndex]) + tPARAMS['b_f' + layerIndex])
		output_gate = T.nnet.sigmoid(w_ox + T.dot(h_previous,tPARAMS['U_o' + layerIndex]) + tPARAMS['b_o' + layerIndex])
		state_new = forget_gate*state_previous + input_gate * T.tanh(w_cx + T.dot(h_previous,tPARAMS['U_c' + layerIndex]) + tPARAMS['b_c' + layerIndex])
		state_new = stepMask[:, None]*state_new + (1. - stepMask)[:, None]*state_previous
		r = T.tanh(state_new)
		m = T.dot(r,tPARAMS['W_p' + layerIndex])
		h_new = output_gate*(m + v_hx)
		h_new = stepMask[:, None] * h_new + (1. - stepMask)[:, None] * h_previous  #h_new = mask * h_new + (1 - mask) * h_previous
		return h_new, state_new

	results, _ = theano.scan(fn=stepFn,
			   sequences=[mask, W_ix, W_fx, W_ox, W_cx, V_hx],
			   outputs_info=[T.alloc(numpy_floatX(0.0), numberOfPatientsInTheBatch, hiddenDimSize),T.alloc(numpy_floatX(0.0), numberOfPatientsInTheBatch, hiddenDimSize)],
			   name='LSTM_layer' + layerIndex,
			   n_steps=maxNumberOfVisits)

	return results[1]

def build_model():
	xf = T.tensor3('xf', dtype=config.floatX)
	mask = T.matrix('mask', dtype=config.floatX)
	maxNumberOfAdmissions = xf.shape[0]

	flowing_tensorf = xf

	for i, hiddenDimSize in enumerate(ARGS.hiddenDimSize):
		flowing_tensorf = LSTM_layer(flowing_tensorf, str(i), hiddenDimSize, mask=mask)

	results, _ = theano.scan(
		lambda theFlowingTensor: T.nnet.softmax(T.nnet.relu(T.dot(flowing_tensorf[-1], tPARAMS['W_output']) + tPARAMS['b_output'], tPARAMS['olrelu'])),
		# lambda theFlowingTensor: T.nnet.softmax(T.nnet.relu(T.dot(theFlowingTensor, tPARAMS['W_output']) + T.dot(flowing_tensorf[-1], tPARAMS['W_rnn']) + tPARAMS['b_output'], tPARAMS['olrelu'])),
		sequences=[flowing_tensorf],
		outputs_info=None,
		name='softmax_layer',
		n_steps=maxNumberOfAdmissions)

	results = results * mask[:, :, None]

	return xf, mask, results



def load_data():
	testSet_x = np.array(pickle.load(open(ARGS.inputFileRadical+'.test', 'rb')))
	testSet_y = np.array(pickle.load(open(ARGS.inputFileRadical+'.test', 'rb')))
	testIntervalSet = np.array(pickle.load(open(ARGS.inputFileRadical+'.INTERVAL.test', 'rb')))

	def len_argsort(seq):
		return sorted(range(len(seq)), key=lambda x: len(seq[x]))

	# testSet_x = split_by_strenght(testSet_x, testIntervalSet, threshold)
	# testSet_y = split_by_strenght(testSet_y, testIntervalSet, threshold)

	sorted_index = len_argsort(testSet_x)
	testSet_x = [testSet_x[i] for i in sorted_index]
	testSet_y = [testSet_y[i] for i in sorted_index]
	testIntervalSet = [testIntervalSet[i] for i in sorted_index]

	testSet = [testSet_x, testSet_y]
	return testSet, testIntervalSet

def load_patients():
	return np.array(pickle.load(open(ARGS.inputFileRadical+'.map.test', 'rb')))


def testModel():
	threshold = ARGS.strengthThreshold
	global tPARAMS
	tPARAMS = loadModel()
	testSet, testIntervalSet = load_data()
	xf, mask, MODEL = build_model()
	PREDICTOR_COMPILED = theano.function(inputs=[xf, mask], outputs=MODEL, name='PREDICTOR_COMPILED')

	nBatches = int(np.ceil(float(len(testSet[0])) / float(ARGS.batchSize)))
	predictedY_list = []
	actualY_list = []
	predicted_yList = []

	splittedX = []
	splittedY = []

	for batchIndex in range(nBatches):
		batchX = testSet[0][batchIndex * ARGS.batchSize: (batchIndex + 1) * ARGS.batchSize]
		batchY = testSet[1][batchIndex * ARGS.batchSize: (batchIndex + 1) * ARGS.batchSize]
		batchInterval = testIntervalSet[batchIndex * ARGS.batchSize: (batchIndex + 1) * ARGS.batchSize]
		xf, mask, nVisitsOfEachPatient_List = prepareHotVectors(batchX)
		predicted_y = PREDICTOR_COMPILED(xf, mask)
		predicted_yList.append(predicted_y.tolist()[-1])

		splittedTestSet_0 = split_by_strenght(batchX, batchInterval, threshold)
		splittedTestSet_1 = split_by_strenght(batchY, batchInterval, threshold)

		splittedX.extend(splittedTestSet_0)
		splittedY.extend(splittedTestSet_1)

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


	nBatches = int(np.ceil(float(len(splittedX)) / float(ARGS.batchSize)))

	for batchIndex in range(nBatches):
		batchX = splittedX[batchIndex * ARGS.batchSize: (batchIndex + 1) * ARGS.batchSize]
		batchY = splittedY[batchIndex * ARGS.batchSize: (batchIndex + 1) * ARGS.batchSize]
		xf, mask, nVisitsOfEachPatient_List = prepareHotVectors(batchX)
		predicted_y = PREDICTOR_COMPILED(xf, mask)
		predicted_yList.append(predicted_y.tolist()[-1])

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

	print str(finalRecalls[0])
	print	str(finalRecalls[1])
	print str(finalRecalls[2])
	# return predicted_y.reshape(-1, predicted_y.shape[-1]).tolist() # esse nao
	# return patientsSet, predicted_yList # esse sim
	return predicted_yList

	# Pegue n-1 e sai 1
	# Considerar somente ultima predicao pra clusterizacao -1 acho
	# Matriz cada coluna eh probabilidade de diagnostico, e cada linha paciente (considerando ultima predicao)


def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('inputFileRadical', type=str, metavar='<visit_file>', help='File radical name (the software will look for .test file) with pickled data organized as patient x admission x codes.')
	parser.add_argument('modelFile', type=str, metavar='<model_file>', help='The path to the model file .npz')
	parser.add_argument('--hiddenDimSize', type=str, default='[271]', help='Number of layers and their size - for example [100,200] refers to two layers with 100 and 200 nodes.')
	parser.add_argument('--strengthThreshold', type=float, default=0.22, help='Threshold to split patient visits')
	ARGStemp = parser.parse_args()
	hiddenDimSize = [int(strDim) for strDim in ARGStemp.hiddenDimSize[1:-1].split(',')]
	ARGStemp.hiddenDimSize = hiddenDimSize
	return ARGStemp


if __name__ == '__main__':
	global tPARAMS
	tPARAMS = OrderedDict()
	global ARGS
	ARGS = parse_arguments()
	# patients, predictions = testModel()
	predictions = testModel()

	# with open("output.csv", "wb") as f:
		# writer = csv.writer(f)
		# for idx, batch in zip(count(step=ARGS.batchSize), predictions):
		# 	writer.writerows(np.column_stack((patients[idx:idx+len(batch)], np.array(batch))).tolist())
