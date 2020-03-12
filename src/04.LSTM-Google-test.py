#################################################################################################
# author: junio@usp.br - Jose F Rodrigues-Jr
# note: in many places, the code could be shorter, but that would just make it less comprehensible
#################################################################################################
import cPickle as pickle
from collections import OrderedDict
import argparse
import theano
import theano.tensor as T
from theano import config
import numpy as np

global ARGS
global tPARAMS

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
	# https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43905.pdf

	W_ix = T.dot(inputTensor,tPARAMS['W_i' + layerIndex])
	W_fx = T.dot(inputTensor,tPARAMS['W_f' + layerIndex])
	W_ox = T.dot(inputTensor,tPARAMS['W_o' + layerIndex])
	W_cx = T.dot(inputTensor,tPARAMS['W_c' + layerIndex])
	V_hx = T.dot(inputTensor,tPARAMS['V_h' + layerIndex])

	def stepFn(stepMask, w_ix, w_fx, w_ox, w_cx, v_hx, h_previous, state_previous):  # .* -> element-wise multiplication; * -> matrix multiplication
		input_gate = T.nnet.sigmoid(w_ix + T.dot(h_previous,tPARAMS['U_i' + layerIndex]) + T.dot(state_previous,tPARAMS['V_i' + layerIndex] + tPARAMS['b_i' + layerIndex]))
		forget_gate = T.nnet.sigmoid(w_fx + T.dot(h_previous,tPARAMS['U_f' + layerIndex]) + T.dot(state_previous,tPARAMS['V_f' + layerIndex] + tPARAMS['b_f' + layerIndex]))
		state_new = forget_gate*state_previous + input_gate * T.tanh(w_cx + T.dot(h_previous,tPARAMS['U_c' + layerIndex]) + tPARAMS['b_c' + layerIndex])
		state_new = stepMask[:, None]*state_new + (1. - stepMask)[:, None]*state_previous
		output_gate = T.nnet.sigmoid(w_ox + T.dot(h_previous,tPARAMS['U_o' + layerIndex]) + T.dot(state_new,tPARAMS['U_co' + layerIndex]) + tPARAMS['b_o' + layerIndex])
		r = T.tanh(state_new)
		m = T.dot(r,tPARAMS['W_p' + layerIndex])
		h_new = output_gate*(m + v_hx)
		h_new = stepMask[:, None] * h_new + (1. - stepMask)[:, None] * h_previous  # h_new=mask*h_new + (1-mask)*h_previous
		return h_new, state_new

	results, _ = theano.scan(fn=stepFn,
			   sequences=[mask, W_ix, W_fx, W_ox, W_cx, V_hx],
			   outputs_info=[T.alloc(numpy_floatX(0.0), numberOfPatientsInTheBatch, hiddenDimSize),T.alloc(numpy_floatX(0.0), numberOfPatientsInTheBatch, hiddenDimSize)],
			   name='LSTM_layer' + layerIndex,
			   n_steps=maxNumberOfVisits)

	return results[0]

def build_model():
	xf = T.tensor3('xf', dtype=config.floatX)
	mask = T.matrix('mask', dtype=config.floatX)
	maxNumberOfAdmissions = xf.shape[0]

	flowing_tensorf = xf

	for i, hiddenDimSize in enumerate(ARGS.hiddenDimSize):
		flowing_tensorf = LSTM_layer(flowing_tensorf, str(i), hiddenDimSize, mask=mask)

	results, _ = theano.scan(
		lambda theFlowingTensor: T.nnet.softmax(T.nnet.relu(T.dot(theFlowingTensor, tPARAMS['W_output']) + tPARAMS['b_output'], tPARAMS['olrelu'])),
		sequences=[flowing_tensorf],
		outputs_info=None,
		name='softmax_layer',
		n_steps=maxNumberOfAdmissions)

	results = results * mask[:, :, None]

	return xf, mask, results



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
	print '==> model loading'
	global tPARAMS
	tPARAMS = loadModel()

	print '==> data loading'
	testSet = load_data()

	print '==> model rebuilding'
	xf, mask, MODEL = build_model()
	PREDICTOR_COMPILED = theano.function(inputs=[xf, mask], outputs=MODEL, name='PREDICTOR_COMPILED')

	print '==> model execution'
	nBatches = int(np.ceil(float(len(testSet[0])) / float(ARGS.batchSize)))
	predictedY_list = []
	actualY_list = []
	for batchIndex in range(nBatches):
		batchX = testSet[0][batchIndex * ARGS.batchSize: (batchIndex + 1) * ARGS.batchSize]
		batchY = testSet[1][batchIndex * ARGS.batchSize: (batchIndex + 1) * ARGS.batchSize]
		xf, mask, nVisitsOfEachPatient_List = prepareHotVectors(batchX)
		predicted_y = PREDICTOR_COMPILED(xf, mask)

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

	print '==> computation of prediction results'
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

	print 'Results for Precision@10, Precision@20, and Precision@30'
	print str(finalRecalls[0])
	print	str(finalRecalls[1])
	print str(finalRecalls[2])


def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('inputFileRadical', type=str, metavar='<visit_file>', help='File radical name (the software will look for .test file) with pickled data organized as patient x admission x codes.')
	parser.add_argument('modelFile', type=str, metavar='<model_file>', help='The path to the model file .npz')
	parser.add_argument('--hiddenDimSize', type=str, default='[271]', help='Number of layers and their size - for example [100,200] refers to two layers with 100 and 200 nodes.')
	parser.add_argument('--batchSize', type=int, default=100, help='Batch size.')
	ARGStemp = parser.parse_args()
	hiddenDimSize = [int(strDim) for strDim in ARGStemp.hiddenDimSize[1:-1].split(',')]
	ARGStemp.hiddenDimSize = hiddenDimSize
	return ARGStemp


if __name__ == '__main__':
	global tPARAMS
	tPARAMS = OrderedDict()
	global ARGS
	ARGS = parse_arguments()

	testModel()
