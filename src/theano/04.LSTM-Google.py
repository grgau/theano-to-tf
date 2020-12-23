#################################################################################################
# author: junio@usp.br - Jose F Rodrigues-Jr
# note: in many places, the code could be shorter, but that would just make it less comprehensible
#################################################################################################
import random
import math
import cPickle as pickle
import os
from collections import OrderedDict
import argparse
import theano
import theano.tensor as T
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from collections import Counter

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

def unzip(zipped):
	new_params = OrderedDict()
	for key, value in zipped.iteritems():
		new_params[key] = value.get_value()
	return new_params

def numpy_floatX(data):
	return np.asarray(data, dtype=config.floatX)

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

	x_hotvectors_tensorf = np.zeros((maxNumberOfAdmissions, numberOfPatients, ARGS.numberOfInputCodes)).astype(config.floatX)
	y_hotvectors_tensor = np.zeros((maxNumberOfAdmissions, numberOfPatients, ARGS.numberOfInputCodes)).astype(config.floatX)
	mask = np.zeros((maxNumberOfAdmissions, numberOfPatients)).astype(config.floatX)

	for idx, (train_patient_matrix,label_patient_matrix) in enumerate(zip(train_tensor,labels_tensor)):
		for i_th_visit, visit_line in enumerate(train_patient_matrix[:-1]): #ignores the last admission, which is not part of the training
			for code in visit_line:
				x_hotvectors_tensorf[i_th_visit, idx, code] = 1
		for i_th_visit, visit_line in enumerate(label_patient_matrix[1:]):  #label_matrix[1:] = all but the first admission slice, not used to evaluate (this is the answer)
			for code in visit_line:
				y_hotvectors_tensor[i_th_visit, idx, code] = 1
		mask[:nVisitsOfEachPatient_List[idx], idx] = 1.

	nVisitsOfEachPatient_List = np.array(nVisitsOfEachPatient_List, dtype=config.floatX)
	return x_hotvectors_tensorf, y_hotvectors_tensor, mask, nVisitsOfEachPatient_List


#initialize model tPARAMS
def init_params_LSTMGoogle(previousDimSize):
	for count, hiddenDimSize in enumerate(ARGS.hiddenDimSize):  #by default: 0, 200; 1, 200 according to enumerate
		xavier_variance = math.sqrt(2.0 / float(previousDimSize + hiddenDimSize))
		tPARAMS['W_f'  + str(count)] = theano.shared(np.random.normal(0., xavier_variance, (previousDimSize, hiddenDimSize)).astype(config.floatX),'W_f'  + str(count))
		tPARAMS['V_f' + str(count)] = theano.shared(np.identity(hiddenDimSize).astype(config.floatX), 'V_f' + str(count))
		tPARAMS['U_f' + str(count)] = theano.shared(np.identity(hiddenDimSize).astype(config.floatX),'U_f'  + str(count))
		tPARAMS['b_f'  + str(count)] = theano.shared(np.zeros(hiddenDimSize).astype(config.floatX),'b_f'  + str(count))

		tPARAMS['W_i'  + str(count)] = theano.shared(np.random.normal(0., xavier_variance, (previousDimSize, hiddenDimSize)).astype(config.floatX),'W_i'  + str(count))
		tPARAMS['V_i' + str(count)] = theano.shared(np.identity(hiddenDimSize).astype(config.floatX), 'V_i' + str(count))
		tPARAMS['U_i' + str(count)] = theano.shared(np.identity(hiddenDimSize).astype(config.floatX),'U_i'  + str(count))
		tPARAMS['b_i'  + str(count)] = theano.shared(np.zeros(hiddenDimSize).astype(config.floatX),'b_i'  + str(count))

		tPARAMS['V_h'  + str(count)] = theano.shared(np.random.normal(0., xavier_variance, (previousDimSize, hiddenDimSize)).astype(config.floatX),'V_h'  + str(count))

		tPARAMS['W_o'  + str(count)] = theano.shared(np.random.normal(0., xavier_variance, (previousDimSize, hiddenDimSize)).astype(config.floatX),'W_o'  + str(count))
		tPARAMS['U_o' + str(count)] = theano.shared(np.identity(hiddenDimSize).astype(config.floatX), 'U_o' + str(count))
		# tPARAMS['U_co' + str(count)] = theano.shared(np.identity(hiddenDimSize).astype(config.floatX),'U_co'  + str(count))
		tPARAMS['b_o'  + str(count)] = theano.shared(np.zeros(hiddenDimSize).astype(config.floatX),'b_o'  + str(count))

		tPARAMS['W_c'  + str(count)] = theano.shared(np.random.normal(0., xavier_variance, (previousDimSize, hiddenDimSize)).astype(config.floatX),'W_c'  + str(count))
		tPARAMS['U_c' + str(count)] = theano.shared(np.identity(hiddenDimSize).astype(config.floatX), 'U_c' + str(count))
		tPARAMS['b_c'  + str(count)] = theano.shared(np.zeros(hiddenDimSize).astype(config.floatX),'b_c'  + str(count))
		previousDimSize = hiddenDimSize

		tPARAMS['W_p'  + str(count)] = theano.shared(np.random.normal(0., xavier_variance, (previousDimSize, hiddenDimSize)).astype(config.floatX),'W_p'  + str(count))
	return previousDimSize

def LSTMGoogle_layer(inputTensor, layerIndex, hiddenDimSize, mask=None):
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
		# h_new = output_gate*T.tanh(state_new)
		h_new = stepMask[:, None] * h_new + (1. - stepMask)[:, None] * h_previous  #h_new = mask * h_new + (1 - mask) * h_previous
		return h_new, state_new

	results, _ = theano.scan(fn=stepFn,
			   sequences=[mask, W_ix, W_fx, W_ox, W_cx, V_hx],
			   outputs_info=[T.alloc(numpy_floatX(0.0), numberOfPatientsInTheBatch, hiddenDimSize),T.alloc(numpy_floatX(0.0), numberOfPatientsInTheBatch, hiddenDimSize)],
			   name='LSTMGoogle_layer' + layerIndex,
			   n_steps=maxNumberOfVisits)

	return results[1]

def init_params_output_layer(previousDimSize):
	xavier_variance = math.sqrt(2.0 / float(previousDimSize + ARGS.numberOfInputCodes))
	tPARAMS['W_output'] = theano.shared(np.random.normal(0., xavier_variance, (previousDimSize, ARGS.numberOfInputCodes)).astype(config.floatX), 'W_output')
	# tPARAMS['W_rnn'] = theano.shared(np.random.normal(0., xavier_variance, (previousDimSize, ARGS.numberOfInputCodes)).astype(config.floatX), 'W_rnn')
	tPARAMS['b_output'] = theano.shared(np.zeros(ARGS.numberOfInputCodes).astype(config.floatX), name='b_output')
	tPARAMS['olrelu'] = theano.shared(0.1, name='olrelu')

def dropout(nDimensionalData):
	randomS = RandomStreams(13713)
	newTensor = nDimensionalData * randomS.binomial(nDimensionalData.shape, p=ARGS.dropoutRate, dtype=nDimensionalData.dtype)
	#https://www.quora.com/How-do-you-implement-a-dropout-in-deep-neural-networks
	return newTensor

def build_model():
	xf = T.tensor3('xf', dtype=config.floatX)
	y = T.tensor3('y', dtype=config.floatX)
	mask = T.matrix('mask', dtype=config.floatX)

	nVisitsOfEachPatient_List = T.vector('nVisitsOfEachPatient_List', dtype=config.floatX)
	maxNumberOfAdmissions = xf.shape[0]

	flowing_tensorf = xf

	for i, hiddenDimSize in enumerate(ARGS.hiddenDimSize):
		flowing_tensorf = LSTMGoogle_layer(flowing_tensorf, str(i), hiddenDimSize, mask=mask)
		flowing_tensorf = dropout(flowing_tensorf)

	results, _ = theano.scan(
		# lambda theFlowingTensor: T.nnet.softmax(T.nnet.relu(T.dot(theFlowingTensor, tPARAMS['W_output']) + tPARAMS['b_output'], tPARAMS['olrelu'])),
		lambda theFlowingTensor: T.nnet.softmax(T.nnet.relu(T.dot(flowing_tensorf[-1], tPARAMS['W_output']) + tPARAMS['b_output'], tPARAMS['olrelu'])),
		sequences=[flowing_tensorf],
		outputs_info=None,
		name='softmax_layer',
		n_steps=maxNumberOfAdmissions)

	flowing_tensor = results * mask[:, :, None]

	epislon = 1e-8
	cross_entropy = -(y * T.log(flowing_tensor + epislon) + (1. - y) * T.log(1. - flowing_tensor + epislon))
	# the complete crossentropy equation is -1/n* sum(cross_entropy); where n is the number of elements
	# see http://neuralnetworksanddeeplearning.com/chap3.html#regularization
	prediction_loss = cross_entropy.sum(axis=2).sum(axis=0) / nVisitsOfEachPatient_List

	L2_regularized_loss = T.mean(prediction_loss) + ARGS.LregularizationAlpha*(tPARAMS['W_output'] ** 2).sum()
	MODEL = L2_regularized_loss
	return xf, y, mask, nVisitsOfEachPatient_List, MODEL


#this code comes originally from deeplearning.net/tutorial/LSTMGoogle.html
#http://ruder.io/optimizing-gradient-descent/index.html#adadelta
#https://arxiv.org/abs/1212.5701
def addAdadeltaGradientDescent(grads, xf, y, mask, nVisitsOfEachPatient_List, MODEL):
	zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_grad' % k) for k, p in tPARAMS.iteritems()]
	running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rup2' % k) for k, p in tPARAMS.iteritems()]
	running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rgrad2' % k) for k, p in tPARAMS.iteritems()]

	zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
	rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

	TRAIN_MODEL_COMPILED = theano.function([xf, y, mask, nVisitsOfEachPatient_List], MODEL, updates=zgup + rg2up, name='adadelta_TRAIN_MODEL_COMPILED')

	updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
	ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
	param_up = [(p, p + ud) for p, ud in zip(tPARAMS.values(), updir)]

	UPDATE_WEIGHTS_COMPILED = theano.function([], [], updates=ru2up + param_up, name='adadelta_UPDATE_WEIGHTS_COMPILED')
	return TRAIN_MODEL_COMPILED, UPDATE_WEIGHTS_COMPILED


def load_data():
	threshold = ARGS.strengthThreshold
	print("-> Threshold value: " + str(threshold))
	main_trainSet = pickle.load(open(ARGS.inputFileRadical+'.train', 'rb'))
	print("-> " + str(len(main_trainSet)) + " patients at dimension 0 for file: "+ ARGS.inputFileRadical + ".train dimensions ")
	main_testSet = pickle.load(open(ARGS.inputFileRadical+'.test', 'rb'))
	print("-> " + str(len(main_testSet)) + " patients at dimension 0 for file: "+ ARGS.inputFileRadical + ".test dimensions ")
	print("Note: these files carry 3D tensor data; the above numbers refer to dimension 0, dimensions 1 and 2 have irregular sizes.")
	main_trainIntervalSet = pickle.load(open(ARGS.inputFileRadical+'.INTERVAL.train', 'rb'))
	main_testIntervalSet = pickle.load(open(ARGS.inputFileRadical+'.INTERVAL.test', 'rb'))

	ARGS.numberOfInputCodes = getNumberOfCodes([main_trainSet,main_testSet])
	print 'Number of diagnosis input codes: ' + str(ARGS.numberOfInputCodes)

	#uses the same data for testing, but disregarding the fist admission of each patient
	labels_trainSet = pickle.load(open(ARGS.inputFileRadical+'.train', 'rb'))
	labels_testSet = pickle.load(open(ARGS.inputFileRadical+'.test', 'rb'))

	main_trainSet = split_by_strenght(main_trainSet, main_trainIntervalSet, threshold)
	labels_trainSet = split_by_strenght(labels_trainSet, main_trainIntervalSet, threshold)
	# main_testSet = split_by_strenght(main_testSet, main_testIntervalSet, threshold)
	# labels_testSet = split_by_strenght(labels_testSet, main_testIntervalSet, threshold)

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
def performEvaluation(TEST_MODEL_COMPILED, test_Set):
	batchSize = ARGS.batchSize

	n_batches = int(np.ceil(float(len(test_Set[0])) / float(batchSize))) #default batch size is 100
	crossEntropySum = 0.0
	dataCount = 0.0
	#computes de crossEntropy for all the elements in the test_Set, using the batch scheme of partitioning
	for index in xrange(n_batches):
		batchX = test_Set[0][index * batchSize:(index + 1) * batchSize]
		batchY = test_Set[1][index * batchSize:(index + 1) * batchSize]
		xf, y, mask, nVisitsOfEachPatient_List = prepareHotVectors(batchX, batchY)
		crossEntropy = TEST_MODEL_COMPILED(xf, y, mask, nVisitsOfEachPatient_List)

		#accumulation by simple summation taking the batch size into account
		crossEntropySum += crossEntropy * len(batchX)
		dataCount += float(len(batchX))
		#At the end, it returns the mean cross entropy considering all the batches
	return n_batches, crossEntropySum / dataCount

def train_model():
	print '==> data loading'
	trainSet, testSet = load_data()
	previousDimSize = ARGS.numberOfInputCodes

	print '==> parameters initialization'
	print('Using neuron type Long Short-Term Memory by Zen et. al (Google)')
	previousDimSize = init_params_LSTMGoogle(previousDimSize)
	init_params_output_layer(previousDimSize)

	print '==> model building'
	xf, y, mask, nVisitsOfEachPatient_List, MODEL =  build_model()
	grads = T.grad(MODEL, wrt=tPARAMS.values())

	TRAIN_MODEL_COMPILED, UPDATE_WEIGHTS_COMPILED = addAdadeltaGradientDescent(grads, xf, y, mask, nVisitsOfEachPatient_List, MODEL)

	print '==> training and validation'
	batchSize = ARGS.batchSize
	n_batches = int(np.ceil(float(len(trainSet[0])) / float(batchSize)))
	TEST_MODEL_COMPILED = theano.function(inputs=[xf, y, mask, nVisitsOfEachPatient_List], outputs=MODEL, name='TEST_MODEL_COMPILED')

	bestValidationCrossEntropy = 1e20
	bestValidationEpoch = 0
	bestModelFileName = ''

	iImprovementEpochs = 0
	iConsecutiveNonImprovements = 0
	epoch_counter = 0
	for epoch_counter in xrange(ARGS.nEpochs):
		iteration = 0
		trainCrossEntropyVector = []
		for index in random.sample(range(n_batches), n_batches):
			batchX = trainSet[0][index*batchSize:(index+1)*batchSize]
			batchY = trainSet[1][index*batchSize:(index + 1)*batchSize]
			xf, y, mask, nVisitsOfEachPatient_List = prepareHotVectors(batchX, batchY)
			xf += np.random.normal(0, 0.1, xf.shape)  #add gaussian noise as a means to reduce overfitting

			trainCrossEntropy = TRAIN_MODEL_COMPILED(xf, y, mask, nVisitsOfEachPatient_List)
			trainCrossEntropyVector.append(trainCrossEntropy)
			UPDATE_WEIGHTS_COMPILED()
			iteration += 1
		#----------test -> uses TEST_MODEL_COMPILED
		#evaluates the network with the testSet
		print('-> Epoch: %d, mean cross entropy considering %d TRAINING batches: %f' % (epoch_counter, n_batches, np.mean(trainCrossEntropyVector)))
		nValidBatches, validationCrossEntropy = performEvaluation(TEST_MODEL_COMPILED, testSet)
		print('			 mean cross entropy considering %d VALIDATION batches: %f' % (nValidBatches, validationCrossEntropy))
		if validationCrossEntropy < bestValidationCrossEntropy:
			iImprovementEpochs += 1
			iConsecutiveNonImprovements = 0
			bestValidationCrossEntropy = validationCrossEntropy
			bestValidationEpoch = epoch_counter

			tempParams = unzip(tPARAMS)
			if os.path.exists(bestModelFileName):
				os.remove(bestModelFileName)
			np.savez_compressed(ARGS.outFile + '.' + str(epoch_counter), **tempParams)
			bestModelFileName = ARGS.outFile + '.' + str(epoch_counter) + '.npz'
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
	parser.add_argument('--strengthThreshold', type=float, default=0, help='Threshold to split patient visits')

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