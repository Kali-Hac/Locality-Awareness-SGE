import tensorflow as tf
from tensorflow.python.layers.core import Dense
import numpy as np
import time
import matplotlib as mpl
import copy
import os
from tensorflow.python.ops import rnn_cell_impl
# mpl.use('Agg')
# import matplotlib.pyplot as plt 
import os

# Number of Epochs
epochs = 100
# Batch Size
batch_size = 128
# RNN Size k = 256
rnn_size = 256
# Number of Layers, 2-layer LSTM
num_layers = 2
# Time Steps of Input, f = 6 skeleton frames
time_steps = 6
# Length of Series, J = 20 body joints in a sequence
series_length = 20
# Learning Rate
learning_rate = 0.0005
lr_decay = 0.95
momentum = 0.5
lambda_l2_reg = 0.02
dataset = False
attention = False
manner = False
gpu = False
permutation_flag = False
permutation_test_flag = False
permutation_test_2_flag = False
permutation = 0
test_permutation = 0
test_2_permutation = 0
Reverse = True
use_attention = True
Bi_LSTM = False
AGEs =  True
Frozen = False
tf.app.flags.DEFINE_string('attention', 'LA', "(LA) Locality-oriented Attention Alignment or BA (Basic Attention Alignment)")
tf.app.flags.DEFINE_string('manner', 'ap', "average prediction (ap) or sequence-level concatenation (sc)")
tf.app.flags.DEFINE_string('dataset', 'BIWI', "Dataset: BIWI or IAS or KGBD")
tf.app.flags.DEFINE_string('length', '6', "4, 6, 8 or 10")
tf.app.flags.DEFINE_string('gpu', '0', "GPU number")
tf.app.flags.DEFINE_string('frozen', '0', "Freeze CAGEs for contrastive learning")
tf.app.flags.DEFINE_string('c_reid', '0', "Peform re-id use projection vectors")
tf.app.flags.DEFINE_string('t', '0.05', "Temperature for contrastive learning")
tf.app.flags.DEFINE_string('train_flag', '1', "Choose to train (1) or test (0)")
tf.app.flags.DEFINE_string('view', 'None', "Choose different views for KS20")
tf.app.flags.DEFINE_string('transfer', 'None', "Choose a dataset's encoding model to transfer encoding")
tf.app.flags.DEFINE_string('model', 'rev_rec', "prediction, sorting, rev_rec (Rev. Rec.), rev_rec_plus(Rev. Rec. Plus)")
FLAGS = tf.app.flags.FLAGS
config = tf.ConfigProto()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
temperature = 0.1
config.gpu_options.allow_growth = True
view = 'view_'
transfer = 'None'
Model = 'rev_rec'
pre_task = 'rev_rec'
def main(_):
	global attention, dataset, series_length, epochs, time_steps, gpu, manner, frames_ps, \
		temperature, Frozen, C_reid, temperature, train_flag, view, use_attention, transfer, Model, pre_task
	attention, dataset, gpu, manner, length, Frozen, C_reid, temperature, train_flag, \
	view_num, transfer, Model = FLAGS.attention, \
	                                                  FLAGS.dataset, FLAGS.gpu, FLAGS.manner, \
	                                                  FLAGS.length, FLAGS.frozen, FLAGS.c_reid, \
	                                                FLAGS.t, FLAGS.train_flag, FLAGS.view, \
	                                        FLAGS.transfer, FLAGS.model,
	if attention not in ['BA', 'LA']:
		raise Exception('Attention must be BA or LA.')
	if manner not in ['sc', 'ap']:
		raise Exception('Training manner must be sc or ap.')
	if dataset not in ['BIWI', 'IAS', 'KGBD', 'KinectReID', 'KS20']:
		raise Exception('Dataset must be BIWI, IAS, KGBD, KinectReID or KS20.')
	if not gpu.isdigit() or int(gpu) < 0:
		raise Exception('GPU number must be a positive integer.')
	if length not in ['4', '6', '8', '10']:
		raise Exception('Length number must be 4, 6, 8 or 10.')
	if Frozen not in ['0', '1']:
		raise Exception('Frozen state must be 0 or 1.')
	if C_reid not in ['0', '1']:
		raise Exception('C_reid state must be 0 or 1.')
	if train_flag not in ['0', '1', '2']:
		raise Exception('Train_flag must be 0, 1, or 2 (Only Evaluation).')
	if view_num not in ['0', '1', '2', '3', '4', 'None']:
		raise Exception('View_num must be 0, 1, 2, 3, 4 or None')
	if transfer not in ['BIWI', 'IAS', 'KGBD', 'KS20', 'None']:
		raise Exception('Transfer dataset must be BIWI, IAS, KGBD, KS20 or None')
	if Model not in ['rev_rec', 'rev_rec_plus', 'prediction', 'sorting']:
		raise Exception('Model must be prediction, sorting, rev_rec or rev_rec_plus')
	os.environ['CUDA_VISIBLE_DEVICES'] = gpu
	folder_name = dataset + '_' + attention
	series_length = 20
	if dataset == 'KS20':
		series_length = 25
		view += view_num
		if view_num == 'None':
			view = ''
	if transfer != 'None':
		train_flag = '0'
	# time_steps = 6
	time_steps = int(length)
	temperature = float(temperature)
	pre_task = Model
	frames_ps = dataset + '/' + str(time_steps) + '/'
	epochs = 400

	if dataset != 'KS20':
		view = ''
	# if dataset == 'KGBD':
	# 	epochs = 100
	# Obtain CAGEs
	# Train self-supervised gait encoding model on X, Y, Z
	if dataset == 'KGBD':
		temperature = 0.5
	else:
		temperature = 0.1
	if Model == 'rev_rec_plus':
		# rev_rec uses 'LA' and other two pretext tasks use 'BA'
		attention = 'BA'

	print(
		' ## Dataset: %s\n ## Attention: %s\n ## Re-ID Manner: %s\n ## Sequence Length: %s\n ## Tempearture: %s\n ## Pretext Task: %s\n ## GPU: %s\n' %
		(dataset, attention, manner, str(time_steps), str(temperature), Model, str(gpu)))
	if train_flag == '1':
		print(' ## Training Gait Encoding Model: True')
	else:
		print(' ## Training Gait Encoding Model: False')
	print(' ## Training Recognition Network: True\n')
	if train_flag == '1' and Model != 'rev_rec_plus':
		try:
			os.mkdir('./Models/Gait_Encoding_models')
		except:
			pass
		folder_name = './Models/Gait_Encoding_models/' + folder_name
		for i in ['x', 'y', 'z']:
			try:
				os.mkdir(folder_name + '_' + i + '_' + str(time_steps) + '_' + str(temperature) + '_' + Frozen + view + 'pre_' + pre_task)
			except:
				pass
			train(folder_name + '_' + i + '_' + str(time_steps) + '_' + str(temperature)+ '_' + Frozen + view + 'pre_' + pre_task, i, train_dataset=dataset)
	elif train_flag == '1' and Model == 'rev_rec_plus':
		print(' ## Training Three Types of Gait Encoding Model: (1) Rev. Rec. (2) Prediction (3) Sorting, and Combine CAGEs to Train RN')
		attention = 'LA'
		folder_name = dataset + '_' + attention
		try:
			os.mkdir('./Models/Gait_Encoding_models')
		except:
			pass
		folder_name = './Models/Gait_Encoding_models/' + folder_name
		# Rev. Rec.
		pre_task = 'rev_rec'
		for i in ['x', 'y', 'z']:
			try:
				os.mkdir(folder_name + '_' + i + '_' + str(time_steps) + '_' + str(temperature) + '_' + Frozen + view + 'pre_' + pre_task)
			except:
				pass
			train(folder_name + '_' + i + '_' + str(time_steps) + '_' + str(temperature)+ '_' + Frozen + view + 'pre_' + pre_task, i, train_dataset=dataset)
		# Prediction
		attention = 'BA'
		folder_name = dataset + '_' + attention
		folder_name = './Models/Gait_Encoding_models/' + folder_name
		pre_task = 'prediction'
		for i in ['x', 'y', 'z']:
			try:
				os.mkdir(folder_name + '_' + i + '_' + str(time_steps) + '_' + str(temperature) + '_' + Frozen + view + 'pre_' + pre_task)
			except:
				pass
			train(folder_name + '_' + i + '_' + str(time_steps) + '_' + str(temperature)+ '_' + Frozen + view + 'pre_' + pre_task, i, train_dataset=dataset)
		# Sorting
		attention = 'BA'
		folder_name = dataset + '_' + attention
		folder_name = './Models/Gait_Encoding_models/' + folder_name
		pre_task = 'sorting'
		for i in ['x', 'y', 'z']:
			try:
				os.mkdir(folder_name + '_' + i + '_' + str(time_steps) + '_' + str(temperature) + '_' + Frozen + view + 'pre_' + pre_task)
			except:
				pass
			train(folder_name + '_' + i + '_' + str(time_steps) + '_' + str(temperature)+ '_' + Frozen + view + 'pre_' + pre_task, i, train_dataset=dataset)
		pre_task = 'rev_rec_plus'
	print('Generate CAGEs')
	if dataset == 'IAS':
		X, X_y, t_X, t_X_y, t_2_X, t_2_X_y, t_X_att = encoder_classify(dataset + '_' + attention + 'x',
		                                               'x', 'att', dataset)
		Y, Y_y, t_Y, t_Y_y, t_2_Y, t_2_Y_y, t_Y_att = encoder_classify(dataset + '_' + attention + 'y',
		                                               'y', 'att', dataset)
		Z, Z_y, t_Z, t_Z_y, t_2_Z, t_2_Z_y, t_Z_att = encoder_classify(dataset + '_' + attention + 'z',
		                                               'z', 'att', dataset)
	else:
		X, X_y, t_X, t_X_y, t_X_att = encoder_classify(dataset + '_' + attention + 'x', 'x', 'att', dataset)
		Y, Y_y, t_Y, t_Y_y, t_Y_att = encoder_classify(dataset + '_' + attention + 'y', 'y', 'att', dataset)
		Z, Z_y, t_Z, t_Z_y, t_Z_att = encoder_classify(dataset + '_' + attention + 'z', 'z', 'att', dataset)
	assert X_y.tolist() == Y_y.tolist() and Y_y.tolist() == Z_y.tolist()
	assert t_X_y.tolist() == t_Y_y.tolist() and t_Y_y.tolist() == t_Z_y.tolist()
	X = np.column_stack([X,Y,Z])
	y = X_y
	t_X = np.column_stack([t_X, t_Y, t_Z])
	t_y = t_X_y
	if dataset == 'IAS':
		t_2_X = np.column_stack([t_2_X, t_2_Y, t_2_Z])
		t_2_y = t_2_X_y
	if train_flag == '0' or train_flag == '1':
	# direct evaluation
		print('Train a recognition network on CAGEs')
		if dataset == 'IAS':
			if Model == 'rev_rec_plus':
				encoder_classify_union_directly_IAS(X, y, t_X, t_y, t_2_X, t_2_y, './Models/CAGEs_RN_models',
				                                    dataset + '_RN_' + manner + '_' + str(time_steps)
				                                    + '_' + str(temperature) + '_' + str(Frozen) + 'pre_' + Model,
				                                    dataset)
			else:
				encoder_classify_union_directly_IAS(X, y, t_X, t_y, t_2_X, t_2_y, './Models/CAGEs_RN_models',
				                                dataset + '_' + attention + '_RN_' + manner + '_' + str(time_steps)
				                                    + '_' + str(temperature) + '_' + str(Frozen) + 'pre_' + Model, dataset)
		else:
			if Model == 'rev_rec_plus':
				encoder_classify_union_directly(X,y,t_X,t_y,'./Models/CAGEs_RN_models',
	                       dataset + '_RN_' + manner + '_' + str(time_steps) + '_' + str(temperature) + '_' + str(Frozen) + view + 'pre_' + Model, dataset)
			else:
				encoder_classify_union_directly(X, y, t_X, t_y, './Models/CAGEs_RN_models',
				                                dataset + '_' + attention + '_RN_' + manner + '_' + str(
					                                time_steps) + '_' + str(temperature) + '_' + str(
					                                Frozen) + view + 'pre_' + Model, dataset)
		print ('Dataset: %s' % dataset)
		if dataset == 'KS20':
			print ('View: %s' % view)
	if Model == 'rev_rec_plus':
		evaluate_reid('./Models/CAGEs_RN_models/' + dataset + '_RN_' + manner + '_' + str(time_steps)
		              + '_' + str(temperature) + '_' + Frozen + view + 'pre_' + Model)
	else:
		evaluate_reid('./Models/CAGEs_RN_models/' + dataset + '_' + attention + '_RN_' + manner + '_' + str(time_steps)
		              + '_' + str(temperature) + '_' + Frozen + view + 'pre_' + Model)

def get_inputs():
	inputs = tf.placeholder(tf.float32, [batch_size, time_steps, series_length], name='inputs')
	targets = tf.placeholder(tf.float32, [batch_size, time_steps, series_length], name='targets')
	learning_rate = tf.Variable(0.001, trainable=False, dtype=tf.float32, name='learning_rate')
	learning_rate_decay_op = learning_rate.assign(learning_rate * 0.5)
	target_sequence_length = tf.placeholder(tf.int32, (None, ), name='target_sequence_length')
	max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
	source_sequence_length = tf.placeholder(tf.int32, (None, ), name='source_sequence_length')
	keep_prob = tf.placeholder(tf.float32, name='keep_prob')
	return inputs, targets, learning_rate, learning_rate_decay_op, target_sequence_length, max_target_sequence_length, source_sequence_length, keep_prob

def get_data_KGBD(dimension, fr):
	input_data = np.load('Datasets/'+ frames_ps +'KGBD_train_npy_data/source_' + dimension + '_KGBD_' + str(fr) + '.npy')
	input_data = input_data.reshape([-1,time_steps, series_length])
	# rev_rec
	if Model == 'rev_rec' or Model == 'rev_rec_plus':
		targets = np.load('Datasets/'+ frames_ps +'KGBD_train_npy_data/target_' + dimension + '_KGBD_' + str(fr) + '.npy')
		targets = targets.reshape([-1,time_steps, series_length])
		input_data = input_data.tolist()
		targets = targets.tolist()
	# prediction
	elif Model == 'prediction':
		targets = np.concatenate((input_data[1:,:,:], input_data[-1,:, :].reshape([1, time_steps, series_length])), axis=0)
		# input_data = input_data[:-1]
		input_data = input_data.tolist()
		targets = targets.tolist()
	# sorting
	elif Model == 'sorting':
		targets = copy.deepcopy(input_data)
		for i in range(input_data.shape[0]):
			permutation_ = np.random.permutation(time_steps)
			input_data[i] = input_data[i, permutation_]
		input_data = input_data.tolist()
		targets = targets.tolist()
	return input_data, targets

def get_data_KinectReID(dimension, fr):
	input_data = np.load('Datasets/'+ frames_ps +'KinectReID_train_npy_data/source_' + dimension + '_KinectReID_' + str(fr) + '.npy')
	input_data = input_data.reshape([-1,time_steps, series_length])
	input_data = input_data.tolist()
	targets = np.load('Datasets/'+ frames_ps +'KinectReID_train_npy_data/target_' + dimension + '_KinectReID_' + str(fr) + '.npy')
	targets = targets.reshape([-1,time_steps, series_length])
	targets = targets.tolist()
	return input_data, targets

def get_data_KS20(dimension, fr):
	global  view
	if view != '':
		view_dir = view + '/'
	else:
		view_dir = ''
	input_data = np.load('Datasets/'+ frames_ps + view_dir +'KS20_train_npy_data/source_' + dimension + '_KS20_' + str(fr) + '.npy')
	input_data = input_data.reshape([-1,time_steps, series_length])
	if Model == 'rev_rec' or Model == 'rev_rec_plus':
		input_data = input_data.tolist()
		targets = np.load('Datasets/'+ frames_ps + view_dir +'KS20_train_npy_data/target_' + dimension + '_KS20_' + str(fr) + '.npy')
		targets = targets.reshape([-1,time_steps, series_length])
		targets = targets.tolist()
	# prediction
	elif Model == 'prediction':
		targets = np.concatenate((input_data[1:, :, :], input_data[-1, :, :].reshape([1, time_steps, series_length])),
		                         axis=0)
		# input_data = input_data[:-1]
		input_data = input_data.tolist()
		targets = targets.tolist()
	# 2. permutation
	elif Model == 'sorting':
		targets = copy.deepcopy(input_data)
		for i in range(input_data.shape[0]):
			permutation_ = np.random.permutation(time_steps)
			input_data[i] = input_data[i, permutation_]
		input_data = input_data.tolist()
		targets = targets.tolist()
	return input_data, targets

def get_data_IAS(dimension, fr):
	input_data = np.load('Datasets/'+ frames_ps +'IAS_train_npy_data/source_' + dimension + '_IAS_' + str(fr) + '.npy')
	input_data = input_data.reshape([-1, time_steps, series_length])
	if Model == 'rev_rec' or Model == 'rev_rec_plus':
		input_data = input_data.tolist()
		targets = np.load('Datasets/'+ frames_ps +'IAS_train_npy_data/target_' + dimension + '_IAS_' + str(fr) + '.npy')
		targets = targets.reshape([-1,time_steps, series_length])
		targets = targets.tolist()
	# prediction
	elif Model == 'prediction':
		targets = np.concatenate((input_data[1:, :, :], input_data[-1, :, :].reshape([1, time_steps, series_length])),
		                         axis=0)
		# input_data = input_data[:-1]
		input_data = input_data.tolist()
		targets = targets.tolist()
	# 2. sorting
	elif Model == 'sorting':
		targets = copy.deepcopy(input_data)
		for i in range(input_data.shape[0]):
			permutation_ = np.random.permutation(time_steps)
			input_data[i] = input_data[i, permutation_]
		input_data = input_data.tolist()
		targets = targets.tolist()
	return input_data, targets

def get_data_BIWI(dimension, fr):
	input_data = np.load('Datasets/' + frames_ps + 'BIWI_train_npy_data/source_' + dimension + '_BIWI_' + str(fr) + '.npy')
	input_data = input_data.reshape([-1, time_steps, series_length])
	if Model == 'rev_rec' or Model == 'rev_rec_plus':
		input_data = input_data.tolist()
		targets = np.load('Datasets/'+ frames_ps +'BIWI_train_npy_data/target_' + dimension + '_BIWI_' + str(fr) + '.npy')
		targets = targets.reshape([-1,time_steps, series_length])
		targets = targets.tolist()
	# prediction
	elif Model == 'prediction':
		targets = np.concatenate((input_data[1:, :, :], input_data[-1, :, :].reshape([1, time_steps, series_length])),
		                         axis=0)
		# input_data = input_data[:-1]
		input_data = input_data.tolist()
		targets = targets.tolist()
	# 2. permutation
	elif Model == 'sorting':
		targets = copy.deepcopy(input_data)
		for i in range(input_data.shape[0]):
			permutation_ = np.random.permutation(time_steps)
			input_data[i] = input_data[i, permutation_]
		input_data = input_data.tolist()
		targets = targets.tolist()
	# t_input_data = np.load('Datasets/'+ frames_ps +'BIWI_test_npy_data/t_source_' + dimension + '_BIWI_' + str(fr) + '.npy')
	# t_input_data = t_input_data.reshape([-1, time_steps, series_length])
	# t_input_data = t_input_data.tolist()
	# t_targets = np.load('Datasets/'+ frames_ps +'BIWI_test_npy_data/t_target_' + dimension + '_BIWI_' + str(fr) + '.npy')
	# t_targets = t_targets.reshape([-1, time_steps, series_length])
	# t_targets = t_targets.tolist()
	# # return input_data, targets, t_input_data, t_targets
	return input_data[:-len(input_data)//3], targets[:-len(input_data)//3], input_data[-len(input_data)//3:], targets[-len(input_data)//3:]


def pad_batch(batch_data, pad_int):
	'''
	padding the first skeleton of target sequence with zeros —— Z
	transform the target sequence (1,2,3,...,f) to (Z,1,2,3,...,f-1) as input to decoder in training
	parameters：
	- batch_data
	- pad_int: position (0)
	'''
	max_sentence = max([len(sentence) for sentence in batch_data])
	return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in batch_data]


def get_batches(targets, sources, batch_size, source_pad_int, target_pad_int):
	for batch_i in range(0, len(sources) // batch_size):
		start_i = batch_i * batch_size
		sources_batch = sources[start_i:start_i + batch_size]
		targets_batch = targets[start_i:start_i + batch_size]

		# transform the target sequence (1,2,3,...,f) to (Z,1,2,3,...,f-1) as input to decoder in training
		pad_sources_batch = np.array(pad_batch(sources_batch, source_pad_int))
		pad_targets_batch = np.array(pad_batch(targets_batch, target_pad_int))

		# record the lengths of sequence (not neccessary)
		targets_lengths = []
		for target in targets_batch:
			targets_lengths.append(len(target))

		source_lengths = []
		for source in sources_batch:
			source_lengths.append(len(source))

		yield pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths

def get_batches_plain(targets, sources, batch_size, source_pad_int, target_pad_int):
	for batch_i in range(0, len(sources) // batch_size):
		start_i = batch_i * batch_size
		sources_batch = sources[start_i:start_i + batch_size]
		targets_batch = sources[start_i:start_i + batch_size]

		# transform the target sequence (1,2,3,...,f) to (Z,1,2,3,...,f-1) as input to decoder in training
		pad_sources_batch = np.array(pad_batch(sources_batch, source_pad_int))
		pad_targets_batch = np.array(pad_batch(targets_batch, target_pad_int))

		# record the lengths of sequence (not neccessary)
		targets_lengths = []
		for target in targets_batch:
			targets_lengths.append(len(target))

		source_lengths = []
		for source in sources_batch:
			source_lengths.append(len(source))

		yield pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths


def GE(input_data, rnn_size, num_layers, source_sequence_length, encoding_embedding_size):
	'''
	Gait Encoder (GE)

	Parameters:
	- input_data: skeleton sequences (X,Y,Z series)
	- rnn_size: 256
	- num_layers: 2
	- source_sequence_length: 
	- encoding_embedding_size: embedding size
	'''

	encoder_embed_input = input_data

	def get_lstm_cell(rnn_size):
		lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
		# if use_dropout:
		# 	lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=0.5)
		return lstm_cell
	if Bi_LSTM:
		fw_cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(rnn_size) for _ in range(num_layers)])
		bw_cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(rnn_size) for _ in range(num_layers)])

		encoder_output, encoder_state = \
			tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, encoder_embed_input, sequence_length=source_sequence_length, dtype=tf.float32)
		weights = fw_cell.variables
		# print(encoder_state)
		# print('1')
		# print(encoder_output)
		# exit(1)
	else:
		cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(rnn_size) for _ in range(num_layers)])
		encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, encoder_embed_input,
	                                                  sequence_length=source_sequence_length, dtype=tf.float32)
		weights = cell.variables
	if Bi_LSTM:
		# print(encoder_state)
		# exit(1)
		c_1 = encoder_state[1][1][0] + encoder_state[0][1][0]
		h_1 = encoder_state[1][1][1] + encoder_state[0][1][1]
		c_0 = encoder_state[1][0][0] + encoder_state[0][0][0]
		h_0 = encoder_state[1][0][1] + encoder_state[0][0][1]
		# bidirectional_rnn/fw/fw/transpose_1:0
		# ReverseSequence: 0
		return encoder_output[0], (rnn_cell_impl.LSTMStateTuple(c_0, h_0), rnn_cell_impl.LSTMStateTuple(c_1, h_1)), weights, source_sequence_length
	else:
		return encoder_output, encoder_state, weights, source_sequence_length


def GD(decoding_embedding_size, num_layers, rnn_size,
                   target_sequence_length, source_sequence_length, max_target_sequence_length, encoder_output, encoder_state, decoder_input):
	'''
	Gait Decoder (GD)

	parameters：
	- decoding_embedding_size: embedding size
	- num_layers: 2
	- rnn_size: 256
	- target_sequence_length: 6
	- max_target_sequence_length: 6
	- encoder_state: gait encoded state
	- decoder_input:
	'''

	decoder_embed_input = decoder_input

	def get_decoder_cell(rnn_size):
		decoder_cell = tf.contrib.rnn.LSTMCell(rnn_size,
		                                       initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
		return decoder_cell

	cell = tf.contrib.rnn.MultiRNNCell([get_decoder_cell(rnn_size) for _ in range(num_layers)])

	if use_attention:
		attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=rnn_size, memory=encoder_output,
		 memory_sequence_length=source_sequence_length)
		cell = tf.contrib.seq2seq.AttentionWrapper(cell=cell, attention_mechanism=attention_mechanism,
	                                    attention_layer_size=rnn_size, alignment_history=True, output_attention=True,
	                                               name='Attention_Wrapper')
	# FC layer
	output_layer = Dense(series_length,
	                     use_bias=True,
	                     kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

	with tf.variable_scope("decode"):
		training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
		                                                    sequence_length=target_sequence_length,
		                                                    time_major=False)
		if not use_attention:
			training_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
			                                                   training_helper,
			                                                   encoder_state,
			                                                   output_layer)
		else:
			decoder_initial_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32).clone(
				cell_state=encoder_state)
			training_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
			                                                   training_helper,
			                                                   initial_state=decoder_initial_state,
			                                                   output_layer=output_layer,
			                                                   )
		training_decoder_output, training_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
		                                                               impute_finished=True,
		                                                               maximum_iterations=max_target_sequence_length)
	with tf.variable_scope("decode", reuse=True):
		def initialize_fn():
			finished = tf.tile([False], [batch_size])
			start_inputs = decoder_embed_input[:, 0]
			return (finished, start_inputs)

		def sample_fn(time, outputs, state):
			del time, state
			return tf.constant([0] * batch_size)

		def next_inputs_fn(time, outputs, state, sample_ids):
			del sample_ids
			finished = time >= tf.shape(decoder_embed_input)[1]
			all_finished = tf.reduce_all(finished)
			next_inputs = tf.cond(
				all_finished,
				lambda: tf.zeros_like(outputs),
				lambda: outputs)
			return (finished, next_inputs, state)

		predicting_helper = tf.contrib.seq2seq.CustomHelper(initialize_fn=initialize_fn,
	                      sample_fn=sample_fn,
	                      next_inputs_fn=next_inputs_fn)

		if not use_attention:
			predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
			                                                     predicting_helper,
			                                                     encoder_state,
			                                                     output_layer)
		else:
			decoder_initial_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32).clone(
				cell_state=encoder_state)
			predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
			                                                     predicting_helper,
			                                                     initial_state=decoder_initial_state,
			                                                     output_layer=output_layer)

		predicting_decoder_output, predicting_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
		                                                                 impute_finished=True,
		                                                                 maximum_iterations=max_target_sequence_length)

	return training_decoder_output, predicting_decoder_output, training_decoder_state, predicting_decoder_state


def process_decoder_input(data, batch_size):
    '''
    transform the target sequence (1,2,3,...,f) to (Z,1,2,3,...,f-1) as input to decoder in training
    '''
    ending = tf.strided_slice(data, [0, 0, 0], [batch_size, -1, series_length], [1, 1, 1])
    decoder_input = tf.concat([tf.fill([batch_size, time_steps, series_length], 0.), ending], 1)
    return decoder_input

def encoder_decoder(input_data, targets, lr, target_sequence_length,
                  max_target_sequence_length, source_sequence_length,
                  encoder_embedding_size, decoder_embedding_size,
                  rnn_size, num_layers):
	encoding_embedding_size = 128
	decoding_embedding_size = 128
	encoder_output, encoder_state, weights, source_sequence_length = GE(input_data,
	                                     rnn_size,
	                                     num_layers,
	                                     source_sequence_length,
	                                     encoding_embedding_size)
	decoder_input = process_decoder_input(targets, batch_size)
	lstm_weights_1 = tf.Variable(weights[0], dtype=tf.float32, name='lstm_weights_layer_1')
	lstm_weights_2 = tf.Variable(weights[3], dtype=tf.float32, name='lstm_weights_layer_2')
	training_decoder_output, predicting_decoder_output, training_state, predicting_state = GD(
	                                                                    decoding_embedding_size,
	                                                                    num_layers,
	                                                                    rnn_size,
	                                                                    target_sequence_length,
	                                                                    source_sequence_length,
	                                                                    max_target_sequence_length,
	                                                                    encoder_output,
	                                                                    encoder_state,
	                                                                    decoder_input)
	if use_attention:
		attention_matrices = training_state.alignment_history.stack()
		return training_decoder_output, predicting_decoder_output, lstm_weights_1, lstm_weights_2, attention_matrices
	else:
		return training_decoder_output, predicting_decoder_output, lstm_weights_1, lstm_weights_2

def train(folder_name, dim, train_dataset=False):
	global series_length, time_steps, dataset, attention, Frozen, epochs
	if train_dataset == 'KGBD':
		input_data_, targets_ = get_data_KGBD(dim, fr=time_steps)
		epochs = 150
	elif train_dataset == 'IAS':
		input_data_, targets_ = get_data_IAS(dim, fr=time_steps)
	elif train_dataset == 'BIWI':
		input_data_, targets_, t_input_data_, t_targets_ = get_data_BIWI(dim, fr=time_steps)
	elif train_dataset == 'KinectReID':
		input_data_, targets_ = get_data_KinectReID(dim, fr=time_steps)
	elif train_dataset == 'KS20':
		input_data_, targets_ = get_data_KS20(dim, fr=time_steps)
	else:
		raise Error('No dataset is chosen!')
	if not Reverse:
		targets_ = copy.deepcopy(input_data_)
		if train_dataset == 'BIWI':
			t_targets_ = copy.deepcopy(t_input_data_)
	train_graph = tf.Graph()
	encoding_embedding_size = 128
	decoding_embedding_size = 128
	with train_graph.as_default():
		input_data, targets, lr, lr_decay_op, target_sequence_length, max_target_sequence_length, source_sequence_length, keep_prob = get_inputs()
		if use_attention:
			training_decoder_output, predicting_decoder_output, lstm_weights_1, lstm_weights_2, attention_matrices = encoder_decoder(input_data,
			                                                                   targets,
			                                                                   lr,
			                                                                   target_sequence_length,
			                                                                   max_target_sequence_length,
			                                                                   source_sequence_length,
			                                                                   encoding_embedding_size,
			                                                                   decoding_embedding_size,
			                                                                   rnn_size,
			                                                                   num_layers)
		else:
			training_decoder_output, predicting_decoder_output, lstm_weights_1, lstm_weights_2 = encoder_decoder(input_data,
			                                                                   targets,
			                                                                   lr,
			                                                                   target_sequence_length,
			                                                                   max_target_sequence_length,
			                                                                   source_sequence_length,
			                                                                   encoding_embedding_size,
			                                                                   decoding_embedding_size,
			                                                                   rnn_size,
			                                                                   num_layers)
		training_decoder_output = training_decoder_output.rnn_output
		predicting_output = tf.identity(predicting_decoder_output.rnn_output, name='predictions')
		training_output = tf.identity(predicting_decoder_output.rnn_output, name='train_output')
		train_loss = tf.reduce_mean(tf.nn.l2_loss(training_decoder_output - targets))
		real_loss = tf.identity(train_loss, name='real_loss')
		encoder_output = train_graph.get_tensor_by_name('rnn/transpose_1:0')
		if use_attention:
			attention_matrices = tf.identity(attention_matrices, name='train_attention_matrix')

			# Locality-oriented attention loss
			if attention == 'LA' or attention == 'LA-R':
				objective_attention = np.ones(shape=[time_steps, time_steps])
				for index, _ in enumerate(objective_attention.tolist()):
					if not Reverse:
						pt = index
					else:
						pt = time_steps - 1 - index
					D = time_steps
					objective_attention[index][pt] = 1
					for i in range(1, D+1):
						if pt + i <= time_steps - 1:
							objective_attention[index][min(pt + i, time_steps - 1)] = np.exp(-(i)**2/(2*(D/2)**2))
						if pt-i >= 0:
							objective_attention[index][max(pt-i, 0)] = np.exp(-(i)**2/(2*(D/2)**2))
					objective_attention[index][pt] = 1
				objective_attention = np.tile(objective_attention, [batch_size, 1, 1])
				objective_attention = objective_attention.swapaxes(1,0)
				att_loss = tf.reduce_mean(tf.nn.l2_loss(attention_matrices - attention_matrices * objective_attention))
				train_loss += att_loss
			if Frozen == '0':
				attention_trans = tf.transpose(attention_matrices, [1, 0, 2])
				AGEs = tf.matmul(attention_trans, encoder_output)
				AGEs = tf.reshape(AGEs, [batch_size, -1])
				first_size = rnn_size * time_steps
				# C_input = tf.placeholder(tf.float32, [None, first_size], name='C_input')
				C_lr = tf.Variable(0.0005, trainable=False, dtype=tf.float32, name='learning_rate')
				# learning_rate_1:0
				W1 = tf.Variable(tf.random_normal([first_size, rnn_size]), name='W1')
				b1 = tf.Variable(tf.zeros(shape=[rnn_size, ]), name='b1')
				Wx_plus_b1 = tf.matmul(AGEs, W1) + b1
				l1 = tf.nn.relu(Wx_plus_b1)
				W = tf.Variable(tf.random_normal([rnn_size, rnn_size]), name='W')
				b = tf.Variable(tf.zeros(shape=[rnn_size, ], name='b'))
				contrast_v = tf.matmul(l1, W) + b
				# print (encoder_output)
				# print(attention_trans)
				# print(AGEs)
				# print(contrast_v)
				# exit(1)
				# add_2:0
				# with tf.name_scope("C_train"):
				t = temperature
				C_optimizer = tf.train.AdamOptimizer(learning_rate, name="Adam_C")
				z1 = contrast_v[1:]
				z2 = contrast_v[:-1]
				z = tf.concat((z1, z2), axis=0)
				unorm_sim = tf.matmul(z, tf.transpose(z))
				z_norm = tf.sqrt(tf.reduce_sum(tf.pow(z, 2), axis=1))
				z_norm = tf.expand_dims(z_norm, axis=1)
				norm_matrix = tf.matmul(z_norm, tf.transpose(z_norm))
				sim = unorm_sim / (t * norm_matrix)
				C_loss = tf.zeros(1)
				sample_num = batch_size - 1
				for i in range(sample_num):
					C_loss = C_loss - tf.log(
						tf.exp(sim[i, i + sample_num]) / (tf.reduce_sum(tf.exp(sim[i, :])) - tf.exp(sim[i, i])))
					C_loss = C_loss - tf.log(tf.exp(sim[i + sample_num, i]) / (
							tf.reduce_sum(tf.exp(sim[i + sample_num, :])) - tf.exp(
						sim[i + sample_num, i + sample_num])))
				C_loss = C_loss / (2 * sample_num)
				# C_train_op = C_optimizer.minimize(C_loss)
				train_loss += C_loss
					# print (C_lr, C_loss, contrast_v)
					# <tf.Variable 'learning_rate_1:0' shape=() dtype=float32_ref>
					# Tensor("C_train/truediv_255:0", shape=(1,), dtype=float32)
					# Tensor("add_2:0", shape=(128, 256), dtype=float32)
					# exit(1)
		else:
			h_s = tf.reshape(encoder_output, [batch_size, -1])
			first_size = rnn_size * time_steps
			C_lr = tf.Variable(0.0005, trainable=False, dtype=tf.float32, name='learning_rate')
			W1 = tf.Variable(tf.random_normal([first_size, rnn_size]), name='W1')
			b1 = tf.Variable(tf.zeros(shape=[rnn_size, ]), name='b1')
			Wx_plus_b1 = tf.matmul(h_s, W1) + b1
			l1 = tf.nn.relu(Wx_plus_b1)
			W = tf.Variable(tf.random_normal([rnn_size, rnn_size]), name='W')
			b = tf.Variable(tf.zeros(shape=[rnn_size, ], name='b'))
			contrast_v = tf.matmul(l1, W) + b
			t = temperature
			C_optimizer = tf.train.AdamOptimizer(learning_rate, name="Adam_C")
			z1 = contrast_v[1:]
			z2 = contrast_v[:-1]
			z = tf.concat((z1, z2), axis=0)
			unorm_sim = tf.matmul(z, tf.transpose(z))
			z_norm = tf.sqrt(tf.reduce_sum(tf.pow(z, 2), axis=1))
			z_norm = tf.expand_dims(z_norm, axis=1)
			norm_matrix = tf.matmul(z_norm, tf.transpose(z_norm))
			sim = unorm_sim / (t * norm_matrix)
			C_loss = tf.zeros(1)
			sample_num = batch_size - 1
			for i in range(sample_num):
				C_loss = C_loss - tf.log(
					tf.exp(sim[i, i + sample_num]) / (tf.reduce_sum(tf.exp(sim[i, :])) - tf.exp(sim[i, i])))
				C_loss = C_loss - tf.log(tf.exp(sim[i + sample_num, i]) / (
						tf.reduce_sum(tf.exp(sim[i + sample_num, :])) - tf.exp(
					sim[i + sample_num, i + sample_num])))
			C_loss = C_loss / (2 * sample_num)
			train_loss += C_loss
		l2 = lambda_l2_reg * sum(
			tf.nn.l2_loss(tf_var)
			for tf_var in tf.trainable_variables()
			if not ("noreg" in tf_var.name or "Bias" in tf_var.name)
		)
		# train_loss += att_loss
		cost = tf.add(l2, train_loss, name='cost')

		with tf.name_scope("optimization"):
			# Optimizer
			optimizer = tf.train.AdamOptimizer(lr, name='Adam')
			# Gradient Clipping
			gradients = optimizer.compute_gradients(cost)
			capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
			train_op = optimizer.apply_gradients(capped_gradients, name='train_op')

		# Contrast learning after freezing AGEs
		if Frozen == '1':
			first_size = rnn_size * time_steps
			C_input = tf.placeholder(tf.float32, [None, first_size], name='C_input')
			lr = tf.Variable(0.0005, trainable=False, dtype=tf.float32, name='learning_rate')
			# learning_rate_1:0
			W1 = tf.Variable(tf.random_normal([first_size, rnn_size]), name='W1')
			b1 = tf.Variable(tf.zeros(shape=[rnn_size, ]), name='b1')
			Wx_plus_b1 = tf.matmul(C_input, W1) + b1
			l1 = tf.nn.relu(Wx_plus_b1)
			W = tf.Variable(tf.random_normal([rnn_size, rnn_size]), name='W')
			b = tf.Variable(tf.zeros(shape=[rnn_size, ], name='b'))
			contrast_v = tf.matmul(l1, W) + b
			# add_16:0
			with tf.name_scope("C_train"):
				t = temperature
				C_optimizer = tf.train.AdamOptimizer(learning_rate, name="Adam_C")
				z1 = contrast_v[1:]
				z2 = contrast_v[:-1]
				# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_input))

				# (batch_size-1, )
				z = tf.concat((z1, z2), axis=0)
				unorm_sim = tf.matmul(z, tf.transpose(z))
				z_norm = tf.sqrt(tf.reduce_sum(tf.pow(z, 2), axis=1))
				z_norm = tf.expand_dims(z_norm, axis=1)
				# z_norm = z_norm.unsqueeze(1)
				norm_matrix = tf.matmul(z_norm, tf.transpose(z_norm))
			# 	norm_matrix = z_norm.mm(z_norm.t())
				sim = unorm_sim / (t * norm_matrix)
				# print (z, unorm_sim, z_norm, norm_matrix, sim)
				# exit(1)
			# 	sim = unorm_sim / (self.t * norm_matrix)
			# 	# print(sim[batch_size+2, 5])
			# 	# print(sim[5, batch_size+2])
			# 	# exit(1)
			# 	loss = torch.zeros(1, requires_grad=True)
				C_loss = tf.zeros(1)
			# 	loss = Variable(loss.type(Tensor))
			# 	sample_num = z1.size(0)
				sample_num = batch_size - 1
			# 	for i in range(sample_num):
			# 		loss = loss - torch.log(
			# 			torch.exp(sim[i, i + sample_num]) / (torch.sum(torch.exp(sim[i, :])) - torch.exp(sim[i, i])))
			# 		loss = loss - torch.log(torch.exp(sim[i + sample_num, i]) / (
			# 					torch.sum(torch.exp(sim[i + sample_num, :])) - torch.exp(
			# 				sim[i + sample_num, i + sample_num])))
				for i in range(sample_num):
					C_loss = C_loss - tf.log(
						tf.exp(sim[i, i + sample_num]) / (tf.reduce_sum(tf.exp(sim[i, :])) - tf.exp(sim[i, i])))
					C_loss = C_loss - tf.log(tf.exp(sim[i + sample_num, i]) / (
							tf.reduce_sum(tf.exp(sim[i + sample_num, :])) - tf.exp(
						sim[i + sample_num, i + sample_num])))
				C_loss = C_loss / (2 * sample_num)

				# gradients = optimizer.compute_gradients(cost)
				# capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
				C_train_op = C_optimizer.minimize(C_loss)
				# correct_pred = tf.equal(tf.argmax(pred, 1),tf.argmax(y_input, 1))
				# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	input_data_ = np.array(input_data_)
	targets_ = np.array(targets_)
	# just not permuated first
	# permutation = np.random.permutation(input_data_.shape[0])
	# input_data_= input_data_[permutation]
	# targets_ = targets_[permutation]
	train_source = input_data_
	train_target = targets_

	train_source = train_source.tolist()
	train_target =train_target.tolist()
	# input_data_ = input_data_.tolist()
	# targets_ = targets_.tolist()
	valid_source = train_source[:batch_size]
	valid_target = train_target[:batch_size]

	# print(len(train_source), len(train_target), len(valid_source), len(valid_target))
	(valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths) = next(
		get_batches(valid_target, valid_source, batch_size, source_pad_int=0, target_pad_int=0))

	display_step = 50

	checkpoint = "./" + folder_name + "/trained_model.ckpt"
	best_checkpoint = './' + folder_name + '/best_model.ckpt'

	with tf.Session(graph=train_graph, config=config) as sess:
		sess.run(tf.global_variables_initializer())
		print('Begin Training on Dimension [' + dim.upper() + ']')
		train_loss = []
		test_loss = []
		c_train_loss = []
		losses = [0, 0, 0]
		loss_cnt = 0
		conv_cnt = 0
		best_val = 100000
		over_flag = False
		if use_attention:
			alignment_history = train_graph.get_tensor_by_name('train_attention_matrix:0')
		encoder_output = train_graph.get_tensor_by_name('rnn/transpose_1:0')
		for epoch_i in range(1, epochs + 1):
			if over_flag:
				break
			for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
					get_batches(train_target, train_source, batch_size, source_pad_int=0, target_pad_int=0)):
				# print (sources_batch[5, 3:, :])
				# print (sources_batch[6, :3, :])
				# print (sources_batch[9, 3:, :])
				# print (sources_batch[10, :3, :])
				# exit(1)
				if use_attention:
					if Frozen == '0':
						_, loss, c_loss, att, en_outputs, att_history = sess.run(
							[train_op, real_loss, C_loss, attention_matrices, encoder_output, alignment_history],
							{input_data: sources_batch,
							 targets: targets_batch,
							 lr: learning_rate,
							 C_lr: learning_rate,
							 target_sequence_length: targets_lengths,
							 source_sequence_length: sources_lengths,
							 keep_prob: 0.5})
					else:
						_, loss, att, en_outputs, att_history = sess.run([train_op, real_loss, attention_matrices, encoder_output, alignment_history],
						                   {input_data: sources_batch,
						                    targets: targets_batch,
						                    lr: learning_rate,
						                    target_sequence_length: targets_lengths,
						                    source_sequence_length: sources_lengths,
						                    keep_prob: 0.5})
						att_batch = []
						for index in range(en_outputs.shape[0]):
							t1 = np.reshape(en_outputs[index], [-1]).tolist()
							if use_attention and AGEs:
								weights = att_history[:, index, :]
								f_o = en_outputs[index, :, :]
								att_op = np.matmul(weights, f_o)
								# if manner == 'sc':
								att_op = np.reshape(att_op, [-1]).tolist()
								att_batch.append(att_op)
						att_batch = np.array(att_batch)
						_, c_loss, c_vec = sess.run([C_train_op, C_loss, contrast_v],
						                 {C_input: att_batch,
						                  lr: learning_rate
						                 })
						# print(c_loss)
								# else:
								# 	X.extend(att_op.tolist())
				else:
					_, loss = sess.run([train_op, real_loss],
					                        {input_data: sources_batch,
					                         targets: targets_batch,
					                         lr: learning_rate,
					                         target_sequence_length: targets_lengths,
					                         source_sequence_length: sources_lengths,
					                         keep_prob: 0.5})
			# if batch_i % display_step == 0:
			if epoch_i % 1 == 0:
				if Frozen == '0':
					validation_loss, c_loss = sess.run(
						[real_loss, C_loss],
						{input_data: valid_sources_batch,
						 targets: valid_targets_batch,
						 lr: learning_rate,
						 C_lr: learning_rate,
						 target_sequence_length: valid_targets_lengths,
						 source_sequence_length: valid_sources_lengths,
						 keep_prob: 1.0})
				else:
					validation_loss = sess.run(
						[real_loss],
						{input_data: valid_sources_batch,
						 targets: valid_targets_batch,
						 lr: learning_rate,
						 target_sequence_length: valid_targets_lengths,
						 source_sequence_length: valid_sources_lengths,
						 keep_prob: 1.0})
				# if epoch_i % 25 == 0 and validation_loss[0] < best_val:
				# 	saver = tf.train.Saver()
				# 	saver.save(sess, best_checkpoint)
				# 	print('The Best Model Saved Again')
				# 	best_val = validation_loss[0]
				train_loss.append(loss)
				if Frozen == '1':
					c_train_loss.append(c_loss[0])
					test_loss.append(validation_loss[0])
					losses[loss_cnt % 3] = validation_loss[0]
					print(
						'Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  - Validation loss: {:>6.3f} - Contrastive loss: {:>6.3f}'
						.format(epoch_i,
						        epochs,
						        batch_i,
						        len(train_source) // batch_size,
						        loss,
						        validation_loss[0],
						        c_loss[0]))
				else:
					c_train_loss.append(c_loss[0])
					test_loss.append(validation_loss)
					losses[loss_cnt % 3] = validation_loss
					print(
						'Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  - Validation loss: {:>6.3f} - Contrastive loss: {:>6.3f}'
						.format(epoch_i,
						        epochs,
						        batch_i,
						        len(train_source) // batch_size,
						        loss,
						        validation_loss,
						        c_loss[0]))
				loss_cnt += 1
				# print(losses)
				if conv_cnt > 0 and validation_loss[0] >= max(losses):
					over_flag = True
					break
				if (round(losses[(loss_cnt - 1) % 3], 5) == round(losses[loss_cnt % 3], 5)) and (round(losses[(loss_cnt - 2) % 3], 5)\
						== round(losses[loss_cnt % 3], 5))  :
					sess.run(lr_decay_op)
					conv_cnt += 1
		saver = tf.train.Saver()
		saver.save(sess, checkpoint)
		print('Model Trained and Saved')
		np.save(folder_name + '/train_loss.npy', np.array(train_loss))
		np.save(folder_name + '/test_loss.npy', np.array(test_loss))
		np.save(folder_name + '/c_train_loss.npy', np.array(c_train_loss))

def encoder_classify(model_name, dimension, type, dataset):
	global manner, transfer
	number = ord(dimension) - ord('x') + 1
	print('Run the gait encoding model to obtain CAGEs (%d / 3)' % number)
	global epochs, series_length, attention, Frozen, C_reid, view
	epochs = 200
	if view != '':
		view_dir = view + '/'
	else:
		view_dir = ''
	_input_data = np.load('Datasets/' + frames_ps + view_dir + dataset + '_train_npy_data/source_' + dimension + '_'+ dataset + '_'+str(time_steps)+'.npy')
	_input_data = _input_data.reshape([-1, time_steps, series_length])
	if Model == 'rev_rec' or Model == 'rev_rec_plus':
		_targets = np.load('Datasets/'+ frames_ps + view_dir + dataset + '_train_npy_data/target_' + dimension + '_' + dataset + '_'+str(time_steps)+'.npy')
		_targets = _targets.reshape([-1, time_steps, series_length])
	# prediction
	elif Model == 'prediction':
		_targets = np.concatenate((_input_data[1:, :, :], _input_data[-1, :, :].reshape([1, time_steps, series_length])),
		                         axis=0)
		# _input_data = _input_data[:-1]
	# permutation
	elif Model == 'sorting':
		_targets = copy.deepcopy(_input_data)
		for i in range(_input_data.shape[0]):
			permutation_ = np.random.permutation(time_steps)
			_input_data[i] = _input_data[i, permutation_]
	if dataset == 'IAS':
		t_input_data = np.load(
			'Datasets/' + frames_ps + dataset + '_test_npy_data/t_source_' + dimension + '_' + dataset + '-A_' + str(
				time_steps) + '.npy')
		t_input_data = t_input_data.reshape([-1, time_steps, series_length])
		if Model == 'rev_rec' or Model == 'rev_rec_plus':
			t_targets = np.load(
				'Datasets/' + frames_ps + dataset + '_test_npy_data/t_target_' + dimension + '_' + dataset + '-A_' + str(
					time_steps) + '.npy')
			t_targets = t_targets.reshape([-1, time_steps, series_length])
		elif Model == 'prediction':
			t_targets = np.concatenate(
				(t_input_data[1:, :, :], t_input_data[-1, :, :].reshape([1, time_steps, series_length])), axis=0)
			# t_input_data = t_input_data[:-1]
		# permutation
		elif Model == 'sorting':
			t_targets = copy.deepcopy(t_input_data)
			for i in range(t_input_data.shape[0]):
				permutation_ = np.random.permutation(time_steps)
				t_input_data[i] = t_input_data[i, permutation_]
		t_2_input_data = np.load(
			'Datasets/' + frames_ps + dataset + '_test_npy_data/t_source_' + dimension + '_' + dataset + '-B_' + str(
				time_steps) + '.npy')
		t_2_input_data = t_2_input_data.reshape([-1, time_steps, series_length])
		if Model == 'rev_rec' or Model == 'rev_rec_plus':
			t_2_targets = np.load(
				'Datasets/' + frames_ps + dataset + '_test_npy_data/t_target_' + dimension + '_' + dataset + '-B_' + str(
					time_steps) + '.npy')
			t_2_targets = t_2_targets.reshape([-1, time_steps, series_length])
		elif Model == 'prediction':
			t_2_targets = np.concatenate(
				(t_2_input_data[1:, :, :], t_2_input_data[-1, :, :].reshape([1, time_steps, series_length])), axis=0)
			# t_2_input_data = t_2_input_data[:-1]
		# permutation
		elif Model == 'sorting':
			t_2_targets = copy.deepcopy(t_2_input_data)
			for i in range(t_2_input_data.shape[0]):
				permutation_ = np.random.permutation(time_steps)
				t_2_input_data[i] = t_2_input_data[i, permutation_]
	else:
		t_input_data = np.load('Datasets/' + frames_ps + view_dir + dataset + '_test_npy_data/t_source_' + dimension + '_' + dataset + '_'+str(time_steps)+'.npy')
		t_input_data = t_input_data.reshape([-1, time_steps, series_length])
		if Model == 'rev_rec' or Model == 'rev_rec_plus':
			t_targets = np.load('Datasets/' + frames_ps + view_dir + dataset + '_test_npy_data/t_target_' + dimension + '_' + dataset + '_'+str(time_steps)+'.npy')
			t_targets = t_targets.reshape([-1, time_steps, series_length])
		elif Model == 'prediction':
			t_targets = np.concatenate(
				(t_input_data[1:, :, :], t_input_data[-1, :, :].reshape([1, time_steps, series_length])), axis=0)
			# t_input_data = t_input_data[:-1]
		# permutation
		elif Model == 'sorting':
			t_targets = copy.deepcopy(t_input_data)
			for i in range(t_input_data.shape[0]):
				permutation_ = np.random.permutation(time_steps)
				t_input_data[i] = t_input_data[i, permutation_]
	ids = np.load('Datasets/' + frames_ps + view_dir + dataset + '_train_npy_data/ids_' + dataset +'_'+str(time_steps)+'.npy')
	# print(ids)
	# exit(0)
	ids = ids.item()
	if not Reverse:
		_targets = copy.deepcopy(_input_data)
		t_targets = copy.deepcopy(t_input_data)
		if dataset == 'IAS':
			t_2_targets = copy.deepcopy(t_2_input_data)
	# print(ids)
	if dataset == 'IAS':
		t_ids = np.load('Datasets/' + frames_ps + dataset + '_test_npy_data/ids_' + dataset + '-A_'+str(time_steps)+'.npy')
		t_ids = t_ids.item()
		t_2_ids = np.load('Datasets/' + frames_ps + dataset + '_test_npy_data/ids_' + dataset + '-B_'+str(time_steps)+'.npy')
		t_2_ids = t_2_ids.item()
	else:
		t_ids = np.load('Datasets/' + frames_ps + view_dir + dataset + '_test_npy_data/ids_' + dataset + '_'+str(time_steps)+'.npy')
		t_ids = t_ids.item()
		# print(t_ids)
	if transfer == 'None':
		if Model == 'rev_rec_plus':
			# modify: using LA for the proposed reconstruction
			checkpoint = 'Models/Gait_Encoding_models/' + dataset + '_' + 'LA' + '_' + dimension + '_' + str(time_steps) \
		             + '_' + str(temperature) + '_' + str(Frozen) + view + 'pre_rev_rec' + "/trained_model.ckpt"
			checkpoint_1 = 'Models/Gait_Encoding_models/' + dataset + '_' + attention + '_' + dimension + '_' + str(
				time_steps) \
			             + '_' + str(temperature) + '_' + str(Frozen) + view + 'pre_prediction' + "/trained_model.ckpt"
			checkpoint_2 = 'Models/Gait_Encoding_models/' + dataset + '_' + attention + '_' + dimension + '_' + str(
				time_steps) \
			             + '_' + str(temperature) + '_' + str(Frozen) + view + 'pre_sorting' + "/trained_model.ckpt"
		else:
			checkpoint = 'Models/Gait_Encoding_models/' + dataset + '_' + attention + '_' + dimension + '_' + str(
				time_steps) \
			               + '_' + str(temperature) + '_' + str(
				Frozen) + view + 'pre_' + pre_task + "/trained_model.ckpt"
		# print(checkpoint)
		# print(checkpoint_1)
		# print(checkpoint_2)
		# exit(1)
	else:
		checkpoint = 'Models/Gait_Encoding_models/' + transfer + '_' + attention + '_' + dimension + '_' + str(
			time_steps) \
		             + '_' + str(temperature) + '_' + str(Frozen) + view + "/trained_model.ckpt"
	if Model != 'rev_rec_plus':
		loaded_graph = tf.Graph()
		with tf.Session(graph=loaded_graph, config=config) as sess:
			loader = tf.train.import_meta_graph(checkpoint + '.meta')
			loader.restore(sess, checkpoint)
			input_data = loaded_graph.get_tensor_by_name('inputs:0')
			targets = loaded_graph.get_tensor_by_name('targets:0')
			if Frozen == '1':
				contrast_v = loaded_graph.get_tensor_by_name('add_16:0')
				C_lr = loaded_graph.get_tensor_by_name('learning_rate_1:0')
				C_input = loaded_graph.get_tensor_by_name('C_inptuiut:0')
			else:
				contrast_v = loaded_graph.get_tensor_by_name("add_2:0")
				C_lr = loaded_graph.get_tensor_by_name('learning_rate_1:0')
			if Bi_LSTM:
				encoder_output = loaded_graph.get_tensor_by_name('bidirectional_rnn/fw/fw/transpose_1:0')
				encoder_c_1 = loaded_graph.get_tensor_by_name('bidirectional_rnn/fw/fw/while/Exit_3:0')
				encoder_h_1 = loaded_graph.get_tensor_by_name('bidirectional_rnn/fw/fw/while/Exit_4:0')
				encoder_c = loaded_graph.get_tensor_by_name('bidirectional_rnn/fw/fw/while/Exit_5:0')
				encoder_h = loaded_graph.get_tensor_by_name('bidirectional_rnn/fw/fw/while/Exit_6:0')
				encoder_output_bw = loaded_graph.get_tensor_by_name('ReverseSequence: 0')
				encoder_c_1_bw = loaded_graph.get_tensor_by_name('bidirectional_rnn/bw/bw/while/Exit_3:0')
				encoder_h_1_bw = loaded_graph.get_tensor_by_name('bidirectional_rnn/bw/bw/while/Exit_4:0')
				encoder_c_bw = loaded_graph.get_tensor_by_name('bidirectional_rnn/bw/bw/while/Exit_5:0')
				encoder_h_bw = loaded_graph.get_tensor_by_name('bidirectional_rnn/bw/bw/while/Exit_6:0')
				predictions = loaded_graph.get_tensor_by_name('predictions:0')
			else:
				encoder_output = loaded_graph.get_tensor_by_name('rnn/transpose_1:0')
				encoder_c_1 = loaded_graph.get_tensor_by_name('rnn/while/Exit_3:0')
				encoder_h_1 = loaded_graph.get_tensor_by_name('rnn/while/Exit_4:0')
				encoder_c = loaded_graph.get_tensor_by_name('rnn/while/Exit_5:0')
				encoder_h = loaded_graph.get_tensor_by_name('rnn/while/Exit_6:0')
				predictions = loaded_graph.get_tensor_by_name('predictions:0')
			# train_output = loaded_graph.get_tensor_by_name('train_output:0')
			if use_attention:
				alignment_history = loaded_graph.get_tensor_by_name('train_attention_matrix:0')
				# train_attention_matrix = loaded_graph.get_tensor_by_name('train_attention_matrix:0')
				attention_state = loaded_graph.get_tensor_by_name('decode/decoder/while/Exit_12:0')
				attention_weights = loaded_graph.get_tensor_by_name('decode/decoder/while/Exit_8:0')
				alignment = loaded_graph.get_tensor_by_name('decode/decoder/while/Exit_10:0')
			lr = loaded_graph.get_tensor_by_name('learning_rate:0')
			keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
			source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
			target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
			X = []
			C_X = []
			X_all_op = []
			X_final_op = []
			X_final_c = []
			X_final_h = []
			X_final_c1 = []
			X_final_h1 = []
			X_final_ch = []
			X_final_ch1 = []
			y = []
			C_y = []
			X_pred = []
			t_X = []
			t_C_X = []
			t_y = []
			t_C_y = []
			t_2_X = []
			t_2_C_X = []
			t_2_y = []
			t_2_C_y = []
			t_X_pred = []
			t_X_att = []
			# print(t_ids)
			# print(test_attention)
			ids_ = sorted(ids.items(), key=lambda item: item[0])
			t_ids_ = sorted(t_ids.items(), key=lambda item: item[0])
			if dataset == 'IAS':
				t_2_ids_ = sorted(t_2_ids.items(), key=lambda item: item[0])
			# print(ids_)
			# exit(1)
			for k, v in ids_:
				if len(v) == 0:
					print(k)
					continue
				if len(v) < batch_size:
					v.extend([v[0]] * (batch_size - len(v)))
				# print('%s - %d' % (k, len(v)))
				for batch_i in range(len(v) // batch_size):
					this_input = _input_data[v[batch_i * batch_size: (batch_i + 1) * batch_size]]
					this_targets = _targets[v[batch_i * batch_size: (batch_i + 1) * batch_size]]
					if use_attention:
						if Frozen == '1':
							en_outputs, en_c, en_h, en_c_1, en_h_1, pred, att_state, att_history, att, align = sess.run(
								[encoder_output, encoder_c,
								 encoder_h, encoder_c_1, encoder_h_1, predictions, attention_state, alignment_history,
								 attention_weights, alignment],
								{input_data: this_input,
								 targets: this_targets,
								 lr: learning_rate,
								 target_sequence_length: [time_steps] * batch_size,
								 source_sequence_length: [time_steps] * batch_size,
								 keep_prob: 1.0})
						else:
							en_outputs, c_vec, en_c, en_h, en_c_1, en_h_1, pred, att_state, att_history, att, align = sess.run(
								[encoder_output, contrast_v, encoder_c,
								 encoder_h, encoder_c_1, encoder_h_1, predictions, attention_state, alignment_history,
								 attention_weights, alignment],
								{input_data: this_input,
								 targets: this_targets,
								 lr: learning_rate,
								 C_lr: learning_rate,
								 target_sequence_length: [time_steps] * batch_size,
								 source_sequence_length: [time_steps] * batch_size,
								 keep_prob: 1.0})
							if C_reid == '1':
								C_X.extend(c_vec)
					else:
						if Bi_LSTM:
							en_outputs, encoder_outputs_bw, en_c, en_h, en_c_1, en_h_1, pred = sess.run(
								[encoder_output, encoder_output_bw, encoder_c,
								 encoder_h, encoder_c_1, encoder_h_1, predictions],
								{input_data: this_input,
								 targets: this_targets,
								 lr: learning_rate,
								 target_sequence_length: [time_steps] * batch_size,
								 source_sequence_length: [time_steps] * batch_size,
								 keep_prob: 1.0})
						else:
							en_outputs, en_c, en_h, en_c_1, en_h_1, pred = sess.run(
								[encoder_output, encoder_c,
								 encoder_h, encoder_c_1, encoder_h_1, predictions],
								{input_data: this_input,
								 targets: this_targets,
								 lr: learning_rate,
								 target_sequence_length: [time_steps] * batch_size,
								 source_sequence_length: [time_steps] * batch_size,
								 keep_prob: 1.0})
					att_batch = []
					for index in range(en_outputs.shape[0]):
						t1 = np.reshape(en_outputs[index], [-1]).tolist()
						t2 = np.reshape(en_c[index], [-1]).tolist()
						t3 = np.reshape(en_h[index], [-1]).tolist()
						t4 = np.reshape(en_c_1[index], [-1]).tolist()
						t5 = np.reshape(en_h_1[index], [-1]).tolist()
						if type == 'c':
							X.append(t2)
						elif type == 'ch':
							t3.extend(t2)
							X.append(t3)
						elif type == 'o':
							X.append(t1)
						elif type == 'oc':
							t1.extend(t2)
							X.append(t1)
						elif type == 'att':
							if use_attention and AGEs:
								weights = att_history[:, index, :]
								f_o = en_outputs[index, :, :]
								att_op = np.matmul(weights, f_o)
								if manner == 'sc' or Frozen == '1':
									att_op = np.reshape(att_op, [-1]).tolist()
									if manner == 'sc':
										X.append(att_op)
									else:
										att_batch.append(att_op)
								else:
									X.extend(att_op.tolist())
							else:
								f_o = en_outputs[index, :, :]
								if Bi_LSTM:
									f_o_bw = encoder_outputs_bw[index, :, :]
									f_o = f_o + f_o_bw
								if manner == 'sc':
									f_o = np.reshape(f_o, [-1]).tolist()
									X.append(f_o)
								else:
									X.extend(f_o.tolist())
						elif type == 'attc':
							weights = att_history[:, index, :]
							f_o = en_outputs[index, :, :]
							att_op = np.matmul(weights, f_o)
							att_op = np.reshape(att_op, [-1]).tolist()
							att_op.extend(t2)
							X.append(att_op)
					if Frozen == '1':
						att_batch = np.array(att_batch)
						[c_vec] = sess.run([contrast_v],
						                   {C_input: att_batch,
						                    lr: learning_rate
						                    })
						X.extend(c_vec)
					pred = pred.tolist()
					for index, i in enumerate(pred):
						pred[index].reverse()
					pred = np.array(pred)
					if manner == 'sc' or Frozen == '1' or C_reid == '1':
						y.extend([k] * batch_size)
					# C_y.extend([k] * batch_size)
					else:
						y.extend([k] * batch_size * time_steps)
			# exit(1)
			for k, v in t_ids_:
				flag = 0
				if len(v) == 0:
					continue
				if len(v) < batch_size:
					flag = batch_size - len(v)
					v.extend([v[0]] * (batch_size - len(v)))
				# print('%s - %d' % (k, len(v)))
				for batch_i in range(len(v) // batch_size):
					this_input = t_input_data[v[batch_i * batch_size: (batch_i + 1) * batch_size]]
					this_targets = t_targets[v[batch_i * batch_size: (batch_i + 1) * batch_size]]
					if use_attention:
						if Frozen == '1':
							en_outputs, en_c, en_h, en_c_1, en_h_1, pred, att_state, att_history, att, align = sess.run(
								[encoder_output, encoder_c,
								 encoder_h, encoder_c_1, encoder_h_1, predictions, attention_state, alignment_history,
								 attention_weights, alignment],
								{input_data: this_input,
								 targets: this_targets,
								 lr: learning_rate,
								 target_sequence_length: [time_steps] * batch_size,
								 source_sequence_length: [time_steps] * batch_size,
								 keep_prob: 1.0})
						else:
							en_outputs, c_vec, en_c, en_h, en_c_1, en_h_1, pred, att_state, att_history, att, align = sess.run(
								[encoder_output, contrast_v, encoder_c,
								 encoder_h, encoder_c_1, encoder_h_1, predictions, attention_state, alignment_history,
								 attention_weights, alignment],
								{input_data: this_input,
								 targets: this_targets,
								 lr: learning_rate,
								 C_lr: learning_rate,
								 target_sequence_length: [time_steps] * batch_size,
								 source_sequence_length: [time_steps] * batch_size,
								 keep_prob: 1.0})
							if C_reid == '1':
								t_C_X.extend(c_vec)
					else:
						if Bi_LSTM:
							en_outputs, encoder_outputs_bw, en_c, en_h, en_c_1, en_h_1, pred = sess.run(
								[encoder_output, encoder_output_bw, encoder_c,
								 encoder_h, encoder_c_1, encoder_h_1, predictions],
								{input_data: this_input,
								 targets: this_targets,
								 lr: learning_rate,
								 target_sequence_length: [time_steps] * batch_size,
								 source_sequence_length: [time_steps] * batch_size,
								 keep_prob: 1.0})
						else:
							en_outputs, en_c, en_h, en_c_1, en_h_1, pred = sess.run(
								[encoder_output, encoder_c,
								 encoder_h, encoder_c_1, encoder_h_1, predictions],
								{input_data: this_input,
								 targets: this_targets,
								 lr: learning_rate,
								 target_sequence_length: [time_steps] * batch_size,
								 source_sequence_length: [time_steps] * batch_size,
								 keep_prob: 1.0})
					if flag > 0:
						en_outputs = en_outputs[:-flag]
					att_batch = []
					for index in range(en_outputs.shape[0]):
						t1 = np.reshape(en_outputs[index], [-1]).tolist()
						t2 = np.reshape(en_c[index], [-1]).tolist()
						t3 = np.reshape(en_h[index], [-1]).tolist()
						t4 = np.reshape(en_c_1[index], [-1]).tolist()
						t5 = np.reshape(en_h_1[index], [-1]).tolist()
						if type == 'att':
							if use_attention and AGEs:
								weights = att_history[:, index, :]
								f_o = en_outputs[index, :, :]
								att_op = np.matmul(weights, f_o)
								if manner == 'sc' or Frozen == '1':
									att_op = np.reshape(att_op, [-1]).tolist()
									if manner == 'sc':
										t_X.append(att_op)
									else:
										att_batch.append(att_op)
								else:
									t_X.extend(att_op.tolist())
								t_X_att.append(weights)
							else:
								f_o = en_outputs[index, :, :]
								if Bi_LSTM:
									f_o_bw = encoder_outputs_bw[index, :, :]
									f_o = f_o_bw + f_o
								if manner == 'sc':
									f_o = np.reshape(f_o, [-1]).tolist()
									t_X.append(f_o)
								else:
									t_X.extend(f_o.tolist())
						elif type == 'attc':
							weights = att_history[:, index, :]
							f_o = en_outputs[index, :, :]
							att_op = np.matmul(weights, f_o)
							att_op = np.reshape(att_op, [-1]).tolist()
							att_op.extend(t2)
							t_X.append(att_op)
					if Frozen == '1':
						att_batch = np.array(att_batch)
						[c_vec] = sess.run([contrast_v],
						                   {C_input: att_batch,
						                    lr: learning_rate
						                    })
						t_X.extend(c_vec)
					pred = pred.tolist()
					for index, i in enumerate(pred):
						pred[index].reverse()
					pred = np.array(pred)
					if manner == 'sc' or Frozen == '1' or C_reid == '1':
						t_y.extend([k] * (batch_size - flag))
					# t_C_y.extend([k] * (batch_size - flag))
					else:
						t_y.extend([k] * (batch_size - flag) * time_steps)
			if dataset == 'IAS':
				for k, v in t_2_ids_:
					flag = 0
					if len(v) == 0:
						continue
					if len(v) < batch_size:
						flag = batch_size - len(v)
						v.extend([v[0]] * (batch_size - len(v)))
					# print('%s - %d' % (k, len(v)))
					for batch_i in range(len(v) // batch_size):
						this_input = t_2_input_data[v[batch_i * batch_size: (batch_i + 1) * batch_size]]
						this_targets = t_2_targets[v[batch_i * batch_size: (batch_i + 1) * batch_size]]
						if use_attention:
							if Frozen == '1':
								en_outputs, en_c, en_h, en_c_1, en_h_1, pred, att_state, att_history, att, align = sess.run(
									[encoder_output, encoder_c,
									 encoder_h, encoder_c_1, encoder_h_1, predictions, attention_state,
									 alignment_history,
									 attention_weights, alignment],
									{input_data: this_input,
									 targets: this_targets,
									 lr: learning_rate,
									 target_sequence_length: [time_steps] * batch_size,
									 source_sequence_length: [time_steps] * batch_size,
									 keep_prob: 1.0})
							else:
								en_outputs, c_vec, en_c, en_h, en_c_1, en_h_1, pred, att_state, att_history, att, align = sess.run(
									[encoder_output, contrast_v, encoder_c,
									 encoder_h, encoder_c_1, encoder_h_1, predictions, attention_state,
									 alignment_history,
									 attention_weights, alignment],
									{input_data: this_input,
									 targets: this_targets,
									 lr: learning_rate,
									 C_lr: learning_rate,
									 target_sequence_length: [time_steps] * batch_size,
									 source_sequence_length: [time_steps] * batch_size,
									 keep_prob: 1.0})
								if C_reid == '1':
									t_2_C_X.extend(c_vec)
						else:
							if Bi_LSTM:
								en_outputs, encoder_outputs_bw, en_c, en_h, en_c_1, en_h_1, pred = sess.run(
									[encoder_output, encoder_output_bw, encoder_c,
									 encoder_h, encoder_c_1, encoder_h_1, predictions],
									{input_data: this_input,
									 targets: this_targets,
									 lr: learning_rate,
									 target_sequence_length: [time_steps] * batch_size,
									 source_sequence_length: [time_steps] * batch_size,
									 keep_prob: 1.0})
							else:
								en_outputs, en_c, en_h, en_c_1, en_h_1, pred = sess.run(
									[encoder_output, encoder_c,
									 encoder_h, encoder_c_1, encoder_h_1, predictions],
									{input_data: this_input,
									 targets: this_targets,
									 lr: learning_rate,
									 target_sequence_length: [time_steps] * batch_size,
									 source_sequence_length: [time_steps] * batch_size,
									 keep_prob: 1.0})
						if flag > 0:
							en_outputs = en_outputs[:-flag]
						att_batch = []
						for index in range(en_outputs.shape[0]):
							t1 = np.reshape(en_outputs[index], [-1]).tolist()
							t2 = np.reshape(en_c[index], [-1]).tolist()
							t3 = np.reshape(en_h[index], [-1]).tolist()
							t4 = np.reshape(en_c_1[index], [-1]).tolist()
							t5 = np.reshape(en_h_1[index], [-1]).tolist()
							if type == 'att':
								if use_attention and AGEs:
									weights = att_history[:, index, :]
									f_o = en_outputs[index, :, :]
									att_op = np.matmul(weights, f_o)
									if manner == 'sc' or Frozen == '1':
										att_op = np.reshape(att_op, [-1]).tolist()
										if manner == 'sc':
											t_2_X.append(att_op)
										else:
											att_batch.append(att_op)
									else:
										t_2_X.extend(att_op.tolist())
								else:
									f_o = en_outputs[index, :, :]
									if Bi_LSTM:
										f_o_bw = encoder_outputs_bw[index, :, :]
										f_o = f_o_bw + f_o
									if manner == 'sc':
										f_o = np.reshape(f_o, [-1]).tolist()
										t_2_X.append(f_o)
									else:
										t_2_X.extend(f_o.tolist())
						if Frozen == '1':
							att_batch = np.array(att_batch)
							[c_vec] = sess.run([contrast_v],
							                   {C_input: att_batch,
							                    lr: learning_rate
							                    })
							t_2_X.extend(c_vec)
						if manner == 'sc' or Frozen == '1' or C_reid == '1':
							t_2_y.extend([k] * (batch_size - flag))
						# t_2_C_y.extend([k] * (batch_size - flag))
						else:
							t_2_y.extend([k] * (batch_size - flag) * time_steps)
		X_0 = np.array(X)
		y_0 = np.array(y)
		X_pred_0 = np.array(X_pred)
		t_X_pred_0 = np.array(t_X_pred)
		from sklearn.preprocessing import label_binarize
		ids_keys = sorted(list(ids.keys()))
		t_ids_keys = sorted(list(t_ids.keys()))
		classes = [i for i in ids_keys]
		t_classes = [i for i in t_ids_keys]
		t_y = label_binarize(t_y, classes=t_classes)
		if dataset == 'IAS':
			t_2_ids_keys = sorted(list(t_2_ids.keys()))
			t_2_classes = [i for i in t_2_ids_keys]
			t_2_y = label_binarize(t_2_y, classes=t_2_classes)
			t_2_y_0 = t_2_y
			t_2_X_0 = t_2_X
		t_y = np.array(t_y)
		if C_reid == '1':
			t_C_X = np.array(t_C_X)
		else:
			t_X = np.array(t_X)
		t_y_0 = t_y
		t_X_0 = t_X
	else:
		# checkpoint 0
		_targets = np.load(
			'Datasets/' + frames_ps + view_dir + dataset + '_train_npy_data/target_' + dimension + '_' + dataset + '_' + str(
				time_steps) + '.npy')
		_targets = _targets.reshape([-1, time_steps, series_length])
		loaded_graph = tf.Graph()
		with tf.Session(graph=loaded_graph, config=config) as sess:
			loader = tf.train.import_meta_graph(checkpoint + '.meta')
			loader.restore(sess, checkpoint)
			input_data = loaded_graph.get_tensor_by_name('inputs:0')
			targets = loaded_graph.get_tensor_by_name('targets:0')
			if Frozen == '1':
				contrast_v = loaded_graph.get_tensor_by_name('add_16:0')
				C_lr = loaded_graph.get_tensor_by_name('learning_rate_1:0')
				C_input = loaded_graph.get_tensor_by_name('C_inptuiut:0')
			else:
				contrast_v = loaded_graph.get_tensor_by_name("add_2:0")
				C_lr = loaded_graph.get_tensor_by_name('learning_rate_1:0')
			if Bi_LSTM:
				encoder_output = loaded_graph.get_tensor_by_name('bidirectional_rnn/fw/fw/transpose_1:0')
				encoder_c_1 = loaded_graph.get_tensor_by_name('bidirectional_rnn/fw/fw/while/Exit_3:0')
				encoder_h_1 = loaded_graph.get_tensor_by_name('bidirectional_rnn/fw/fw/while/Exit_4:0')
				encoder_c = loaded_graph.get_tensor_by_name('bidirectional_rnn/fw/fw/while/Exit_5:0')
				encoder_h = loaded_graph.get_tensor_by_name('bidirectional_rnn/fw/fw/while/Exit_6:0')
				encoder_output_bw = loaded_graph.get_tensor_by_name('ReverseSequence: 0')
				encoder_c_1_bw = loaded_graph.get_tensor_by_name('bidirectional_rnn/bw/bw/while/Exit_3:0')
				encoder_h_1_bw = loaded_graph.get_tensor_by_name('bidirectional_rnn/bw/bw/while/Exit_4:0')
				encoder_c_bw = loaded_graph.get_tensor_by_name('bidirectional_rnn/bw/bw/while/Exit_5:0')
				encoder_h_bw = loaded_graph.get_tensor_by_name('bidirectional_rnn/bw/bw/while/Exit_6:0')
				predictions = loaded_graph.get_tensor_by_name('predictions:0')
			else:
				encoder_output = loaded_graph.get_tensor_by_name('rnn/transpose_1:0')
				encoder_c_1 = loaded_graph.get_tensor_by_name('rnn/while/Exit_3:0')
				encoder_h_1 = loaded_graph.get_tensor_by_name('rnn/while/Exit_4:0')
				encoder_c = loaded_graph.get_tensor_by_name('rnn/while/Exit_5:0')
				encoder_h = loaded_graph.get_tensor_by_name('rnn/while/Exit_6:0')
				predictions = loaded_graph.get_tensor_by_name('predictions:0')
				# train_output = loaded_graph.get_tensor_by_name('train_output:0')
			if use_attention:
				alignment_history = loaded_graph.get_tensor_by_name('train_attention_matrix:0')
				# train_attention_matrix = loaded_graph.get_tensor_by_name('train_attention_matrix:0')
				attention_state = loaded_graph.get_tensor_by_name('decode/decoder/while/Exit_12:0')
				attention_weights = loaded_graph.get_tensor_by_name('decode/decoder/while/Exit_8:0')
				alignment = loaded_graph.get_tensor_by_name('decode/decoder/while/Exit_10:0')
			lr = loaded_graph.get_tensor_by_name('learning_rate:0')
			keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
			source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
			target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
			X = []
			C_X = []
			X_all_op = []
			X_final_op = []
			X_final_c = []
			X_final_h = []
			X_final_c1 = []
			X_final_h1 = []
			X_final_ch = []
			X_final_ch1 = []
			y = []
			C_y = []
			X_pred = []
			t_X = []
			t_C_X = []
			t_y = []
			t_C_y = []
			t_2_X = []
			t_2_C_X = []
			t_2_y = []
			t_2_C_y = []
			t_X_pred = []
			t_X_att = []
			# print(t_ids)
			# print(test_attention)
			ids_ = sorted(ids.items(), key=lambda item:item[0])
			t_ids_ = sorted(t_ids.items(), key=lambda item: item[0])
			if dataset == 'IAS':
				t_2_ids_ = sorted(t_2_ids.items(), key=lambda item: item[0])
			# print(ids_)
			# exit(1)
			for k, v in ids_:
				if len(v) == 0:
					print(k)
					continue
				if len(v) < batch_size:
					v.extend([v[0]] * (batch_size - len(v)))
				# print('%s - %d' % (k, len(v)))
				for batch_i in range(len(v) // batch_size):
					this_input = _input_data[v[batch_i * batch_size : (batch_i + 1) * batch_size]]
					this_targets = _targets[v[batch_i * batch_size : (batch_i + 1) * batch_size]]
					if use_attention:
						if Frozen == '1':
							en_outputs, en_c, en_h, en_c_1, en_h_1, pred, att_state, att_history, att, align = sess.run([encoder_output, encoder_c,
							encoder_h, encoder_c_1, encoder_h_1, predictions, attention_state, alignment_history, attention_weights, alignment],
							                      {input_data: this_input,
							                       targets: this_targets,
							                       lr: learning_rate,
							                       target_sequence_length: [time_steps] * batch_size,
							                       source_sequence_length: [time_steps] * batch_size,
							                       keep_prob: 1.0})
						else:
							en_outputs, c_vec, en_c, en_h, en_c_1, en_h_1, pred, att_state, att_history, att, align = sess.run(
								[encoder_output, contrast_v, encoder_c,
								 encoder_h, encoder_c_1, encoder_h_1, predictions, attention_state, alignment_history,
								 attention_weights, alignment],
								{input_data: this_input,
								 targets: this_targets,
								 lr: learning_rate,
								 C_lr: learning_rate,
								 target_sequence_length: [time_steps] * batch_size,
								 source_sequence_length: [time_steps] * batch_size,
								 keep_prob: 1.0})
							if C_reid == '1':
								C_X.extend(c_vec)
					else:
						if Bi_LSTM:
							en_outputs, encoder_outputs_bw, en_c, en_h, en_c_1, en_h_1, pred = sess.run(
								[encoder_output, encoder_output_bw, encoder_c,
								 encoder_h, encoder_c_1, encoder_h_1, predictions],
								{input_data: this_input,
								 targets: this_targets,
								 lr: learning_rate,
								 target_sequence_length: [time_steps] * batch_size,
								 source_sequence_length: [time_steps] * batch_size,
								 keep_prob: 1.0})
						else:
							en_outputs, en_c, en_h, en_c_1, en_h_1, pred = sess.run(
								[encoder_output, encoder_c,
								 encoder_h, encoder_c_1, encoder_h_1, predictions],
								{input_data: this_input,
								 targets: this_targets,
								 lr: learning_rate,
								 target_sequence_length: [time_steps] * batch_size,
								 source_sequence_length: [time_steps] * batch_size,
								 keep_prob: 1.0})
					att_batch = []
					for index in range(en_outputs.shape[0]):
						t1 = np.reshape(en_outputs[index], [-1]).tolist()
						t2 = np.reshape(en_c[index], [-1]).tolist()
						t3 = np.reshape(en_h[index], [-1]).tolist()
						t4 = np.reshape(en_c_1[index], [-1]).tolist()
						t5 = np.reshape(en_h_1[index], [-1]).tolist()
						if type == 'c':
							X.append(t2)
						elif type == 'ch':
							t3.extend(t2)
							X.append(t3)
						elif type == 'o':
							X.append(t1)
						elif type == 'oc':
							t1.extend(t2)
							X.append(t1)
						elif type == 'att':
							if use_attention and AGEs:
								weights = att_history[:, index, :]
								f_o = en_outputs[index, :, :]
								att_op = np.matmul(weights, f_o)
								if manner == 'sc' or Frozen == '1':
									att_op = np.reshape(att_op, [-1]).tolist()
									if manner == 'sc':
										X.append(att_op)
									else:
										att_batch.append(att_op)
									# else:
									# 	att_batch = np.array(att_batch)
									# 	c_vec = sess.run([contrast_v],
							        #                     {C_input: att_batch,
							        #                      lr: learning_rate
							        #                      })
									# 	X.append(c_vec)
								else:
									X.extend(att_op.tolist())
							# for index in range(en_outputs.shape[0]):
							# 	t1 = np.reshape(en_outputs[index], [-1]).tolist()
							# 	if use_attention and AGEs:
							# 		weights = att_history[:, index, :]
							# 		f_o = en_outputs[index, :, :]
							# 		att_op = np.matmul(weights, f_o)
							# 		# if manner == 'sc':
							# 		att_op = np.reshape(att_op, [-1]).tolist()
							# 		att_batch.append(att_op)
							# att_batch = np.array(att_batch)
							# _, c_loss, c_vec = sess.run([C_train_op, C_loss, contrast_v],
							#                  {C_input: att_batch,
							#                   lr: learning_rate
							#                  })
							else:
								f_o = en_outputs[index, :, :]
								if Bi_LSTM:
									f_o_bw = encoder_outputs_bw[index, :, :]
									f_o = f_o + f_o_bw
								if manner == 'sc':
									f_o = np.reshape(f_o, [-1]).tolist()
									X.append(f_o)
								else:
									X.extend(f_o.tolist())
						elif type == 'attc':
							weights = att_history[:, index, :]
							f_o = en_outputs[index, :, :]
							att_op = np.matmul(weights, f_o)
							att_op = np.reshape(att_op, [-1]).tolist()
							att_op.extend(t2)
							X.append(att_op)
					if Frozen == '1':
						att_batch = np.array(att_batch)
						[c_vec] = sess.run([contrast_v],
					                    {C_input: att_batch,
					                     lr: learning_rate
					                     })
						X.extend(c_vec)
					pred = pred.tolist()
					for index, i in enumerate(pred):
						pred[index].reverse()
					pred = np.array(pred)
					# en_outputs, en_c, en_h, en_c_1, en_h_1, pred_2 = sess.run(
					# 	[encoder_output, encoder_c, encoder_h, encoder_c_1, encoder_h_1, predictions],
					# 	{input_data: pred,
					# 	 targets: this_targets,
					# 	 lr: learning_rate,
					# 	 target_sequence_length: [time_steps] * batch_size,
					# 	 source_sequence_length: [time_steps] * batch_size,
					# 	 keep_prob: 0.5})
					# for index in range(en_outputs.shape[0]):
					# 	t1 = np.reshape(en_outputs[index], [-1]).tolist()
					# 	t2 = np.reshape(en_c[index], [-1]).tolist()
					# 	t3 = np.reshape(en_h[index], [-1]).tolist()
					# 	t4 = np.reshape(en_c_1[index], [-1]).tolist()
					# 	t5 = np.reshape(en_h_1[index], [-1]).tolist()
					# 	if type == 'c':
					# 		X_pred.append(t2)
					# 	elif type == 'ch':
					# 		t3.extend(t2)
					# 		X_pred.append(t3)
					# 	elif type == 'o':
					# 		X_pred.append(t1)
					# 	elif type == 'oc':
					# 		t1.extend(t2)
					# 		X_pred.append(t1)
					# 	elif type == 'att':
					# 		if use_attention and AGEs:
					# 			weights = att_history[:, index, :]
					# 			f_o = en_outputs[index, :, :]
					# 			att_op = np.matmul(weights, f_o)
					# 			if manner == 'sc':
					# 				att_op = np.reshape(att_op, [-1]).tolist()
					# 				X_pred.append(att_op)
					# 			else:
					# 				X_pred.extend(att_op.tolist())
					# 		else:
					# 			f_o = en_outputs[index, :, :]
					# 			# if Bi_LSTM:
					# 			# 	f_o_bw = encoder_outputs_bw[index, :, :]
					# 			# 	f_o = f_o + f_o_bw
					# 			if manner == 'sc':
					# 				f_o = np.reshape(f_o, [-1]).tolist()
					# 				X_pred.append(f_o)
					# 			else:
					# 				X_pred.extend(f_o.tolist())
					# 	elif type == 'attc':
					# 		weights = att_history[:, index, :]
					# 		f_o = en_outputs[index, :, :]
					# 		att_op = np.matmul(weights, f_o)
					# 		att_op = np.reshape(att_op, [-1]).tolist()
					# 		att_op.extend(t2)
					# 		X_pred.append(att_op)
					if manner == 'sc' or Frozen == '1' or C_reid == '1':
						y.extend([k] * batch_size)
						# C_y.extend([k] * batch_size)
					else:
						y.extend([k] * batch_size * time_steps)
			# exit(1)
			for k, v in t_ids_:
				flag = 0
				if len(v) == 0:
					continue
				if len(v) < batch_size:
					flag = batch_size - len(v)
					v.extend([v[0]] * (batch_size - len(v)))
				# print('%s - %d' % (k, len(v)))
				for batch_i in range(len(v) // batch_size):
					this_input = t_input_data[v[batch_i * batch_size : (batch_i + 1) * batch_size]]
					this_targets = t_targets[v[batch_i * batch_size : (batch_i + 1) * batch_size]]
					if use_attention:
						if Frozen == '1':
							en_outputs, en_c, en_h, en_c_1, en_h_1, pred, att_state, att_history, att, align = sess.run([encoder_output, encoder_c,
							encoder_h, encoder_c_1, encoder_h_1, predictions, attention_state, alignment_history, attention_weights, alignment],
							                      {input_data: this_input,
							                       targets: this_targets,
							                       lr: learning_rate,
							                       target_sequence_length: [time_steps] * batch_size,
							                       source_sequence_length: [time_steps] * batch_size,
							                       keep_prob: 1.0})
						else:
							en_outputs, c_vec, en_c, en_h, en_c_1, en_h_1, pred, att_state, att_history, att, align = sess.run(
								[encoder_output, contrast_v, encoder_c,
								 encoder_h, encoder_c_1, encoder_h_1, predictions, attention_state, alignment_history,
								 attention_weights, alignment],
								{input_data: this_input,
								 targets: this_targets,
								 lr: learning_rate,
								 C_lr: learning_rate,
								 target_sequence_length: [time_steps] * batch_size,
								 source_sequence_length: [time_steps] * batch_size,
								 keep_prob: 1.0})
							if C_reid == '1':
								t_C_X.extend(c_vec)
					else:
						if Bi_LSTM:
							en_outputs, encoder_outputs_bw, en_c, en_h, en_c_1, en_h_1, pred = sess.run(
								[encoder_output, encoder_output_bw, encoder_c,
								 encoder_h, encoder_c_1, encoder_h_1, predictions],
								{input_data: this_input,
								 targets: this_targets,
								 lr: learning_rate,
								 target_sequence_length: [time_steps] * batch_size,
								 source_sequence_length: [time_steps] * batch_size,
								 keep_prob: 1.0})
						else:
							en_outputs, en_c, en_h, en_c_1, en_h_1, pred = sess.run(
								[encoder_output, encoder_c,
								 encoder_h, encoder_c_1, encoder_h_1, predictions],
								{input_data: this_input,
								 targets: this_targets,
								 lr: learning_rate,
								 target_sequence_length: [time_steps] * batch_size,
								 source_sequence_length: [time_steps] * batch_size,
								 keep_prob: 1.0})
					if flag > 0:
						en_outputs = en_outputs[:-flag]
					att_batch = []
					for index in range(en_outputs.shape[0]):
						t1 = np.reshape(en_outputs[index], [-1]).tolist()
						t2 = np.reshape(en_c[index], [-1]).tolist()
						t3 = np.reshape(en_h[index], [-1]).tolist()
						t4 = np.reshape(en_c_1[index], [-1]).tolist()
						t5 = np.reshape(en_h_1[index], [-1]).tolist()
						if type == 'att':
							if use_attention and AGEs:
								weights = att_history[:, index, :]
								f_o = en_outputs[index, :, :]
								att_op = np.matmul(weights, f_o)
								# if manner == 'sc':
								# 	att_op = np.reshape(att_op, [-1]).tolist()
								# 	t_X.append(att_op)
								if manner == 'sc' or Frozen == '1':
									att_op = np.reshape(att_op, [-1]).tolist()
									if manner == 'sc':
										t_X.append(att_op)
									else:
										att_batch.append(att_op)
										# att_batch = np.array(att_batch)
										# c_vec = sess.run([contrast_v],
										#                  {C_input: att_batch,
										#                   lr: learning_rate
										#                   })
										# t_X.append(c_vec)
								else:
									t_X.extend(att_op.tolist())
								t_X_att.append(weights)
							else:
								f_o = en_outputs[index, :, :]
								if Bi_LSTM:
									f_o_bw = encoder_outputs_bw[index, :, :]
									f_o = f_o_bw + f_o
								if manner == 'sc':
									f_o = np.reshape(f_o, [-1]).tolist()
									t_X.append(f_o)
								else:
									t_X.extend(f_o.tolist())
						elif type == 'attc':
							weights = att_history[:, index, :]
							f_o = en_outputs[index, :, :]
							att_op = np.matmul(weights, f_o)
							att_op = np.reshape(att_op, [-1]).tolist()
							att_op.extend(t2)
							t_X.append(att_op)
					if Frozen == '1':
						att_batch = np.array(att_batch)
						[c_vec] = sess.run([contrast_v],
					                    {C_input: att_batch,
					                     lr: learning_rate
					                     })
						t_X.extend(c_vec)
					pred = pred.tolist()
					for index, i in enumerate(pred):
						pred[index].reverse()
					pred = np.array(pred)
					# en_outputs, en_c, en_h, en_c_1, en_h_1, pred_2 = sess.run(
					# 	[encoder_output, encoder_c, encoder_h, encoder_c_1, encoder_h_1, predictions],
					# 	{input_data: pred,
					# 	 targets: this_targets,
					# 	 lr: learning_rate,
					# 	 target_sequence_length: [time_steps] * batch_size,
					# 	 source_sequence_length: [time_steps] * batch_size,
					# 	 keep_prob: 0.5})
					# for index in range(en_outputs.shape[0]):
					# 	t1 = np.reshape(en_outputs[index], [-1]).tolist()
					# 	t2 = np.reshape(en_c[index], [-1]).tolist()
					# 	t3 = np.reshape(en_h[index], [-1]).tolist()
					# 	t4 = np.reshape(en_c_1[index], [-1]).tolist()
					# 	t5 = np.reshape(en_h_1[index], [-1]).tolist()
					# 	if type == 'att':
					# 		if use_attention and AGEs:
					# 			weights = att_history[:, index, :]
					# 			f_o = en_outputs[index, :, :]
					# 			att_op = np.matmul(weights, f_o)
					# 			if manner == 'sc':
					# 				att_op = np.reshape(att_op, [-1]).tolist()
					# 				t_X_pred.append(att_op)
					# 			else:
					# 				t_X_pred.extend(att_op.tolist())
					# 		else:
					# 			f_o = en_outputs[index, :, :]
					# 			if manner == 'sc':
					# 				f_o = np.reshape(f_o, [-1]).tolist()
					# 				t_X_pred.append(f_o)
					# 			else:
					# 				t_X_pred.extend(f_o.tolist())
					# 	elif type == 'attc':
					# 		weights = att_history[:, index, :]
					# 		f_o = en_outputs[index, :, :]
					# 		att_op = np.matmul(weights, f_o)
					# 		att_op = np.reshape(att_op, [-1]).tolist()
					# 		att_op.extend(t2)
					# 		t_X_pred.append(att_op)
					if manner == 'sc' or Frozen == '1' or C_reid == '1':
						t_y.extend([k] * (batch_size - flag))
						# t_C_y.extend([k] * (batch_size - flag))
					else:
						t_y.extend([k] * (batch_size - flag) * time_steps)
			if dataset == 'IAS':
				for k, v in t_2_ids_:
					flag = 0
					if len(v) == 0:
						continue
					if len(v) < batch_size:
						flag = batch_size - len(v)
						v.extend([v[0]] * (batch_size - len(v)))
					# print('%s - %d' % (k, len(v)))
					for batch_i in range(len(v) // batch_size):
						this_input = t_2_input_data[v[batch_i * batch_size : (batch_i + 1) * batch_size]]
						this_targets = t_2_targets[v[batch_i * batch_size : (batch_i + 1) * batch_size]]
						if use_attention:
							if Frozen == '1':
								en_outputs, en_c, en_h, en_c_1, en_h_1, pred, att_state, att_history, att, align = sess.run(
									[encoder_output, encoder_c,
									 encoder_h, encoder_c_1, encoder_h_1, predictions, attention_state, alignment_history,
									 attention_weights, alignment],
									{input_data: this_input,
									 targets: this_targets,
									 lr: learning_rate,
									 target_sequence_length: [time_steps] * batch_size,
									 source_sequence_length: [time_steps] * batch_size,
									 keep_prob: 1.0})
							else:
								en_outputs, c_vec, en_c, en_h, en_c_1, en_h_1, pred, att_state, att_history, att, align = sess.run(
									[encoder_output, contrast_v, encoder_c,
									 encoder_h, encoder_c_1, encoder_h_1, predictions, attention_state, alignment_history,
									 attention_weights, alignment],
									{input_data: this_input,
									 targets: this_targets,
									 lr: learning_rate,
									 C_lr: learning_rate,
									 target_sequence_length: [time_steps] * batch_size,
									 source_sequence_length: [time_steps] * batch_size,
									 keep_prob: 1.0})
								if C_reid == '1':
									t_2_C_X.extend(c_vec)
						else:
							if Bi_LSTM:
								en_outputs, encoder_outputs_bw, en_c, en_h, en_c_1, en_h_1, pred = sess.run(
									[encoder_output, encoder_output_bw, encoder_c,
									 encoder_h, encoder_c_1, encoder_h_1, predictions],
									{input_data: this_input,
									 targets: this_targets,
									 lr: learning_rate,
									 target_sequence_length: [time_steps] * batch_size,
									 source_sequence_length: [time_steps] * batch_size,
									 keep_prob: 1.0})
							else:
								en_outputs, en_c, en_h, en_c_1, en_h_1, pred = sess.run(
									[encoder_output, encoder_c,
									 encoder_h, encoder_c_1, encoder_h_1, predictions],
									{input_data: this_input,
									 targets: this_targets,
									 lr: learning_rate,
									 target_sequence_length: [time_steps] * batch_size,
									 source_sequence_length: [time_steps] * batch_size,
									 keep_prob: 1.0})
						if flag > 0:
							en_outputs = en_outputs[:-flag]
						att_batch = []
						for index in range(en_outputs.shape[0]):
							t1 = np.reshape(en_outputs[index], [-1]).tolist()
							t2 = np.reshape(en_c[index], [-1]).tolist()
							t3 = np.reshape(en_h[index], [-1]).tolist()
							t4 = np.reshape(en_c_1[index], [-1]).tolist()
							t5 = np.reshape(en_h_1[index], [-1]).tolist()
							if type == 'att':
								if use_attention and AGEs:
									weights = att_history[:, index, :]
									f_o = en_outputs[index, :, :]
									att_op = np.matmul(weights, f_o)
									if manner == 'sc' or Frozen == '1':
										att_op = np.reshape(att_op, [-1]).tolist()
										if manner == 'sc':
											t_2_X.append(att_op)
										else:
											att_batch.append(att_op)
											# att_batch = np.array(att_batch)
											# c_vec = sess.run([contrast_v],
											#                  {C_input: att_batch,
											#                   lr: learning_rate
											#                   })
											# t_2_X.append(c_vec)
									else:
										t_2_X.extend(att_op.tolist())
								else:
									f_o = en_outputs[index, :, :]
									if Bi_LSTM:
										f_o_bw = encoder_outputs_bw[index, :, :]
										f_o = f_o_bw + f_o
									if manner == 'sc':
										f_o = np.reshape(f_o, [-1]).tolist()
										t_2_X.append(f_o)
									else:
										t_2_X.extend(f_o.tolist())
						if Frozen == '1':
							att_batch = np.array(att_batch)
							[c_vec] = sess.run([contrast_v],
							                   {C_input: att_batch,
							                    lr: learning_rate
							                    })
							t_2_X.extend(c_vec)
						if manner == 'sc' or Frozen == '1' or C_reid == '1':
							t_2_y.extend([k] * (batch_size - flag))
							# t_2_C_y.extend([k] * (batch_size - flag))
						else:
							t_2_y.extend([k] * (batch_size - flag) * time_steps)
		X_0 = np.array(X)
		y_0 = np.array(y)
		X_pred_0 = np.array(X_pred)
		t_X_pred_0 = np.array(t_X_pred)
		from sklearn.preprocessing import label_binarize
		ids_keys = sorted(list(ids.keys()))
		t_ids_keys = sorted(list(t_ids.keys()))
		classes = [i for i in ids_keys]
		t_classes = [i for i in t_ids_keys]
		t_y = label_binarize(t_y, classes=t_classes)
		if dataset == 'IAS':
			t_2_ids_keys = sorted(list(t_2_ids.keys()))
			t_2_classes = [i for i in t_2_ids_keys]
			t_2_y = label_binarize(t_2_y, classes=t_2_classes)
			t_2_y_0 = t_2_y
			t_2_X_0 = t_2_X
		t_y = np.array(t_y)
		if C_reid == '1':
			t_C_X = np.array(t_C_X)
		else:
			t_X = np.array(t_X)
		t_y_0 = t_y
		t_X_0 = t_X

		# checkpoint 0
		_targets = np.load(
			'Datasets/' + frames_ps + view_dir + dataset + '_train_npy_data/target_' + dimension + '_' + dataset + '_' + str(
				time_steps) + '.npy')
		_targets = _targets.reshape([-1, time_steps, series_length])
		loaded_graph = tf.Graph()
		with tf.Session(graph=loaded_graph, config=config) as sess:
			loader = tf.train.import_meta_graph(checkpoint + '.meta')
			loader.restore(sess, checkpoint)
			input_data = loaded_graph.get_tensor_by_name('inputs:0')
			targets = loaded_graph.get_tensor_by_name('targets:0')
			if Frozen == '1':
				contrast_v = loaded_graph.get_tensor_by_name('add_16:0')
				C_lr = loaded_graph.get_tensor_by_name('learning_rate_1:0')
				C_input = loaded_graph.get_tensor_by_name('C_inptuiut:0')
			else:
				contrast_v = loaded_graph.get_tensor_by_name("add_2:0")
				C_lr = loaded_graph.get_tensor_by_name('learning_rate_1:0')
			if Bi_LSTM:
				encoder_output = loaded_graph.get_tensor_by_name('bidirectional_rnn/fw/fw/transpose_1:0')
				encoder_c_1 = loaded_graph.get_tensor_by_name('bidirectional_rnn/fw/fw/while/Exit_3:0')
				encoder_h_1 = loaded_graph.get_tensor_by_name('bidirectional_rnn/fw/fw/while/Exit_4:0')
				encoder_c = loaded_graph.get_tensor_by_name('bidirectional_rnn/fw/fw/while/Exit_5:0')
				encoder_h = loaded_graph.get_tensor_by_name('bidirectional_rnn/fw/fw/while/Exit_6:0')
				encoder_output_bw = loaded_graph.get_tensor_by_name('ReverseSequence: 0')
				encoder_c_1_bw = loaded_graph.get_tensor_by_name('bidirectional_rnn/bw/bw/while/Exit_3:0')
				encoder_h_1_bw = loaded_graph.get_tensor_by_name('bidirectional_rnn/bw/bw/while/Exit_4:0')
				encoder_c_bw = loaded_graph.get_tensor_by_name('bidirectional_rnn/bw/bw/while/Exit_5:0')
				encoder_h_bw = loaded_graph.get_tensor_by_name('bidirectional_rnn/bw/bw/while/Exit_6:0')
				predictions = loaded_graph.get_tensor_by_name('predictions:0')
			else:
				encoder_output = loaded_graph.get_tensor_by_name('rnn/transpose_1:0')
				encoder_c_1 = loaded_graph.get_tensor_by_name('rnn/while/Exit_3:0')
				encoder_h_1 = loaded_graph.get_tensor_by_name('rnn/while/Exit_4:0')
				encoder_c = loaded_graph.get_tensor_by_name('rnn/while/Exit_5:0')
				encoder_h = loaded_graph.get_tensor_by_name('rnn/while/Exit_6:0')
				predictions = loaded_graph.get_tensor_by_name('predictions:0')
				# train_output = loaded_graph.get_tensor_by_name('train_output:0')
			if use_attention:
				alignment_history = loaded_graph.get_tensor_by_name('train_attention_matrix:0')
				# train_attention_matrix = loaded_graph.get_tensor_by_name('train_attention_matrix:0')
				attention_state = loaded_graph.get_tensor_by_name('decode/decoder/while/Exit_12:0')
				attention_weights = loaded_graph.get_tensor_by_name('decode/decoder/while/Exit_8:0')
				alignment = loaded_graph.get_tensor_by_name('decode/decoder/while/Exit_10:0')
			lr = loaded_graph.get_tensor_by_name('learning_rate:0')
			keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
			source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
			target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
			X = []
			C_X = []
			X_all_op = []
			X_final_op = []
			X_final_c = []
			X_final_h = []
			X_final_c1 = []
			X_final_h1 = []
			X_final_ch = []
			X_final_ch1 = []
			y = []
			C_y = []
			X_pred = []
			t_X = []
			t_C_X = []
			t_y = []
			t_C_y = []
			t_2_X = []
			t_2_C_X = []
			t_2_y = []
			t_2_C_y = []
			t_X_pred = []
			t_X_att = []
			# print(t_ids)
			# print(test_attention)
			ids_ = sorted(ids.items(), key=lambda item:item[0])
			t_ids_ = sorted(t_ids.items(), key=lambda item: item[0])
			if dataset == 'IAS':
				t_2_ids_ = sorted(t_2_ids.items(), key=lambda item: item[0])
			# print(ids_)
			# exit(1)
			for k, v in ids_:
				if len(v) == 0:
					print(k)
					continue
				if len(v) < batch_size:
					v.extend([v[0]] * (batch_size - len(v)))
				# print('%s - %d' % (k, len(v)))
				for batch_i in range(len(v) // batch_size):
					this_input = _input_data[v[batch_i * batch_size : (batch_i + 1) * batch_size]]
					this_targets = _targets[v[batch_i * batch_size : (batch_i + 1) * batch_size]]
					if use_attention:
						if Frozen == '1':
							en_outputs, en_c, en_h, en_c_1, en_h_1, pred, att_state, att_history, att, align = sess.run([encoder_output, encoder_c,
							encoder_h, encoder_c_1, encoder_h_1, predictions, attention_state, alignment_history, attention_weights, alignment],
							                      {input_data: this_input,
							                       targets: this_targets,
							                       lr: learning_rate,
							                       target_sequence_length: [time_steps] * batch_size,
							                       source_sequence_length: [time_steps] * batch_size,
							                       keep_prob: 1.0})
						else:
							en_outputs, c_vec, en_c, en_h, en_c_1, en_h_1, pred, att_state, att_history, att, align = sess.run(
								[encoder_output, contrast_v, encoder_c,
								 encoder_h, encoder_c_1, encoder_h_1, predictions, attention_state, alignment_history,
								 attention_weights, alignment],
								{input_data: this_input,
								 targets: this_targets,
								 lr: learning_rate,
								 C_lr: learning_rate,
								 target_sequence_length: [time_steps] * batch_size,
								 source_sequence_length: [time_steps] * batch_size,
								 keep_prob: 1.0})
							if C_reid == '1':
								C_X.extend(c_vec)
					else:
						if Bi_LSTM:
							en_outputs, encoder_outputs_bw, en_c, en_h, en_c_1, en_h_1, pred = sess.run(
								[encoder_output, encoder_output_bw, encoder_c,
								 encoder_h, encoder_c_1, encoder_h_1, predictions],
								{input_data: this_input,
								 targets: this_targets,
								 lr: learning_rate,
								 target_sequence_length: [time_steps] * batch_size,
								 source_sequence_length: [time_steps] * batch_size,
								 keep_prob: 1.0})
						else:
							en_outputs, en_c, en_h, en_c_1, en_h_1, pred = sess.run(
								[encoder_output, encoder_c,
								 encoder_h, encoder_c_1, encoder_h_1, predictions],
								{input_data: this_input,
								 targets: this_targets,
								 lr: learning_rate,
								 target_sequence_length: [time_steps] * batch_size,
								 source_sequence_length: [time_steps] * batch_size,
								 keep_prob: 1.0})
					att_batch = []
					for index in range(en_outputs.shape[0]):
						t1 = np.reshape(en_outputs[index], [-1]).tolist()
						t2 = np.reshape(en_c[index], [-1]).tolist()
						t3 = np.reshape(en_h[index], [-1]).tolist()
						t4 = np.reshape(en_c_1[index], [-1]).tolist()
						t5 = np.reshape(en_h_1[index], [-1]).tolist()
						if type == 'c':
							X.append(t2)
						elif type == 'ch':
							t3.extend(t2)
							X.append(t3)
						elif type == 'o':
							X.append(t1)
						elif type == 'oc':
							t1.extend(t2)
							X.append(t1)
						elif type == 'att':
							if use_attention and AGEs:
								weights = att_history[:, index, :]
								f_o = en_outputs[index, :, :]
								att_op = np.matmul(weights, f_o)
								if manner == 'sc' or Frozen == '1':
									att_op = np.reshape(att_op, [-1]).tolist()
									if manner == 'sc':
										X.append(att_op)
									else:
										att_batch.append(att_op)
									# else:
									# 	att_batch = np.array(att_batch)
									# 	c_vec = sess.run([contrast_v],
							        #                     {C_input: att_batch,
							        #                      lr: learning_rate
							        #                      })
									# 	X.append(c_vec)
								else:
									X.extend(att_op.tolist())
							# for index in range(en_outputs.shape[0]):
							# 	t1 = np.reshape(en_outputs[index], [-1]).tolist()
							# 	if use_attention and AGEs:
							# 		weights = att_history[:, index, :]
							# 		f_o = en_outputs[index, :, :]
							# 		att_op = np.matmul(weights, f_o)
							# 		# if manner == 'sc':
							# 		att_op = np.reshape(att_op, [-1]).tolist()
							# 		att_batch.append(att_op)
							# att_batch = np.array(att_batch)
							# _, c_loss, c_vec = sess.run([C_train_op, C_loss, contrast_v],
							#                  {C_input: att_batch,
							#                   lr: learning_rate
							#                  })
							else:
								f_o = en_outputs[index, :, :]
								if Bi_LSTM:
									f_o_bw = encoder_outputs_bw[index, :, :]
									f_o = f_o + f_o_bw
								if manner == 'sc':
									f_o = np.reshape(f_o, [-1]).tolist()
									X.append(f_o)
								else:
									X.extend(f_o.tolist())
						elif type == 'attc':
							weights = att_history[:, index, :]
							f_o = en_outputs[index, :, :]
							att_op = np.matmul(weights, f_o)
							att_op = np.reshape(att_op, [-1]).tolist()
							att_op.extend(t2)
							X.append(att_op)
					if Frozen == '1':
						att_batch = np.array(att_batch)
						[c_vec] = sess.run([contrast_v],
					                    {C_input: att_batch,
					                     lr: learning_rate
					                     })
						X.extend(c_vec)
					pred = pred.tolist()
					for index, i in enumerate(pred):
						pred[index].reverse()
					pred = np.array(pred)
					# en_outputs, en_c, en_h, en_c_1, en_h_1, pred_2 = sess.run(
					# 	[encoder_output, encoder_c, encoder_h, encoder_c_1, encoder_h_1, predictions],
					# 	{input_data: pred,
					# 	 targets: this_targets,
					# 	 lr: learning_rate,
					# 	 target_sequence_length: [time_steps] * batch_size,
					# 	 source_sequence_length: [time_steps] * batch_size,
					# 	 keep_prob: 0.5})
					# for index in range(en_outputs.shape[0]):
					# 	t1 = np.reshape(en_outputs[index], [-1]).tolist()
					# 	t2 = np.reshape(en_c[index], [-1]).tolist()
					# 	t3 = np.reshape(en_h[index], [-1]).tolist()
					# 	t4 = np.reshape(en_c_1[index], [-1]).tolist()
					# 	t5 = np.reshape(en_h_1[index], [-1]).tolist()
					# 	if type == 'c':
					# 		X_pred.append(t2)
					# 	elif type == 'ch':
					# 		t3.extend(t2)
					# 		X_pred.append(t3)
					# 	elif type == 'o':
					# 		X_pred.append(t1)
					# 	elif type == 'oc':
					# 		t1.extend(t2)
					# 		X_pred.append(t1)
					# 	elif type == 'att':
					# 		if use_attention and AGEs:
					# 			weights = att_history[:, index, :]
					# 			f_o = en_outputs[index, :, :]
					# 			att_op = np.matmul(weights, f_o)
					# 			if manner == 'sc':
					# 				att_op = np.reshape(att_op, [-1]).tolist()
					# 				X_pred.append(att_op)
					# 			else:
					# 				X_pred.extend(att_op.tolist())
					# 		else:
					# 			f_o = en_outputs[index, :, :]
					# 			# if Bi_LSTM:
					# 			# 	f_o_bw = encoder_outputs_bw[index, :, :]
					# 			# 	f_o = f_o + f_o_bw
					# 			if manner == 'sc':
					# 				f_o = np.reshape(f_o, [-1]).tolist()
					# 				X_pred.append(f_o)
					# 			else:
					# 				X_pred.extend(f_o.tolist())
					# 	elif type == 'attc':
					# 		weights = att_history[:, index, :]
					# 		f_o = en_outputs[index, :, :]
					# 		att_op = np.matmul(weights, f_o)
					# 		att_op = np.reshape(att_op, [-1]).tolist()
					# 		att_op.extend(t2)
					# 		X_pred.append(att_op)
					if manner == 'sc' or Frozen == '1' or C_reid == '1':
						y.extend([k] * batch_size)
						# C_y.extend([k] * batch_size)
					else:
						y.extend([k] * batch_size * time_steps)
			# exit(1)
			for k, v in t_ids_:
				flag = 0
				if len(v) == 0:
					continue
				if len(v) < batch_size:
					flag = batch_size - len(v)
					v.extend([v[0]] * (batch_size - len(v)))
				# print('%s - %d' % (k, len(v)))
				for batch_i in range(len(v) // batch_size):
					this_input = t_input_data[v[batch_i * batch_size : (batch_i + 1) * batch_size]]
					this_targets = t_targets[v[batch_i * batch_size : (batch_i + 1) * batch_size]]
					if use_attention:
						if Frozen == '1':
							en_outputs, en_c, en_h, en_c_1, en_h_1, pred, att_state, att_history, att, align = sess.run([encoder_output, encoder_c,
							encoder_h, encoder_c_1, encoder_h_1, predictions, attention_state, alignment_history, attention_weights, alignment],
							                      {input_data: this_input,
							                       targets: this_targets,
							                       lr: learning_rate,
							                       target_sequence_length: [time_steps] * batch_size,
							                       source_sequence_length: [time_steps] * batch_size,
							                       keep_prob: 1.0})
						else:
							en_outputs, c_vec, en_c, en_h, en_c_1, en_h_1, pred, att_state, att_history, att, align = sess.run(
								[encoder_output, contrast_v, encoder_c,
								 encoder_h, encoder_c_1, encoder_h_1, predictions, attention_state, alignment_history,
								 attention_weights, alignment],
								{input_data: this_input,
								 targets: this_targets,
								 lr: learning_rate,
								 C_lr: learning_rate,
								 target_sequence_length: [time_steps] * batch_size,
								 source_sequence_length: [time_steps] * batch_size,
								 keep_prob: 1.0})
							if C_reid == '1':
								t_C_X.extend(c_vec)
					else:
						if Bi_LSTM:
							en_outputs, encoder_outputs_bw, en_c, en_h, en_c_1, en_h_1, pred = sess.run(
								[encoder_output, encoder_output_bw, encoder_c,
								 encoder_h, encoder_c_1, encoder_h_1, predictions],
								{input_data: this_input,
								 targets: this_targets,
								 lr: learning_rate,
								 target_sequence_length: [time_steps] * batch_size,
								 source_sequence_length: [time_steps] * batch_size,
								 keep_prob: 1.0})
						else:
							en_outputs, en_c, en_h, en_c_1, en_h_1, pred = sess.run(
								[encoder_output, encoder_c,
								 encoder_h, encoder_c_1, encoder_h_1, predictions],
								{input_data: this_input,
								 targets: this_targets,
								 lr: learning_rate,
								 target_sequence_length: [time_steps] * batch_size,
								 source_sequence_length: [time_steps] * batch_size,
								 keep_prob: 1.0})
					if flag > 0:
						en_outputs = en_outputs[:-flag]
					att_batch = []
					for index in range(en_outputs.shape[0]):
						t1 = np.reshape(en_outputs[index], [-1]).tolist()
						t2 = np.reshape(en_c[index], [-1]).tolist()
						t3 = np.reshape(en_h[index], [-1]).tolist()
						t4 = np.reshape(en_c_1[index], [-1]).tolist()
						t5 = np.reshape(en_h_1[index], [-1]).tolist()
						if type == 'att':
							if use_attention and AGEs:
								weights = att_history[:, index, :]
								f_o = en_outputs[index, :, :]
								att_op = np.matmul(weights, f_o)
								# if manner == 'sc':
								# 	att_op = np.reshape(att_op, [-1]).tolist()
								# 	t_X.append(att_op)
								if manner == 'sc' or Frozen == '1':
									att_op = np.reshape(att_op, [-1]).tolist()
									if manner == 'sc':
										t_X.append(att_op)
									else:
										att_batch.append(att_op)
										# att_batch = np.array(att_batch)
										# c_vec = sess.run([contrast_v],
										#                  {C_input: att_batch,
										#                   lr: learning_rate
										#                   })
										# t_X.append(c_vec)
								else:
									t_X.extend(att_op.tolist())
								t_X_att.append(weights)
							else:
								f_o = en_outputs[index, :, :]
								if Bi_LSTM:
									f_o_bw = encoder_outputs_bw[index, :, :]
									f_o = f_o_bw + f_o
								if manner == 'sc':
									f_o = np.reshape(f_o, [-1]).tolist()
									t_X.append(f_o)
								else:
									t_X.extend(f_o.tolist())
						elif type == 'attc':
							weights = att_history[:, index, :]
							f_o = en_outputs[index, :, :]
							att_op = np.matmul(weights, f_o)
							att_op = np.reshape(att_op, [-1]).tolist()
							att_op.extend(t2)
							t_X.append(att_op)
					if Frozen == '1':
						att_batch = np.array(att_batch)
						[c_vec] = sess.run([contrast_v],
					                    {C_input: att_batch,
					                     lr: learning_rate
					                     })
						t_X.extend(c_vec)
					pred = pred.tolist()
					for index, i in enumerate(pred):
						pred[index].reverse()
					pred = np.array(pred)
					# en_outputs, en_c, en_h, en_c_1, en_h_1, pred_2 = sess.run(
					# 	[encoder_output, encoder_c, encoder_h, encoder_c_1, encoder_h_1, predictions],
					# 	{input_data: pred,
					# 	 targets: this_targets,
					# 	 lr: learning_rate,
					# 	 target_sequence_length: [time_steps] * batch_size,
					# 	 source_sequence_length: [time_steps] * batch_size,
					# 	 keep_prob: 0.5})
					# for index in range(en_outputs.shape[0]):
					# 	t1 = np.reshape(en_outputs[index], [-1]).tolist()
					# 	t2 = np.reshape(en_c[index], [-1]).tolist()
					# 	t3 = np.reshape(en_h[index], [-1]).tolist()
					# 	t4 = np.reshape(en_c_1[index], [-1]).tolist()
					# 	t5 = np.reshape(en_h_1[index], [-1]).tolist()
					# 	if type == 'att':
					# 		if use_attention and AGEs:
					# 			weights = att_history[:, index, :]
					# 			f_o = en_outputs[index, :, :]
					# 			att_op = np.matmul(weights, f_o)
					# 			if manner == 'sc':
					# 				att_op = np.reshape(att_op, [-1]).tolist()
					# 				t_X_pred.append(att_op)
					# 			else:
					# 				t_X_pred.extend(att_op.tolist())
					# 		else:
					# 			f_o = en_outputs[index, :, :]
					# 			if manner == 'sc':
					# 				f_o = np.reshape(f_o, [-1]).tolist()
					# 				t_X_pred.append(f_o)
					# 			else:
					# 				t_X_pred.extend(f_o.tolist())
					# 	elif type == 'attc':
					# 		weights = att_history[:, index, :]
					# 		f_o = en_outputs[index, :, :]
					# 		att_op = np.matmul(weights, f_o)
					# 		att_op = np.reshape(att_op, [-1]).tolist()
					# 		att_op.extend(t2)
					# 		t_X_pred.append(att_op)
					if manner == 'sc' or Frozen == '1' or C_reid == '1':
						t_y.extend([k] * (batch_size - flag))
						# t_C_y.extend([k] * (batch_size - flag))
					else:
						t_y.extend([k] * (batch_size - flag) * time_steps)
			if dataset == 'IAS':
				for k, v in t_2_ids_:
					flag = 0
					if len(v) == 0:
						continue
					if len(v) < batch_size:
						flag = batch_size - len(v)
						v.extend([v[0]] * (batch_size - len(v)))
					# print('%s - %d' % (k, len(v)))
					for batch_i in range(len(v) // batch_size):
						this_input = t_2_input_data[v[batch_i * batch_size : (batch_i + 1) * batch_size]]
						this_targets = t_2_targets[v[batch_i * batch_size : (batch_i + 1) * batch_size]]
						if use_attention:
							if Frozen == '1':
								en_outputs, en_c, en_h, en_c_1, en_h_1, pred, att_state, att_history, att, align = sess.run(
									[encoder_output, encoder_c,
									 encoder_h, encoder_c_1, encoder_h_1, predictions, attention_state, alignment_history,
									 attention_weights, alignment],
									{input_data: this_input,
									 targets: this_targets,
									 lr: learning_rate,
									 target_sequence_length: [time_steps] * batch_size,
									 source_sequence_length: [time_steps] * batch_size,
									 keep_prob: 1.0})
							else:
								en_outputs, c_vec, en_c, en_h, en_c_1, en_h_1, pred, att_state, att_history, att, align = sess.run(
									[encoder_output, contrast_v, encoder_c,
									 encoder_h, encoder_c_1, encoder_h_1, predictions, attention_state, alignment_history,
									 attention_weights, alignment],
									{input_data: this_input,
									 targets: this_targets,
									 lr: learning_rate,
									 C_lr: learning_rate,
									 target_sequence_length: [time_steps] * batch_size,
									 source_sequence_length: [time_steps] * batch_size,
									 keep_prob: 1.0})
								if C_reid == '1':
									t_2_C_X.extend(c_vec)
						else:
							if Bi_LSTM:
								en_outputs, encoder_outputs_bw, en_c, en_h, en_c_1, en_h_1, pred = sess.run(
									[encoder_output, encoder_output_bw, encoder_c,
									 encoder_h, encoder_c_1, encoder_h_1, predictions],
									{input_data: this_input,
									 targets: this_targets,
									 lr: learning_rate,
									 target_sequence_length: [time_steps] * batch_size,
									 source_sequence_length: [time_steps] * batch_size,
									 keep_prob: 1.0})
							else:
								en_outputs, en_c, en_h, en_c_1, en_h_1, pred = sess.run(
									[encoder_output, encoder_c,
									 encoder_h, encoder_c_1, encoder_h_1, predictions],
									{input_data: this_input,
									 targets: this_targets,
									 lr: learning_rate,
									 target_sequence_length: [time_steps] * batch_size,
									 source_sequence_length: [time_steps] * batch_size,
									 keep_prob: 1.0})
						if flag > 0:
							en_outputs = en_outputs[:-flag]
						att_batch = []
						for index in range(en_outputs.shape[0]):
							t1 = np.reshape(en_outputs[index], [-1]).tolist()
							t2 = np.reshape(en_c[index], [-1]).tolist()
							t3 = np.reshape(en_h[index], [-1]).tolist()
							t4 = np.reshape(en_c_1[index], [-1]).tolist()
							t5 = np.reshape(en_h_1[index], [-1]).tolist()
							if type == 'att':
								if use_attention and AGEs:
									weights = att_history[:, index, :]
									f_o = en_outputs[index, :, :]
									att_op = np.matmul(weights, f_o)
									if manner == 'sc' or Frozen == '1':
										att_op = np.reshape(att_op, [-1]).tolist()
										if manner == 'sc':
											t_2_X.append(att_op)
										else:
											att_batch.append(att_op)
											# att_batch = np.array(att_batch)
											# c_vec = sess.run([contrast_v],
											#                  {C_input: att_batch,
											#                   lr: learning_rate
											#                   })
											# t_2_X.append(c_vec)
									else:
										t_2_X.extend(att_op.tolist())
								else:
									f_o = en_outputs[index, :, :]
									if Bi_LSTM:
										f_o_bw = encoder_outputs_bw[index, :, :]
										f_o = f_o_bw + f_o
									if manner == 'sc':
										f_o = np.reshape(f_o, [-1]).tolist()
										t_2_X.append(f_o)
									else:
										t_2_X.extend(f_o.tolist())
						if Frozen == '1':
							att_batch = np.array(att_batch)
							[c_vec] = sess.run([contrast_v],
							                   {C_input: att_batch,
							                    lr: learning_rate
							                    })
							t_2_X.extend(c_vec)
						if manner == 'sc' or Frozen == '1' or C_reid == '1':
							t_2_y.extend([k] * (batch_size - flag))
							# t_2_C_y.extend([k] * (batch_size - flag))
						else:
							t_2_y.extend([k] * (batch_size - flag) * time_steps)
		X_0 = np.array(X)
		y_0 = np.array(y)
		X_pred_0 = np.array(X_pred)
		t_X_pred_0 = np.array(t_X_pred)
		from sklearn.preprocessing import label_binarize
		ids_keys = sorted(list(ids.keys()))
		t_ids_keys = sorted(list(t_ids.keys()))
		classes = [i for i in ids_keys]
		t_classes = [i for i in t_ids_keys]
		t_y = label_binarize(t_y, classes=t_classes)
		if dataset == 'IAS':
			t_2_ids_keys = sorted(list(t_2_ids.keys()))
			t_2_classes = [i for i in t_2_ids_keys]
			t_2_y = label_binarize(t_2_y, classes=t_2_classes)
			t_2_y_0 = t_2_y
			t_2_X_0 = t_2_X
		t_y = np.array(t_y)
		if C_reid == '1':
			t_C_X = np.array(t_C_X)
		else:
			t_X = np.array(t_X)
		t_y_0 = t_y
		t_X_0 = t_X

		# checkpoint 1
		_targets = np.concatenate((_input_data[1:, :, :], _input_data[-1, :, :].reshape([1, time_steps, series_length])),
		                          axis=0)
		loaded_graph = tf.Graph()
		with tf.Session(graph=loaded_graph, config=config) as sess:
			loader = tf.train.import_meta_graph(checkpoint_1 + '.meta')
			loader.restore(sess, checkpoint)
			input_data = loaded_graph.get_tensor_by_name('inputs:0')
			targets = loaded_graph.get_tensor_by_name('targets:0')
			if Frozen == '1':
				contrast_v = loaded_graph.get_tensor_by_name('add_16:0')
				C_lr = loaded_graph.get_tensor_by_name('learning_rate_1:0')
				C_input = loaded_graph.get_tensor_by_name('C_inptuiut:0')
			else:
				contrast_v = loaded_graph.get_tensor_by_name("add_2:0")
				C_lr = loaded_graph.get_tensor_by_name('learning_rate_1:0')
			if Bi_LSTM:
				encoder_output = loaded_graph.get_tensor_by_name('bidirectional_rnn/fw/fw/transpose_1:0')
				encoder_c_1 = loaded_graph.get_tensor_by_name('bidirectional_rnn/fw/fw/while/Exit_3:0')
				encoder_h_1 = loaded_graph.get_tensor_by_name('bidirectional_rnn/fw/fw/while/Exit_4:0')
				encoder_c = loaded_graph.get_tensor_by_name('bidirectional_rnn/fw/fw/while/Exit_5:0')
				encoder_h = loaded_graph.get_tensor_by_name('bidirectional_rnn/fw/fw/while/Exit_6:0')
				encoder_output_bw = loaded_graph.get_tensor_by_name('ReverseSequence: 0')
				encoder_c_1_bw = loaded_graph.get_tensor_by_name('bidirectional_rnn/bw/bw/while/Exit_3:0')
				encoder_h_1_bw = loaded_graph.get_tensor_by_name('bidirectional_rnn/bw/bw/while/Exit_4:0')
				encoder_c_bw = loaded_graph.get_tensor_by_name('bidirectional_rnn/bw/bw/while/Exit_5:0')
				encoder_h_bw = loaded_graph.get_tensor_by_name('bidirectional_rnn/bw/bw/while/Exit_6:0')
				predictions = loaded_graph.get_tensor_by_name('predictions:0')
			else:
				encoder_output = loaded_graph.get_tensor_by_name('rnn/transpose_1:0')
				encoder_c_1 = loaded_graph.get_tensor_by_name('rnn/while/Exit_3:0')
				encoder_h_1 = loaded_graph.get_tensor_by_name('rnn/while/Exit_4:0')
				encoder_c = loaded_graph.get_tensor_by_name('rnn/while/Exit_5:0')
				encoder_h = loaded_graph.get_tensor_by_name('rnn/while/Exit_6:0')
				predictions = loaded_graph.get_tensor_by_name('predictions:0')
				# train_output = loaded_graph.get_tensor_by_name('train_output:0')
			if use_attention:
				alignment_history = loaded_graph.get_tensor_by_name('train_attention_matrix:0')
				# train_attention_matrix = loaded_graph.get_tensor_by_name('train_attention_matrix:0')
				attention_state = loaded_graph.get_tensor_by_name('decode/decoder/while/Exit_12:0')
				attention_weights = loaded_graph.get_tensor_by_name('decode/decoder/while/Exit_8:0')
				alignment = loaded_graph.get_tensor_by_name('decode/decoder/while/Exit_10:0')
			lr = loaded_graph.get_tensor_by_name('learning_rate:0')
			keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
			source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
			target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
			X = []
			C_X = []
			X_all_op = []
			X_final_op = []
			X_final_c = []
			X_final_h = []
			X_final_c1 = []
			X_final_h1 = []
			X_final_ch = []
			X_final_ch1 = []
			y = []
			C_y = []
			X_pred = []
			t_X = []
			t_C_X = []
			t_y = []
			t_C_y = []
			t_2_X = []
			t_2_C_X = []
			t_2_y = []
			t_2_C_y = []
			t_X_pred = []
			t_X_att = []
			# print(t_ids)
			# print(test_attention)
			ids_ = sorted(ids.items(), key=lambda item:item[0])
			t_ids_ = sorted(t_ids.items(), key=lambda item: item[0])
			if dataset == 'IAS':
				t_2_ids_ = sorted(t_2_ids.items(), key=lambda item: item[0])
			# print(ids_)
			# exit(1)
			for k, v in ids_:
				if len(v) == 0:
					print(k)
					continue
				if len(v) < batch_size:
					v.extend([v[0]] * (batch_size - len(v)))
				# print('%s - %d' % (k, len(v)))
				for batch_i in range(len(v) // batch_size):
					this_input = _input_data[v[batch_i * batch_size : (batch_i + 1) * batch_size]]
					this_targets = _targets[v[batch_i * batch_size : (batch_i + 1) * batch_size]]
					if use_attention:
						if Frozen == '1':
							en_outputs, en_c, en_h, en_c_1, en_h_1, pred, att_state, att_history, att, align = sess.run([encoder_output, encoder_c,
							encoder_h, encoder_c_1, encoder_h_1, predictions, attention_state, alignment_history, attention_weights, alignment],
							                      {input_data: this_input,
							                       targets: this_targets,
							                       lr: learning_rate,
							                       target_sequence_length: [time_steps] * batch_size,
							                       source_sequence_length: [time_steps] * batch_size,
							                       keep_prob: 1.0})
						else:
							en_outputs, c_vec, en_c, en_h, en_c_1, en_h_1, pred, att_state, att_history, att, align = sess.run(
								[encoder_output, contrast_v, encoder_c,
								 encoder_h, encoder_c_1, encoder_h_1, predictions, attention_state, alignment_history,
								 attention_weights, alignment],
								{input_data: this_input,
								 targets: this_targets,
								 lr: learning_rate,
								 C_lr: learning_rate,
								 target_sequence_length: [time_steps] * batch_size,
								 source_sequence_length: [time_steps] * batch_size,
								 keep_prob: 1.0})
							if C_reid == '1':
								C_X.extend(c_vec)
					else:
						if Bi_LSTM:
							en_outputs, encoder_outputs_bw, en_c, en_h, en_c_1, en_h_1, pred = sess.run(
								[encoder_output, encoder_output_bw, encoder_c,
								 encoder_h, encoder_c_1, encoder_h_1, predictions],
								{input_data: this_input,
								 targets: this_targets,
								 lr: learning_rate,
								 target_sequence_length: [time_steps] * batch_size,
								 source_sequence_length: [time_steps] * batch_size,
								 keep_prob: 1.0})
						else:
							en_outputs, en_c, en_h, en_c_1, en_h_1, pred = sess.run(
								[encoder_output, encoder_c,
								 encoder_h, encoder_c_1, encoder_h_1, predictions],
								{input_data: this_input,
								 targets: this_targets,
								 lr: learning_rate,
								 target_sequence_length: [time_steps] * batch_size,
								 source_sequence_length: [time_steps] * batch_size,
								 keep_prob: 1.0})
					att_batch = []
					for index in range(en_outputs.shape[0]):
						t1 = np.reshape(en_outputs[index], [-1]).tolist()
						t2 = np.reshape(en_c[index], [-1]).tolist()
						t3 = np.reshape(en_h[index], [-1]).tolist()
						t4 = np.reshape(en_c_1[index], [-1]).tolist()
						t5 = np.reshape(en_h_1[index], [-1]).tolist()
						if type == 'c':
							X.append(t2)
						elif type == 'ch':
							t3.extend(t2)
							X.append(t3)
						elif type == 'o':
							X.append(t1)
						elif type == 'oc':
							t1.extend(t2)
							X.append(t1)
						elif type == 'att':
							if use_attention and AGEs:
								weights = att_history[:, index, :]
								f_o = en_outputs[index, :, :]
								att_op = np.matmul(weights, f_o)
								if manner == 'sc' or Frozen == '1':
									att_op = np.reshape(att_op, [-1]).tolist()
									if manner == 'sc':
										X.append(att_op)
									else:
										att_batch.append(att_op)
									# else:
									# 	att_batch = np.array(att_batch)
									# 	c_vec = sess.run([contrast_v],
							        #                     {C_input: att_batch,
							        #                      lr: learning_rate
							        #                      })
									# 	X.append(c_vec)
								else:
									X.extend(att_op.tolist())
							# for index in range(en_outputs.shape[0]):
							# 	t1 = np.reshape(en_outputs[index], [-1]).tolist()
							# 	if use_attention and AGEs:
							# 		weights = att_history[:, index, :]
							# 		f_o = en_outputs[index, :, :]
							# 		att_op = np.matmul(weights, f_o)
							# 		# if manner == 'sc':
							# 		att_op = np.reshape(att_op, [-1]).tolist()
							# 		att_batch.append(att_op)
							# att_batch = np.array(att_batch)
							# _, c_loss, c_vec = sess.run([C_train_op, C_loss, contrast_v],
							#                  {C_input: att_batch,
							#                   lr: learning_rate
							#                  })
							else:
								f_o = en_outputs[index, :, :]
								if Bi_LSTM:
									f_o_bw = encoder_outputs_bw[index, :, :]
									f_o = f_o + f_o_bw
								if manner == 'sc':
									f_o = np.reshape(f_o, [-1]).tolist()
									X.append(f_o)
								else:
									X.extend(f_o.tolist())
						elif type == 'attc':
							weights = att_history[:, index, :]
							f_o = en_outputs[index, :, :]
							att_op = np.matmul(weights, f_o)
							att_op = np.reshape(att_op, [-1]).tolist()
							att_op.extend(t2)
							X.append(att_op)
					if Frozen == '1':
						att_batch = np.array(att_batch)
						[c_vec] = sess.run([contrast_v],
					                    {C_input: att_batch,
					                     lr: learning_rate
					                     })
						X.extend(c_vec)
					pred = pred.tolist()
					for index, i in enumerate(pred):
						pred[index].reverse()
					pred = np.array(pred)
					# en_outputs, en_c, en_h, en_c_1, en_h_1, pred_2 = sess.run(
					# 	[encoder_output, encoder_c, encoder_h, encoder_c_1, encoder_h_1, predictions],
					# 	{input_data: pred,
					# 	 targets: this_targets,
					# 	 lr: learning_rate,
					# 	 target_sequence_length: [time_steps] * batch_size,
					# 	 source_sequence_length: [time_steps] * batch_size,
					# 	 keep_prob: 0.5})
					# for index in range(en_outputs.shape[0]):
					# 	t1 = np.reshape(en_outputs[index], [-1]).tolist()
					# 	t2 = np.reshape(en_c[index], [-1]).tolist()
					# 	t3 = np.reshape(en_h[index], [-1]).tolist()
					# 	t4 = np.reshape(en_c_1[index], [-1]).tolist()
					# 	t5 = np.reshape(en_h_1[index], [-1]).tolist()
					# 	if type == 'c':
					# 		X_pred.append(t2)
					# 	elif type == 'ch':
					# 		t3.extend(t2)
					# 		X_pred.append(t3)
					# 	elif type == 'o':
					# 		X_pred.append(t1)
					# 	elif type == 'oc':
					# 		t1.extend(t2)
					# 		X_pred.append(t1)
					# 	elif type == 'att':
					# 		if use_attention and AGEs:
					# 			weights = att_history[:, index, :]
					# 			f_o = en_outputs[index, :, :]
					# 			att_op = np.matmul(weights, f_o)
					# 			if manner == 'sc':
					# 				att_op = np.reshape(att_op, [-1]).tolist()
					# 				X_pred.append(att_op)
					# 			else:
					# 				X_pred.extend(att_op.tolist())
					# 		else:
					# 			f_o = en_outputs[index, :, :]
					# 			# if Bi_LSTM:
					# 			# 	f_o_bw = encoder_outputs_bw[index, :, :]
					# 			# 	f_o = f_o + f_o_bw
					# 			if manner == 'sc':
					# 				f_o = np.reshape(f_o, [-1]).tolist()
					# 				X_pred.append(f_o)
					# 			else:
					# 				X_pred.extend(f_o.tolist())
					# 	elif type == 'attc':
					# 		weights = att_history[:, index, :]
					# 		f_o = en_outputs[index, :, :]
					# 		att_op = np.matmul(weights, f_o)
					# 		att_op = np.reshape(att_op, [-1]).tolist()
					# 		att_op.extend(t2)
					# 		X_pred.append(att_op)
					if manner == 'sc' or Frozen == '1' or C_reid == '1':
						y.extend([k] * batch_size)
						# C_y.extend([k] * batch_size)
					else:
						y.extend([k] * batch_size * time_steps)
			# exit(1)
			for k, v in t_ids_:
				flag = 0
				if len(v) == 0:
					continue
				if len(v) < batch_size:
					flag = batch_size - len(v)
					v.extend([v[0]] * (batch_size - len(v)))
				# print('%s - %d' % (k, len(v)))
				for batch_i in range(len(v) // batch_size):
					this_input = t_input_data[v[batch_i * batch_size : (batch_i + 1) * batch_size]]
					this_targets = t_targets[v[batch_i * batch_size : (batch_i + 1) * batch_size]]
					if use_attention:
						if Frozen == '1':
							en_outputs, en_c, en_h, en_c_1, en_h_1, pred, att_state, att_history, att, align = sess.run([encoder_output, encoder_c,
							encoder_h, encoder_c_1, encoder_h_1, predictions, attention_state, alignment_history, attention_weights, alignment],
							                      {input_data: this_input,
							                       targets: this_targets,
							                       lr: learning_rate,
							                       target_sequence_length: [time_steps] * batch_size,
							                       source_sequence_length: [time_steps] * batch_size,
							                       keep_prob: 1.0})
						else:
							en_outputs, c_vec, en_c, en_h, en_c_1, en_h_1, pred, att_state, att_history, att, align = sess.run(
								[encoder_output, contrast_v, encoder_c,
								 encoder_h, encoder_c_1, encoder_h_1, predictions, attention_state, alignment_history,
								 attention_weights, alignment],
								{input_data: this_input,
								 targets: this_targets,
								 lr: learning_rate,
								 C_lr: learning_rate,
								 target_sequence_length: [time_steps] * batch_size,
								 source_sequence_length: [time_steps] * batch_size,
								 keep_prob: 1.0})
							if C_reid == '1':
								t_C_X.extend(c_vec)
					else:
						if Bi_LSTM:
							en_outputs, encoder_outputs_bw, en_c, en_h, en_c_1, en_h_1, pred = sess.run(
								[encoder_output, encoder_output_bw, encoder_c,
								 encoder_h, encoder_c_1, encoder_h_1, predictions],
								{input_data: this_input,
								 targets: this_targets,
								 lr: learning_rate,
								 target_sequence_length: [time_steps] * batch_size,
								 source_sequence_length: [time_steps] * batch_size,
								 keep_prob: 1.0})
						else:
							en_outputs, en_c, en_h, en_c_1, en_h_1, pred = sess.run(
								[encoder_output, encoder_c,
								 encoder_h, encoder_c_1, encoder_h_1, predictions],
								{input_data: this_input,
								 targets: this_targets,
								 lr: learning_rate,
								 target_sequence_length: [time_steps] * batch_size,
								 source_sequence_length: [time_steps] * batch_size,
								 keep_prob: 1.0})
					if flag > 0:
						en_outputs = en_outputs[:-flag]
					att_batch = []
					for index in range(en_outputs.shape[0]):
						t1 = np.reshape(en_outputs[index], [-1]).tolist()
						t2 = np.reshape(en_c[index], [-1]).tolist()
						t3 = np.reshape(en_h[index], [-1]).tolist()
						t4 = np.reshape(en_c_1[index], [-1]).tolist()
						t5 = np.reshape(en_h_1[index], [-1]).tolist()
						if type == 'att':
							if use_attention and AGEs:
								weights = att_history[:, index, :]
								f_o = en_outputs[index, :, :]
								att_op = np.matmul(weights, f_o)
								# if manner == 'sc':
								# 	att_op = np.reshape(att_op, [-1]).tolist()
								# 	t_X.append(att_op)
								if manner == 'sc' or Frozen == '1':
									att_op = np.reshape(att_op, [-1]).tolist()
									if manner == 'sc':
										t_X.append(att_op)
									else:
										att_batch.append(att_op)
										# att_batch = np.array(att_batch)
										# c_vec = sess.run([contrast_v],
										#                  {C_input: att_batch,
										#                   lr: learning_rate
										#                   })
										# t_X.append(c_vec)
								else:
									t_X.extend(att_op.tolist())
								t_X_att.append(weights)
							else:
								f_o = en_outputs[index, :, :]
								if Bi_LSTM:
									f_o_bw = encoder_outputs_bw[index, :, :]
									f_o = f_o_bw + f_o
								if manner == 'sc':
									f_o = np.reshape(f_o, [-1]).tolist()
									t_X.append(f_o)
								else:
									t_X.extend(f_o.tolist())
						elif type == 'attc':
							weights = att_history[:, index, :]
							f_o = en_outputs[index, :, :]
							att_op = np.matmul(weights, f_o)
							att_op = np.reshape(att_op, [-1]).tolist()
							att_op.extend(t2)
							t_X.append(att_op)
					if Frozen == '1':
						att_batch = np.array(att_batch)
						[c_vec] = sess.run([contrast_v],
					                    {C_input: att_batch,
					                     lr: learning_rate
					                     })
						t_X.extend(c_vec)
					pred = pred.tolist()
					for index, i in enumerate(pred):
						pred[index].reverse()
					pred = np.array(pred)
					# en_outputs, en_c, en_h, en_c_1, en_h_1, pred_2 = sess.run(
					# 	[encoder_output, encoder_c, encoder_h, encoder_c_1, encoder_h_1, predictions],
					# 	{input_data: pred,
					# 	 targets: this_targets,
					# 	 lr: learning_rate,
					# 	 target_sequence_length: [time_steps] * batch_size,
					# 	 source_sequence_length: [time_steps] * batch_size,
					# 	 keep_prob: 0.5})
					# for index in range(en_outputs.shape[0]):
					# 	t1 = np.reshape(en_outputs[index], [-1]).tolist()
					# 	t2 = np.reshape(en_c[index], [-1]).tolist()
					# 	t3 = np.reshape(en_h[index], [-1]).tolist()
					# 	t4 = np.reshape(en_c_1[index], [-1]).tolist()
					# 	t5 = np.reshape(en_h_1[index], [-1]).tolist()
					# 	if type == 'att':
					# 		if use_attention and AGEs:
					# 			weights = att_history[:, index, :]
					# 			f_o = en_outputs[index, :, :]
					# 			att_op = np.matmul(weights, f_o)
					# 			if manner == 'sc':
					# 				att_op = np.reshape(att_op, [-1]).tolist()
					# 				t_X_pred.append(att_op)
					# 			else:
					# 				t_X_pred.extend(att_op.tolist())
					# 		else:
					# 			f_o = en_outputs[index, :, :]
					# 			if manner == 'sc':
					# 				f_o = np.reshape(f_o, [-1]).tolist()
					# 				t_X_pred.append(f_o)
					# 			else:
					# 				t_X_pred.extend(f_o.tolist())
					# 	elif type == 'attc':
					# 		weights = att_history[:, index, :]
					# 		f_o = en_outputs[index, :, :]
					# 		att_op = np.matmul(weights, f_o)
					# 		att_op = np.reshape(att_op, [-1]).tolist()
					# 		att_op.extend(t2)
					# 		t_X_pred.append(att_op)
					if manner == 'sc' or Frozen == '1' or C_reid == '1':
						t_y.extend([k] * (batch_size - flag))
						# t_C_y.extend([k] * (batch_size - flag))
					else:
						t_y.extend([k] * (batch_size - flag) * time_steps)
			if dataset == 'IAS':
				for k, v in t_2_ids_:
					flag = 0
					if len(v) == 0:
						continue
					if len(v) < batch_size:
						flag = batch_size - len(v)
						v.extend([v[0]] * (batch_size - len(v)))
					# print('%s - %d' % (k, len(v)))
					for batch_i in range(len(v) // batch_size):
						this_input = t_2_input_data[v[batch_i * batch_size : (batch_i + 1) * batch_size]]
						this_targets = t_2_targets[v[batch_i * batch_size : (batch_i + 1) * batch_size]]
						if use_attention:
							if Frozen == '1':
								en_outputs, en_c, en_h, en_c_1, en_h_1, pred, att_state, att_history, att, align = sess.run(
									[encoder_output, encoder_c,
									 encoder_h, encoder_c_1, encoder_h_1, predictions, attention_state, alignment_history,
									 attention_weights, alignment],
									{input_data: this_input,
									 targets: this_targets,
									 lr: learning_rate,
									 target_sequence_length: [time_steps] * batch_size,
									 source_sequence_length: [time_steps] * batch_size,
									 keep_prob: 1.0})
							else:
								en_outputs, c_vec, en_c, en_h, en_c_1, en_h_1, pred, att_state, att_history, att, align = sess.run(
									[encoder_output, contrast_v, encoder_c,
									 encoder_h, encoder_c_1, encoder_h_1, predictions, attention_state, alignment_history,
									 attention_weights, alignment],
									{input_data: this_input,
									 targets: this_targets,
									 lr: learning_rate,
									 C_lr: learning_rate,
									 target_sequence_length: [time_steps] * batch_size,
									 source_sequence_length: [time_steps] * batch_size,
									 keep_prob: 1.0})
								if C_reid == '1':
									t_2_C_X.extend(c_vec)
						else:
							if Bi_LSTM:
								en_outputs, encoder_outputs_bw, en_c, en_h, en_c_1, en_h_1, pred = sess.run(
									[encoder_output, encoder_output_bw, encoder_c,
									 encoder_h, encoder_c_1, encoder_h_1, predictions],
									{input_data: this_input,
									 targets: this_targets,
									 lr: learning_rate,
									 target_sequence_length: [time_steps] * batch_size,
									 source_sequence_length: [time_steps] * batch_size,
									 keep_prob: 1.0})
							else:
								en_outputs, en_c, en_h, en_c_1, en_h_1, pred = sess.run(
									[encoder_output, encoder_c,
									 encoder_h, encoder_c_1, encoder_h_1, predictions],
									{input_data: this_input,
									 targets: this_targets,
									 lr: learning_rate,
									 target_sequence_length: [time_steps] * batch_size,
									 source_sequence_length: [time_steps] * batch_size,
									 keep_prob: 1.0})
						if flag > 0:
							en_outputs = en_outputs[:-flag]
						att_batch = []
						for index in range(en_outputs.shape[0]):
							t1 = np.reshape(en_outputs[index], [-1]).tolist()
							t2 = np.reshape(en_c[index], [-1]).tolist()
							t3 = np.reshape(en_h[index], [-1]).tolist()
							t4 = np.reshape(en_c_1[index], [-1]).tolist()
							t5 = np.reshape(en_h_1[index], [-1]).tolist()
							if type == 'att':
								if use_attention and AGEs:
									weights = att_history[:, index, :]
									f_o = en_outputs[index, :, :]
									att_op = np.matmul(weights, f_o)
									if manner == 'sc' or Frozen == '1':
										att_op = np.reshape(att_op, [-1]).tolist()
										if manner == 'sc':
											t_2_X.append(att_op)
										else:
											att_batch.append(att_op)
											# att_batch = np.array(att_batch)
											# c_vec = sess.run([contrast_v],
											#                  {C_input: att_batch,
											#                   lr: learning_rate
											#                   })
											# t_2_X.append(c_vec)
									else:
										t_2_X.extend(att_op.tolist())
								else:
									f_o = en_outputs[index, :, :]
									if Bi_LSTM:
										f_o_bw = encoder_outputs_bw[index, :, :]
										f_o = f_o_bw + f_o
									if manner == 'sc':
										f_o = np.reshape(f_o, [-1]).tolist()
										t_2_X.append(f_o)
									else:
										t_2_X.extend(f_o.tolist())
						if Frozen == '1':
							att_batch = np.array(att_batch)
							[c_vec] = sess.run([contrast_v],
							                   {C_input: att_batch,
							                    lr: learning_rate
							                    })
							t_2_X.extend(c_vec)
						if manner == 'sc' or Frozen == '1' or C_reid == '1':
							t_2_y.extend([k] * (batch_size - flag))
							# t_2_C_y.extend([k] * (batch_size - flag))
						else:
							t_2_y.extend([k] * (batch_size - flag) * time_steps)
		X_1 = np.array(X)
		y_1 = np.array(y)
		X_pred_1 = np.array(X_pred)
		t_X_pred_1 = np.array(t_X_pred)
		ids_keys = sorted(list(ids.keys()))
		t_ids_keys = sorted(list(t_ids.keys()))
		classes = [i for i in ids_keys]
		t_classes = [i for i in t_ids_keys]
		t_y = label_binarize(t_y, classes=t_classes)
		if dataset == 'IAS':
			t_2_ids_keys = sorted(list(t_2_ids.keys()))
			t_2_classes = [i for i in t_2_ids_keys]
			t_2_y = label_binarize(t_2_y, classes=t_2_classes)
			t_2_y_1 = t_2_y
			t_2_X_1 = t_2_X
		t_y = np.array(t_y)
		if C_reid == '1':
			t_C_X = np.array(t_C_X)
		else:
			t_X = np.array(t_X)
		t_y_1 = t_y
		t_X_1 = t_X

		# checkpoint 2
		_targets = copy.deepcopy(_input_data)
		for i in range(_input_data.shape[0]):
			permutation_ = np.random.permutation(time_steps)
			_input_data[i] = _input_data[i, permutation_]
		loaded_graph = tf.Graph()
		with tf.Session(graph=loaded_graph, config=config) as sess:
			loader = tf.train.import_meta_graph(checkpoint_2 + '.meta')
			loader.restore(sess, checkpoint)
			input_data = loaded_graph.get_tensor_by_name('inputs:0')
			targets = loaded_graph.get_tensor_by_name('targets:0')
			if Frozen == '1':
				contrast_v = loaded_graph.get_tensor_by_name('add_16:0')
				C_lr = loaded_graph.get_tensor_by_name('learning_rate_1:0')
				C_input = loaded_graph.get_tensor_by_name('C_inptuiut:0')
			else:
				contrast_v = loaded_graph.get_tensor_by_name("add_2:0")
				C_lr = loaded_graph.get_tensor_by_name('learning_rate_1:0')
			if Bi_LSTM:
				encoder_output = loaded_graph.get_tensor_by_name('bidirectional_rnn/fw/fw/transpose_1:0')
				encoder_c_1 = loaded_graph.get_tensor_by_name('bidirectional_rnn/fw/fw/while/Exit_3:0')
				encoder_h_1 = loaded_graph.get_tensor_by_name('bidirectional_rnn/fw/fw/while/Exit_4:0')
				encoder_c = loaded_graph.get_tensor_by_name('bidirectional_rnn/fw/fw/while/Exit_5:0')
				encoder_h = loaded_graph.get_tensor_by_name('bidirectional_rnn/fw/fw/while/Exit_6:0')
				encoder_output_bw = loaded_graph.get_tensor_by_name('ReverseSequence: 0')
				encoder_c_1_bw = loaded_graph.get_tensor_by_name('bidirectional_rnn/bw/bw/while/Exit_3:0')
				encoder_h_1_bw = loaded_graph.get_tensor_by_name('bidirectional_rnn/bw/bw/while/Exit_4:0')
				encoder_c_bw = loaded_graph.get_tensor_by_name('bidirectional_rnn/bw/bw/while/Exit_5:0')
				encoder_h_bw = loaded_graph.get_tensor_by_name('bidirectional_rnn/bw/bw/while/Exit_6:0')
				predictions = loaded_graph.get_tensor_by_name('predictions:0')
			else:
				encoder_output = loaded_graph.get_tensor_by_name('rnn/transpose_1:0')
				encoder_c_1 = loaded_graph.get_tensor_by_name('rnn/while/Exit_3:0')
				encoder_h_1 = loaded_graph.get_tensor_by_name('rnn/while/Exit_4:0')
				encoder_c = loaded_graph.get_tensor_by_name('rnn/while/Exit_5:0')
				encoder_h = loaded_graph.get_tensor_by_name('rnn/while/Exit_6:0')
				predictions = loaded_graph.get_tensor_by_name('predictions:0')
				# train_output = loaded_graph.get_tensor_by_name('train_output:0')
			if use_attention:
				alignment_history = loaded_graph.get_tensor_by_name('train_attention_matrix:0')
				# train_attention_matrix = loaded_graph.get_tensor_by_name('train_attention_matrix:0')
				attention_state = loaded_graph.get_tensor_by_name('decode/decoder/while/Exit_12:0')
				attention_weights = loaded_graph.get_tensor_by_name('decode/decoder/while/Exit_8:0')
				alignment = loaded_graph.get_tensor_by_name('decode/decoder/while/Exit_10:0')
			lr = loaded_graph.get_tensor_by_name('learning_rate:0')
			keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
			source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
			target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
			X = []
			C_X = []
			X_all_op = []
			X_final_op = []
			X_final_c = []
			X_final_h = []
			X_final_c1 = []
			X_final_h1 = []
			X_final_ch = []
			X_final_ch1 = []
			y = []
			C_y = []
			X_pred = []
			t_X = []
			t_C_X = []
			t_y = []
			t_C_y = []
			t_2_X = []
			t_2_C_X = []
			t_2_y = []
			t_2_C_y = []
			t_X_pred = []
			t_X_att = []
			# print(t_ids)
			# print(test_attention)
			ids_ = sorted(ids.items(), key=lambda item:item[0])
			t_ids_ = sorted(t_ids.items(), key=lambda item: item[0])
			if dataset == 'IAS':
				t_2_ids_ = sorted(t_2_ids.items(), key=lambda item: item[0])
			# print(ids_)
			# exit(1)
			for k, v in ids_:
				if len(v) == 0:
					print(k)
					continue
				if len(v) < batch_size:
					v.extend([v[0]] * (batch_size - len(v)))
				# print('%s - %d' % (k, len(v)))
				for batch_i in range(len(v) // batch_size):
					this_input = _input_data[v[batch_i * batch_size : (batch_i + 1) * batch_size]]
					this_targets = _targets[v[batch_i * batch_size : (batch_i + 1) * batch_size]]
					if use_attention:
						if Frozen == '1':
							en_outputs, en_c, en_h, en_c_1, en_h_1, pred, att_state, att_history, att, align = sess.run([encoder_output, encoder_c,
							encoder_h, encoder_c_1, encoder_h_1, predictions, attention_state, alignment_history, attention_weights, alignment],
							                      {input_data: this_input,
							                       targets: this_targets,
							                       lr: learning_rate,
							                       target_sequence_length: [time_steps] * batch_size,
							                       source_sequence_length: [time_steps] * batch_size,
							                       keep_prob: 1.0})
						else:
							en_outputs, c_vec, en_c, en_h, en_c_1, en_h_1, pred, att_state, att_history, att, align = sess.run(
								[encoder_output, contrast_v, encoder_c,
								 encoder_h, encoder_c_1, encoder_h_1, predictions, attention_state, alignment_history,
								 attention_weights, alignment],
								{input_data: this_input,
								 targets: this_targets,
								 lr: learning_rate,
								 C_lr: learning_rate,
								 target_sequence_length: [time_steps] * batch_size,
								 source_sequence_length: [time_steps] * batch_size,
								 keep_prob: 1.0})
							if C_reid == '1':
								C_X.extend(c_vec)
					else:
						if Bi_LSTM:
							en_outputs, encoder_outputs_bw, en_c, en_h, en_c_1, en_h_1, pred = sess.run(
								[encoder_output, encoder_output_bw, encoder_c,
								 encoder_h, encoder_c_1, encoder_h_1, predictions],
								{input_data: this_input,
								 targets: this_targets,
								 lr: learning_rate,
								 target_sequence_length: [time_steps] * batch_size,
								 source_sequence_length: [time_steps] * batch_size,
								 keep_prob: 1.0})
						else:
							en_outputs, en_c, en_h, en_c_1, en_h_1, pred = sess.run(
								[encoder_output, encoder_c,
								 encoder_h, encoder_c_1, encoder_h_1, predictions],
								{input_data: this_input,
								 targets: this_targets,
								 lr: learning_rate,
								 target_sequence_length: [time_steps] * batch_size,
								 source_sequence_length: [time_steps] * batch_size,
								 keep_prob: 1.0})
					att_batch = []
					for index in range(en_outputs.shape[0]):
						t1 = np.reshape(en_outputs[index], [-1]).tolist()
						t2 = np.reshape(en_c[index], [-1]).tolist()
						t3 = np.reshape(en_h[index], [-1]).tolist()
						t4 = np.reshape(en_c_1[index], [-1]).tolist()
						t5 = np.reshape(en_h_1[index], [-1]).tolist()
						if type == 'c':
							X.append(t2)
						elif type == 'ch':
							t3.extend(t2)
							X.append(t3)
						elif type == 'o':
							X.append(t1)
						elif type == 'oc':
							t1.extend(t2)
							X.append(t1)
						elif type == 'att':
							if use_attention and AGEs:
								weights = att_history[:, index, :]
								f_o = en_outputs[index, :, :]
								att_op = np.matmul(weights, f_o)
								if manner == 'sc' or Frozen == '1':
									att_op = np.reshape(att_op, [-1]).tolist()
									if manner == 'sc':
										X.append(att_op)
									else:
										att_batch.append(att_op)
									# else:
									# 	att_batch = np.array(att_batch)
									# 	c_vec = sess.run([contrast_v],
							        #                     {C_input: att_batch,
							        #                      lr: learning_rate
							        #                      })
									# 	X.append(c_vec)
								else:
									X.extend(att_op.tolist())
							# for index in range(en_outputs.shape[0]):
							# 	t1 = np.reshape(en_outputs[index], [-1]).tolist()
							# 	if use_attention and AGEs:
							# 		weights = att_history[:, index, :]
							# 		f_o = en_outputs[index, :, :]
							# 		att_op = np.matmul(weights, f_o)
							# 		# if manner == 'sc':
							# 		att_op = np.reshape(att_op, [-1]).tolist()
							# 		att_batch.append(att_op)
							# att_batch = np.array(att_batch)
							# _, c_loss, c_vec = sess.run([C_train_op, C_loss, contrast_v],
							#                  {C_input: att_batch,
							#                   lr: learning_rate
							#                  })
							else:
								f_o = en_outputs[index, :, :]
								if Bi_LSTM:
									f_o_bw = encoder_outputs_bw[index, :, :]
									f_o = f_o + f_o_bw
								if manner == 'sc':
									f_o = np.reshape(f_o, [-1]).tolist()
									X.append(f_o)
								else:
									X.extend(f_o.tolist())
						elif type == 'attc':
							weights = att_history[:, index, :]
							f_o = en_outputs[index, :, :]
							att_op = np.matmul(weights, f_o)
							att_op = np.reshape(att_op, [-1]).tolist()
							att_op.extend(t2)
							X.append(att_op)
					if Frozen == '1':
						att_batch = np.array(att_batch)
						[c_vec] = sess.run([contrast_v],
					                    {C_input: att_batch,
					                     lr: learning_rate
					                     })
						X.extend(c_vec)
					pred = pred.tolist()
					for index, i in enumerate(pred):
						pred[index].reverse()
					pred = np.array(pred)
					# en_outputs, en_c, en_h, en_c_1, en_h_1, pred_2 = sess.run(
					# 	[encoder_output, encoder_c, encoder_h, encoder_c_1, encoder_h_1, predictions],
					# 	{input_data: pred,
					# 	 targets: this_targets,
					# 	 lr: learning_rate,
					# 	 target_sequence_length: [time_steps] * batch_size,
					# 	 source_sequence_length: [time_steps] * batch_size,
					# 	 keep_prob: 0.5})
					# for index in range(en_outputs.shape[0]):
					# 	t1 = np.reshape(en_outputs[index], [-1]).tolist()
					# 	t2 = np.reshape(en_c[index], [-1]).tolist()
					# 	t3 = np.reshape(en_h[index], [-1]).tolist()
					# 	t4 = np.reshape(en_c_1[index], [-1]).tolist()
					# 	t5 = np.reshape(en_h_1[index], [-1]).tolist()
					# 	if type == 'c':
					# 		X_pred.append(t2)
					# 	elif type == 'ch':
					# 		t3.extend(t2)
					# 		X_pred.append(t3)
					# 	elif type == 'o':
					# 		X_pred.append(t1)
					# 	elif type == 'oc':
					# 		t1.extend(t2)
					# 		X_pred.append(t1)
					# 	elif type == 'att':
					# 		if use_attention and AGEs:
					# 			weights = att_history[:, index, :]
					# 			f_o = en_outputs[index, :, :]
					# 			att_op = np.matmul(weights, f_o)
					# 			if manner == 'sc':
					# 				att_op = np.reshape(att_op, [-1]).tolist()
					# 				X_pred.append(att_op)
					# 			else:
					# 				X_pred.extend(att_op.tolist())
					# 		else:
					# 			f_o = en_outputs[index, :, :]
					# 			# if Bi_LSTM:
					# 			# 	f_o_bw = encoder_outputs_bw[index, :, :]
					# 			# 	f_o = f_o + f_o_bw
					# 			if manner == 'sc':
					# 				f_o = np.reshape(f_o, [-1]).tolist()
					# 				X_pred.append(f_o)
					# 			else:
					# 				X_pred.extend(f_o.tolist())
					# 	elif type == 'attc':
					# 		weights = att_history[:, index, :]
					# 		f_o = en_outputs[index, :, :]
					# 		att_op = np.matmul(weights, f_o)
					# 		att_op = np.reshape(att_op, [-1]).tolist()
					# 		att_op.extend(t2)
					# 		X_pred.append(att_op)
					if manner == 'sc' or Frozen == '1' or C_reid == '1':
						y.extend([k] * batch_size)
						# C_y.extend([k] * batch_size)
					else:
						y.extend([k] * batch_size * time_steps)
			# exit(1)
			for k, v in t_ids_:
				flag = 0
				if len(v) == 0:
					continue
				if len(v) < batch_size:
					flag = batch_size - len(v)
					v.extend([v[0]] * (batch_size - len(v)))
				# print('%s - %d' % (k, len(v)))
				for batch_i in range(len(v) // batch_size):
					this_input = t_input_data[v[batch_i * batch_size : (batch_i + 1) * batch_size]]
					this_targets = t_targets[v[batch_i * batch_size : (batch_i + 1) * batch_size]]
					if use_attention:
						if Frozen == '1':
							en_outputs, en_c, en_h, en_c_1, en_h_1, pred, att_state, att_history, att, align = sess.run([encoder_output, encoder_c,
							encoder_h, encoder_c_1, encoder_h_1, predictions, attention_state, alignment_history, attention_weights, alignment],
							                      {input_data: this_input,
							                       targets: this_targets,
							                       lr: learning_rate,
							                       target_sequence_length: [time_steps] * batch_size,
							                       source_sequence_length: [time_steps] * batch_size,
							                       keep_prob: 1.0})
						else:
							en_outputs, c_vec, en_c, en_h, en_c_1, en_h_1, pred, att_state, att_history, att, align = sess.run(
								[encoder_output, contrast_v, encoder_c,
								 encoder_h, encoder_c_1, encoder_h_1, predictions, attention_state, alignment_history,
								 attention_weights, alignment],
								{input_data: this_input,
								 targets: this_targets,
								 lr: learning_rate,
								 C_lr: learning_rate,
								 target_sequence_length: [time_steps] * batch_size,
								 source_sequence_length: [time_steps] * batch_size,
								 keep_prob: 1.0})
							if C_reid == '1':
								t_C_X.extend(c_vec)
					else:
						if Bi_LSTM:
							en_outputs, encoder_outputs_bw, en_c, en_h, en_c_1, en_h_1, pred = sess.run(
								[encoder_output, encoder_output_bw, encoder_c,
								 encoder_h, encoder_c_1, encoder_h_1, predictions],
								{input_data: this_input,
								 targets: this_targets,
								 lr: learning_rate,
								 target_sequence_length: [time_steps] * batch_size,
								 source_sequence_length: [time_steps] * batch_size,
								 keep_prob: 1.0})
						else:
							en_outputs, en_c, en_h, en_c_1, en_h_1, pred = sess.run(
								[encoder_output, encoder_c,
								 encoder_h, encoder_c_1, encoder_h_1, predictions],
								{input_data: this_input,
								 targets: this_targets,
								 lr: learning_rate,
								 target_sequence_length: [time_steps] * batch_size,
								 source_sequence_length: [time_steps] * batch_size,
								 keep_prob: 1.0})
					if flag > 0:
						en_outputs = en_outputs[:-flag]
					att_batch = []
					for index in range(en_outputs.shape[0]):
						t1 = np.reshape(en_outputs[index], [-1]).tolist()
						t2 = np.reshape(en_c[index], [-1]).tolist()
						t3 = np.reshape(en_h[index], [-1]).tolist()
						t4 = np.reshape(en_c_1[index], [-1]).tolist()
						t5 = np.reshape(en_h_1[index], [-1]).tolist()
						if type == 'att':
							if use_attention and AGEs:
								weights = att_history[:, index, :]
								f_o = en_outputs[index, :, :]
								att_op = np.matmul(weights, f_o)
								# if manner == 'sc':
								# 	att_op = np.reshape(att_op, [-1]).tolist()
								# 	t_X.append(att_op)
								if manner == 'sc' or Frozen == '1':
									att_op = np.reshape(att_op, [-1]).tolist()
									if manner == 'sc':
										t_X.append(att_op)
									else:
										att_batch.append(att_op)
										# att_batch = np.array(att_batch)
										# c_vec = sess.run([contrast_v],
										#                  {C_input: att_batch,
										#                   lr: learning_rate
										#                   })
										# t_X.append(c_vec)
								else:
									t_X.extend(att_op.tolist())
								t_X_att.append(weights)
							else:
								f_o = en_outputs[index, :, :]
								if Bi_LSTM:
									f_o_bw = encoder_outputs_bw[index, :, :]
									f_o = f_o_bw + f_o
								if manner == 'sc':
									f_o = np.reshape(f_o, [-1]).tolist()
									t_X.append(f_o)
								else:
									t_X.extend(f_o.tolist())
						elif type == 'attc':
							weights = att_history[:, index, :]
							f_o = en_outputs[index, :, :]
							att_op = np.matmul(weights, f_o)
							att_op = np.reshape(att_op, [-1]).tolist()
							att_op.extend(t2)
							t_X.append(att_op)
					if Frozen == '1':
						att_batch = np.array(att_batch)
						[c_vec] = sess.run([contrast_v],
					                    {C_input: att_batch,
					                     lr: learning_rate
					                     })
						t_X.extend(c_vec)
					pred = pred.tolist()
					for index, i in enumerate(pred):
						pred[index].reverse()
					pred = np.array(pred)
					# en_outputs, en_c, en_h, en_c_1, en_h_1, pred_2 = sess.run(
					# 	[encoder_output, encoder_c, encoder_h, encoder_c_1, encoder_h_1, predictions],
					# 	{input_data: pred,
					# 	 targets: this_targets,
					# 	 lr: learning_rate,
					# 	 target_sequence_length: [time_steps] * batch_size,
					# 	 source_sequence_length: [time_steps] * batch_size,
					# 	 keep_prob: 0.5})
					# for index in range(en_outputs.shape[0]):
					# 	t1 = np.reshape(en_outputs[index], [-1]).tolist()
					# 	t2 = np.reshape(en_c[index], [-1]).tolist()
					# 	t3 = np.reshape(en_h[index], [-1]).tolist()
					# 	t4 = np.reshape(en_c_1[index], [-1]).tolist()
					# 	t5 = np.reshape(en_h_1[index], [-1]).tolist()
					# 	if type == 'att':
					# 		if use_attention and AGEs:
					# 			weights = att_history[:, index, :]
					# 			f_o = en_outputs[index, :, :]
					# 			att_op = np.matmul(weights, f_o)
					# 			if manner == 'sc':
					# 				att_op = np.reshape(att_op, [-1]).tolist()
					# 				t_X_pred.append(att_op)
					# 			else:
					# 				t_X_pred.extend(att_op.tolist())
					# 		else:
					# 			f_o = en_outputs[index, :, :]
					# 			if manner == 'sc':
					# 				f_o = np.reshape(f_o, [-1]).tolist()
					# 				t_X_pred.append(f_o)
					# 			else:
					# 				t_X_pred.extend(f_o.tolist())
					# 	elif type == 'attc':
					# 		weights = att_history[:, index, :]
					# 		f_o = en_outputs[index, :, :]
					# 		att_op = np.matmul(weights, f_o)
					# 		att_op = np.reshape(att_op, [-1]).tolist()
					# 		att_op.extend(t2)
					# 		t_X_pred.append(att_op)
					if manner == 'sc' or Frozen == '1' or C_reid == '1':
						t_y.extend([k] * (batch_size - flag))
						# t_C_y.extend([k] * (batch_size - flag))
					else:
						t_y.extend([k] * (batch_size - flag) * time_steps)
			if dataset == 'IAS':
				for k, v in t_2_ids_:
					flag = 0
					if len(v) == 0:
						continue
					if len(v) < batch_size:
						flag = batch_size - len(v)
						v.extend([v[0]] * (batch_size - len(v)))
					# print('%s - %d' % (k, len(v)))
					for batch_i in range(len(v) // batch_size):
						this_input = t_2_input_data[v[batch_i * batch_size : (batch_i + 1) * batch_size]]
						this_targets = t_2_targets[v[batch_i * batch_size : (batch_i + 1) * batch_size]]
						if use_attention:
							if Frozen == '1':
								en_outputs, en_c, en_h, en_c_1, en_h_1, pred, att_state, att_history, att, align = sess.run(
									[encoder_output, encoder_c,
									 encoder_h, encoder_c_1, encoder_h_1, predictions, attention_state, alignment_history,
									 attention_weights, alignment],
									{input_data: this_input,
									 targets: this_targets,
									 lr: learning_rate,
									 target_sequence_length: [time_steps] * batch_size,
									 source_sequence_length: [time_steps] * batch_size,
									 keep_prob: 1.0})
							else:
								en_outputs, c_vec, en_c, en_h, en_c_1, en_h_1, pred, att_state, att_history, att, align = sess.run(
									[encoder_output, contrast_v, encoder_c,
									 encoder_h, encoder_c_1, encoder_h_1, predictions, attention_state, alignment_history,
									 attention_weights, alignment],
									{input_data: this_input,
									 targets: this_targets,
									 lr: learning_rate,
									 C_lr: learning_rate,
									 target_sequence_length: [time_steps] * batch_size,
									 source_sequence_length: [time_steps] * batch_size,
									 keep_prob: 1.0})
								if C_reid == '1':
									t_2_C_X.extend(c_vec)
						else:
							if Bi_LSTM:
								en_outputs, encoder_outputs_bw, en_c, en_h, en_c_1, en_h_1, pred = sess.run(
									[encoder_output, encoder_output_bw, encoder_c,
									 encoder_h, encoder_c_1, encoder_h_1, predictions],
									{input_data: this_input,
									 targets: this_targets,
									 lr: learning_rate,
									 target_sequence_length: [time_steps] * batch_size,
									 source_sequence_length: [time_steps] * batch_size,
									 keep_prob: 1.0})
							else:
								en_outputs, en_c, en_h, en_c_1, en_h_1, pred = sess.run(
									[encoder_output, encoder_c,
									 encoder_h, encoder_c_1, encoder_h_1, predictions],
									{input_data: this_input,
									 targets: this_targets,
									 lr: learning_rate,
									 target_sequence_length: [time_steps] * batch_size,
									 source_sequence_length: [time_steps] * batch_size,
									 keep_prob: 1.0})
						if flag > 0:
							en_outputs = en_outputs[:-flag]
						att_batch = []
						for index in range(en_outputs.shape[0]):
							t1 = np.reshape(en_outputs[index], [-1]).tolist()
							t2 = np.reshape(en_c[index], [-1]).tolist()
							t3 = np.reshape(en_h[index], [-1]).tolist()
							t4 = np.reshape(en_c_1[index], [-1]).tolist()
							t5 = np.reshape(en_h_1[index], [-1]).tolist()
							if type == 'att':
								if use_attention and AGEs:
									weights = att_history[:, index, :]
									f_o = en_outputs[index, :, :]
									att_op = np.matmul(weights, f_o)
									if manner == 'sc' or Frozen == '1':
										att_op = np.reshape(att_op, [-1]).tolist()
										if manner == 'sc':
											t_2_X.append(att_op)
										else:
											att_batch.append(att_op)
											# att_batch = np.array(att_batch)
											# c_vec = sess.run([contrast_v],
											#                  {C_input: att_batch,
											#                   lr: learning_rate
											#                   })
											# t_2_X.append(c_vec)
									else:
										t_2_X.extend(att_op.tolist())
								else:
									f_o = en_outputs[index, :, :]
									if Bi_LSTM:
										f_o_bw = encoder_outputs_bw[index, :, :]
										f_o = f_o_bw + f_o
									if manner == 'sc':
										f_o = np.reshape(f_o, [-1]).tolist()
										t_2_X.append(f_o)
									else:
										t_2_X.extend(f_o.tolist())
						if Frozen == '1':
							att_batch = np.array(att_batch)
							[c_vec] = sess.run([contrast_v],
							                   {C_input: att_batch,
							                    lr: learning_rate
							                    })
							t_2_X.extend(c_vec)
						if manner == 'sc' or Frozen == '1' or C_reid == '1':
							t_2_y.extend([k] * (batch_size - flag))
							# t_2_C_y.extend([k] * (batch_size - flag))
						else:
							t_2_y.extend([k] * (batch_size - flag) * time_steps)
		X_2 = np.array(X)
		y_2 = np.array(y)
		X_pred_2 = np.array(X_pred)
		t_X_pred_2 = np.array(t_X_pred)
		ids_keys = sorted(list(ids.keys()))
		t_ids_keys = sorted(list(t_ids.keys()))
		classes = [i for i in ids_keys]
		t_classes = [i for i in t_ids_keys]
		t_y = label_binarize(t_y, classes=t_classes)
		if dataset == 'IAS':
			t_2_ids_keys = sorted(list(t_2_ids.keys()))
			t_2_classes = [i for i in t_2_ids_keys]
			t_2_y = label_binarize(t_2_y, classes=t_2_classes)
			t_2_y_2 = t_2_y
			t_2_X_2 = t_2_X
		t_y = np.array(t_y)
		if C_reid == '1':
			t_C_X = np.array(t_C_X)
		else:
			t_X = np.array(t_X)
		t_y_2 = t_y
		t_X_2 = t_X
	# preds = np.array(preds)
	# preditions = np.array(preditions)
	global permutation, permutation_flag, permutation_test_flag, permutation_test_2_flag, test_permutation, test_2_permutation
	if not permutation_flag:
		if C_reid == '1':
			permutation = np.random.permutation(C_X.shape[0])
		else:
			permutation = np.random.permutation(X_0.shape[0])
		permutation_flag = True
	# useless C_X
	if C_reid == '1':
		C_X = C_X[permutation, ]
	else:
		X_0 = X_0[permutation, ]
		if Model == 'rev_rec_plus':
			X_1 = X_1[permutation,]
			X_2 = X_2[permutation,]
	y_0 = y_0[permutation, ]
	if Model == 'rev_rec_plus':
		y_1 = y_1[permutation,]
		y_2 = y_2[permutation,]
	# X_pred = X_pred[permutation, ]

	if not permutation_test_flag:
		if C_reid == '1':
			test_permutation = np.random.permutation(t_C_X.shape[0])
		else:
			test_permutation = np.random.permutation(t_X_0.shape[0])
		permutation_test_flag = True
	if manner == 'sc':
		if C_reid == '1':
			t_C_X = t_C_X[test_permutation, ]
		else:
			t_X_0 = t_X_0[test_permutation,]
			if Model == 'rev_rec_plus':
				t_X_1 = t_X_1[test_permutation,]
				t_X_2 = t_X_2[test_permutation,]
		t_y_0 = t_y_0[test_permutation,]
		if Model == 'rev_rec_plus':
			t_y_1 = t_y_1[test_permutation,]
			t_y_2 = t_y_2[test_permutation,]
	# t_X_att = t_X_att[test_permutation]
	if dataset == 'IAS':
		# valid_2_source = t_2_X
		# valid_2_target = t_2_y
		if C_reid == '1':
			t_2_C_X = np.array(t_2_C_X)
			t_2_C_y = np.array(t_2_C_y)
		else:
			t_2_X_0 = np.array(t_2_X_0)
			t_2_y_0 = np.array(t_2_y_0)
			if Model == 'rev_rec_plus':
				t_2_X_1 = np.array(t_2_X_1)
				t_2_y_1 = np.array(t_2_y_1)
				t_2_X_2 = np.array(t_2_X_2)
				t_2_y_2 = np.array(t_2_y_2)
		if not permutation_test_2_flag:
			if C_reid == '1':
				test_2_permutation = np.random.permutation(t_2_C_X.shape[0])
			else:
				test_2_permutation = np.random.permutation(t_2_X_0.shape[0])
			permutation_test_2_flag = True
		if manner == 'sc':
			if C_reid == '1':
				t_2_C_X = t_2_C_X[test_2_permutation,]
			else:
				t_2_X_0 = t_2_X_0[test_2_permutation,]
				if Model == 'rev_rec_plus':
					t_2_X_1 = t_2_X_1[test_2_permutation,]
					t_2_X_2 = t_2_X_2[test_2_permutation,]
			t_2_y_0 = t_2_y_0[test_2_permutation,]
			if Model == 'rev_rec_plus':
				t_2_y_1 = t_2_y_1[test_2_permutation,]
				t_2_y_2 = t_2_y_2[test_2_permutation,]
	# print(X_0.shape, t_X_0.shape)
	# exit(0)
	if Model == 'rev_rec_plus':
		X = np.concatenate((X_0, X_1, X_2), axis=1)
		t_X = np.concatenate((t_X_0, t_X_1, t_X_2), axis=1)
	else:
		X = X_0
		t_X = t_X_0
	y = y_0
	t_y = t_y_0
	if dataset == 'IAS':
		if Model == 'rev_rec_plus':
			t_2_X = np.concatenate((t_2_X_0, t_2_X_1, t_2_X_2), axis=1)
		else:
			t_2_X = t_2_X_0
		t_2_y = t_2_y_0
		if C_reid == '1':
			return C_X, y, t_C_X, t_y, t_2_C_X, t_2_y, t_X_att
		else:
			return X, y, t_X, t_y, t_2_X, t_2_y, t_X_att
	else:
		if C_reid == '1':
			return C_X, y, t_C_X, t_y, t_X_att
		else:
			return X, y, t_X, t_y, t_X_att

def get_new_train_batches(targets, sources, batch_size):
	if len(targets) < batch_size:
		yield targets, sources
	else:
		for batch_i in range(0, len(sources) // batch_size):
			start_i = batch_i * batch_size
			sources_batch = sources[start_i:start_i + batch_size]
			targets_batch = targets[start_i:start_i + batch_size]
			yield targets_batch, sources_batch

def encoder_classify_union_directly(X, y, t_X, t_y, new_dir, ps, dataset):
	global epochs, attention, manner, view
	epochs = 300
	if dataset == 'KGBD':
		epochs = 200
	if dataset == 'KS20':
		epochs = 800
	try:
		os.mkdir(new_dir)
	except:
		pass
	if view != '':
		view_dir = view + '/'
	else:
		view_dir = ''
	from sklearn.preprocessing import label_binarize
	if dataset == 'IAS':
		dataset = 'IAS'
	ids = np.load('Datasets/' + frames_ps + view_dir + dataset + '_train_npy_data/ids_' + dataset + '_' + str(time_steps) + '.npy')
	ids = ids.item()
	t_ids = np.load('Datasets/' + frames_ps + view_dir + dataset + '_test_npy_data/ids_' + dataset + '_' + str(time_steps) + '.npy')
	t_ids = t_ids.item()
	ids_keys = sorted(list(ids.keys()))
	classes = [i for i in ids_keys]
	y = label_binarize(y, classes=classes)
	t_ids_keys = sorted(list(t_ids.keys()))
	classes = [i for i in t_ids_keys]
	t_y = label_binarize(t_y, classes=classes)
	train_source = X
	train_target = y
	valid_source = t_X
	valid_target = t_y
	if Model == 'rev_rec_plus':
		if manner == 'sc':
			first_size = rnn_size * time_steps * 3 * 3
		else:
			first_size = rnn_size * 3 * 3
		X_input = tf.placeholder(tf.float32, [None, first_size], name='X_input')
		y_input = tf.placeholder(tf.int32, [None, len(classes)], name='y_input')
		lr = tf.Variable(0.0005, trainable=False, dtype=tf.float32, name='learning_rate')
		W1 = tf.Variable(tf.random_normal([first_size, rnn_size]), name='W1')
		b1 = tf.Variable(tf.zeros(shape=[rnn_size, ]), name='b1')
		Wx_plus_b1 = tf.matmul(X_input, W1) + b1
		l1 = tf.nn.relu(Wx_plus_b1)
		W = tf.Variable(tf.random_normal([rnn_size, len(classes)]), name='W')
		b = tf.Variable(tf.zeros(shape=[len(classes), ], name='b'))
		pred = tf.matmul(l1, W) + b
	else:
		if manner == 'sc':
			first_size = rnn_size * time_steps * 3
		else:
			first_size = rnn_size * 3
		X_input = tf.placeholder(tf.float32, [None, first_size], name='X_input')
		y_input = tf.placeholder(tf.int32, [None, len(classes)], name='y_input')
		lr = tf.Variable(0.0005, trainable=False, dtype=tf.float32, name='learning_rate')
		W1 = tf.Variable(tf.random_normal([first_size, rnn_size]), name='W1')
		b1 = tf.Variable(tf.zeros(shape=[rnn_size, ]), name='b1')
		Wx_plus_b1 = tf.matmul(X_input, W1) + b1
		l1 = tf.nn.relu(Wx_plus_b1)
		W = tf.Variable(tf.random_normal([rnn_size, len(classes)]), name='W')
		b = tf.Variable(tf.zeros(shape=[len(classes), ], name='b'))
		pred = tf.matmul(l1, W) + b
	with tf.name_scope("new_train"):
		optimizer = tf.train.AdamOptimizer(learning_rate, name="Adam3")
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_input))
		gradients = optimizer.compute_gradients(cost)
		capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
		train_op = optimizer.minimize(cost)
		correct_pred = tf.equal(tf.argmax(pred, 1),tf.argmax(y_input, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	def get_new_train_batches(targets, sources, batch_size):
		for batch_i in range(0, len(sources) // batch_size):
			start_i = batch_i * batch_size
			sources_batch = sources[start_i:start_i + batch_size]
			targets_batch = targets[start_i:start_i + batch_size]
			yield targets_batch, sources_batch
	init = tf.global_variables_initializer()
	with tf.Session(config=config) as sess:
		sess.run(init)
		step = 0
		train_loss = []
		test_loss = []
		accs = []
		val_accs = [0]
		max_acc = 0
		saver = tf.train.Saver()
		try:
			os.mkdir(new_dir)
		except:
			pass
		new_dir += '/' + ps
		try:
			os.mkdir(new_dir)
		except:
			pass
		# if attention == 'BA':
		# 	manner == 'sc'
		for epoch_i in range(1, epochs + 1):
			for batch_i, (y_batch, X_batch) in enumerate(
					get_new_train_batches(train_target, train_source, batch_size)):
				_, loss, acc = sess.run([train_op, cost, accuracy],
				                   {X_input: X_batch,
				                    y_input: y_batch,
				                    lr: learning_rate})
				accs.append(acc)
			if epoch_i % 1 == 0:
				loss, train_acc = sess.run([cost, accuracy],
		                        {X_input: X_batch,
		                         y_input: y_batch,
		                         lr: learning_rate})
				val_loss = []
				val_acc = []
				flag = 0
				if valid_source.shape[0] < batch_size:
					flag = batch_size - valid_source.shape[0]
					valid_source = valid_source.tolist()
					valid_target = valid_target.tolist()
					valid_source.extend([valid_source[0]] * flag)
					valid_target.extend([valid_target[0]] * flag)
					valid_source = np.array(valid_source)
					valid_target = np.array(valid_target)
				if manner == 'ap':
					all_frame_preds = []
				for k in range(valid_source.shape[0] // batch_size):
					if manner == 'sc':
						val_loss_t, val_acc_t = sess.run(
							[cost, accuracy],
							{X_input: valid_source[k * batch_size: (k + 1) * batch_size],
							 y_input: valid_target[k * batch_size: (k + 1) * batch_size],
							 lr: learning_rate})
						val_loss.append(val_loss_t)
						val_acc.append(val_acc_t)
					else:
						val_loss_t, val_acc_t, frame_preds = sess.run(
							[cost, accuracy, pred],
							{X_input: valid_source[k * batch_size: (k + 1) * batch_size],
							 y_input: valid_target[k * batch_size: (k + 1) * batch_size],
							 lr: learning_rate})
						# pred_prob = frame_preds / np.tile(np.sum(frame_preds, axis=1), [frame_preds.shape[1], 1]).T
						# pred_prob = np.sum(frame_preds, axis=0)
						# all_frame_preds.extend(pred_prob)
						all_frame_preds.extend(frame_preds)
						val_loss.append(val_loss_t)
						val_acc.append(val_acc_t)
				if manner == 'ap':
					sequence_pred_correct = 0
					sequence_num = 0
					for k in range(len(all_frame_preds) // time_steps):
						sequence_labels = np.argmax(valid_target[k * time_steps: (k + 1) * time_steps], axis=1)
						if (sequence_labels == np.tile(sequence_labels[0], [sequence_labels.shape[0]])).all():
							frame_predictions = np.array(all_frame_preds[k * time_steps: (k + 1) * time_steps])
							sequence_pred = np.argmax(np.average(frame_predictions, axis=0))
							if sequence_pred == sequence_labels[0]:
								# print(sequence_pred)
								sequence_pred_correct += 1
							sequence_num += 1
					seq_acc_t = sequence_pred_correct / sequence_num
					# val_acc.append(val_acc_t)
				if manner == 'sc':
					if sum(val_acc) / len(val_acc) >= max(val_accs):
						saver.save(sess, new_dir + "/trained_model.ckpt")
					val_accs.append(sum(val_acc) / len(val_acc))
					print(
						'Epoch {:>3}/{} Batch {:>4}/{} - Train Loss: {:>6.3f}  - Train_Acc: {:>6.3f} - Val_Acc {:>6.3f} {:>6.3f} (max)'
							.format(epoch_i,
						            epochs,
						            batch_i,
						            len(train_target) // batch_size,
						            loss,
						            train_acc,
						            sum(val_acc) / len(val_acc),
                                    max(val_accs)
						            ))
				else:
					if seq_acc_t >= max(val_accs):
						saver.save(sess, new_dir + "/trained_model.ckpt")
						# np.save(new_dir + '/val_X.npy', valid_source)
						# np.save(new_dir + '/val_y.npy', valid_target)
						# if epoch_i % 30 == 0:
						# 	evaluate_reid('CAGEs_RN_models/' + dataset + '_' + attention + '_RN_' + manner)
					val_accs.append(seq_acc_t)
					print(
						'Epoch {:>3}/{} Batch {:>4}/{} - Train Loss: {:>6.3f}  - Train_Acc: {:>6.3f} - Val_Acc {:>6.3f} {:>6.3f} (max)'
							.format(epoch_i,
						            epochs,
						            batch_i,
						            len(train_target) // batch_size,
						            loss,
						            train_acc,
						            seq_acc_t,
                                    max(val_accs)
						            ))
			train_loss.append(loss)
			test_loss.append(sum(val_loss) / len(val_loss))
			step += 1
		# saver.save(sess, new_dir + "/trained_model.ckpt")
		np.save(new_dir + '/train_X.npy', train_source)
		np.save(new_dir + '/train_y.npy', train_target)
		np.save(new_dir + '/val_X.npy', valid_source)
		np.save(new_dir + '/val_y.npy', valid_target)
		print('Model Trained and Saved')
		np.save(new_dir + '/train_loss.npy', np.array(train_loss))
		np.save(new_dir + '/test_loss.npy', np.array(test_loss))
		np.save(new_dir + '/acc.npy', np.array(accs))
		disc_str = ''
		disc_str += str(train_loss[-1]) + '-' + str(np.min(train_loss)) + ' ' + str(test_loss[-1]) + '-' + str(
			np.min(test_loss)) + ' ' \
		            + str(np.max(acc))
		f = open(ps + '.txt', 'w')
		f.write(disc_str)
		f.close()
	return 1


def encoder_classify_union_directly_IAS(X, y, t_X, t_y, t_2_X, t_2_y, new_dir, ps, dataset):
	print(t_X.shape, t_y.shape)
	global epochs, attention, manner
	epochs = 300
	try:
		os.mkdir(new_dir)
	except:
		pass
	from sklearn.preprocessing import label_binarize
	ids = np.load('Datasets/' + frames_ps + dataset + '_train_npy_data/ids_' + dataset + '_' + str(time_steps) + '.npy')
	ids = ids.item()
	t_ids = np.load('Datasets/' + frames_ps + dataset + '_test_npy_data/ids_' + dataset + '-A_' + str(time_steps) + '.npy')
	t_ids = t_ids.item()
	t_2_ids = np.load('Datasets/' + frames_ps + dataset + '_test_npy_data/ids_' + dataset + '-B_' + str(time_steps) + '.npy')
	t_2_ids = t_2_ids.item()
	ids_keys = sorted(list(ids.keys()))
	classes = [i for i in ids_keys]
	y = label_binarize(y, classes=classes)
	t_ids_keys = sorted(list(t_ids.keys()))
	classes = [i for i in t_ids_keys]
	t_y = label_binarize(t_y, classes=classes)
	t_2_ids_keys = sorted(list(t_2_ids.keys()))
	classes = [i for i in t_2_ids_keys]
	t_2_y = label_binarize(t_2_y, classes=classes)
	train_source = X
	train_target = y
	valid_source = t_X
	valid_target = t_y
	valid_2_source = t_2_X
	valid_2_target = t_2_y
	if Model == 'rev_rec_plus':
		if manner == 'sc':
			first_size = rnn_size * time_steps * 3 * 3
		else:
			first_size = rnn_size * 3 * 3
		X_input = tf.placeholder(tf.float32, [None, first_size], name='X_input')
		y_input = tf.placeholder(tf.int32, [None, len(classes)], name='y_input')
		lr = tf.Variable(0.0005, trainable=False, dtype=tf.float32, name='learning_rate')

		W1 = tf.Variable(tf.random_normal([first_size, rnn_size]), name='W1')
		b1 = tf.Variable(tf.zeros(shape=[rnn_size, ]), name='b1')
		Wx_plus_b1 = tf.matmul(X_input, W1) + b1
		l1 = tf.nn.relu(Wx_plus_b1)
		W = tf.Variable(tf.random_normal([rnn_size, len(classes)]), name='W')
		b = tf.Variable(tf.zeros(shape=[len(classes), ], name='b'))
		pred = tf.matmul(l1, W) + b
	else:
		if manner == 'sc':
			first_size = rnn_size * time_steps * 3
		else:
			first_size = rnn_size * 3
		X_input = tf.placeholder(tf.float32, [None, first_size], name='X_input')
		y_input = tf.placeholder(tf.int32, [None, len(classes)], name='y_input')
		lr = tf.Variable(0.0005, trainable=False, dtype=tf.float32, name='learning_rate')

		W1 = tf.Variable(tf.random_normal([first_size, rnn_size]), name='W1')
		b1 = tf.Variable(tf.zeros(shape=[rnn_size, ]), name='b1')
		Wx_plus_b1 = tf.matmul(X_input, W1) + b1
		l1 = tf.nn.relu(Wx_plus_b1)
		W = tf.Variable(tf.random_normal([rnn_size, len(classes)]), name='W')
		b = tf.Variable(tf.zeros(shape=[len(classes), ], name='b'))
		pred = tf.matmul(l1, W) + b
	with tf.name_scope("new_train"):
		optimizer = tf.train.AdamOptimizer(learning_rate, name="Adam3")
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_input))
		gradients = optimizer.compute_gradients(cost)
		capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
		train_op = optimizer.minimize(cost)
		correct_pred = tf.equal(tf.argmax(pred, 1),tf.argmax(y_input, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	def get_new_train_batches(targets, sources, batch_size):
		for batch_i in range(0, len(sources) // batch_size):
			start_i = batch_i * batch_size
			sources_batch = sources[start_i:start_i + batch_size]
			targets_batch = targets[start_i:start_i + batch_size]
			yield targets_batch, sources_batch
	init = tf.global_variables_initializer()
	with tf.Session(config=config) as sess:
		sess.run(init)
		step = 0
		train_loss = []
		test_loss = []
		test_2_loss = []
		accs = []
		val_accs = [0]
		val_2_accs = [0]
		saver = tf.train.Saver()
		max_accs = [0, 0]
		new_dir += '/' + ps
		try:
			os.mkdir(new_dir)
		except:
			pass
		# if attention == 'BA':
		# 	manner == 'sc'
		for epoch_i in range(1, epochs + 1):
			for batch_i, (y_batch, X_batch) in enumerate(
					get_new_train_batches(train_target, train_source, batch_size)):
				_, loss, acc = sess.run([train_op, cost, accuracy],
				                   {X_input: X_batch,
				                    y_input: y_batch,
				                    lr: learning_rate})
				accs.append(acc)
			if epoch_i % 1 == 0:
				loss, train_acc = sess.run([cost, accuracy],
		                        {X_input: X_batch,
		                         y_input: y_batch,
		                         lr: learning_rate})
				val_loss = []
				val_acc = []
				val_2_loss = []
				val_2_acc = []
				flag = 0
				if valid_source.shape[0] < batch_size:
					flag = batch_size - valid_source.shape[0]
					valid_source = valid_source.tolist()
					valid_target = valid_target.tolist()
					valid_source.extend([valid_source[0]] * flag)
					valid_target.extend([valid_target[0]] * flag)
					valid_source = np.array(valid_source)
					valid_target = np.array(valid_target)
				if valid_2_source.shape[0] < batch_size:
					flag = batch_size - valid_2_source.shape[0]
					valid_2_source = valid_2_source.tolist()
					valid_2_target = valid_2_target.tolist()
					valid_2_source.extend([valid_2_source[0]] * flag)
					valid_2_target.extend([valid_2_target[0]] * flag)
					valid_2_source = np.array(valid_2_source)
					valid_2_target = np.array(valid_2_target)
				if manner == 'ap':
					all_frame_preds = []
					all_2_frame_preds = []
				for k in range(valid_source.shape[0] // batch_size):
					if manner == 'sc':
						val_loss_t, val_acc_t = sess.run(
							[cost, accuracy],
							{X_input: valid_source[k * batch_size: (k + 1) * batch_size],
							 y_input: valid_target[k * batch_size: (k + 1) * batch_size],
							 lr: learning_rate})
						val_loss.append(val_loss_t)
						val_acc.append(val_acc_t)
					else:
						val_loss_t, val_acc_t, frame_preds = sess.run(
							[cost, accuracy, pred],
							{X_input: valid_source[k * batch_size: (k + 1) * batch_size],
							 y_input: valid_target[k * batch_size: (k + 1) * batch_size],
							 lr: learning_rate})
						all_frame_preds.extend(frame_preds)
						val_loss.append(val_loss_t)
						val_acc.append(val_acc_t)
				for k in range(valid_2_source.shape[0] // batch_size):
					if manner == 'sc':
						val_2_loss_t, val_2_acc_t = sess.run(
							[cost, accuracy],
							{X_input: valid_2_source[k * batch_size: (k + 1) * batch_size],
							 y_input: valid_2_target[k * batch_size: (k + 1) * batch_size],
							 lr: learning_rate})
						val_2_loss.append(val_2_loss_t)
						val_2_acc.append(val_2_acc_t)
					else:
						val_2_loss_t, val_2_acc_t, frame_2_preds = sess.run(
							[cost, accuracy, pred],
							{X_input: valid_2_source[k * batch_size: (k + 1) * batch_size],
							 y_input: valid_2_target[k * batch_size: (k + 1) * batch_size],
							 lr: learning_rate})
						all_2_frame_preds.extend(frame_2_preds)
						val_2_loss.append(val_2_loss_t)
						val_2_acc.append(val_2_acc_t)
				if manner == 'ap':
					sequence_pred_correct = 0
					sequence_num = 0
					sequence_2_pred_correct = 0
					sequence_2_num = 0
					for k in range(len(all_frame_preds) // time_steps):
						sequence_labels = np.argmax(valid_target[k * time_steps: (k + 1) * time_steps], axis=1)
						if (sequence_labels == np.tile(sequence_labels[0], [sequence_labels.shape[0]])).all():
							frame_predictions = np.array(all_frame_preds[k * time_steps: (k + 1) * time_steps])
							sequence_pred = np.argmax(np.average(frame_predictions, axis=0))
							if sequence_pred == sequence_labels[0]:
								# print(sequence_pred)
								sequence_pred_correct += 1
							sequence_num += 1
					seq_acc_t = sequence_pred_correct / sequence_num
					for k in range(len(all_2_frame_preds) // time_steps):
						sequence_2_labels = np.argmax(valid_2_target[k * time_steps: (k + 1) * time_steps], axis=1)
						if (sequence_2_labels == np.tile(sequence_2_labels[0], [sequence_2_labels.shape[0]])).all():
							frame_2_predictions = np.array(all_2_frame_preds[k * time_steps: (k + 1) * time_steps])
							sequence_2_pred = np.argmax(np.average(frame_2_predictions, axis=0))
							if sequence_2_pred == sequence_2_labels[0]:
								# print(sequence_pred)
								sequence_2_pred_correct += 1
							sequence_2_num += 1
					seq_2_acc_t = sequence_2_pred_correct / sequence_2_num
				# val_acc.append(val_acc_t)
				if manner == 'sc':
					if sum(val_acc) / len(val_acc) >= max(val_accs) or sum(val_2_acc) / len(val_2_acc) >= max(val_2_accs):
						saver.save(sess, new_dir + "/trained_model.ckpt")
						np.save(new_dir + '/val_X.npy', valid_source)
						np.save(new_dir + '/val_y.npy', valid_target)
					val_accs.append(sum(val_acc) / len(val_acc))
					val_2_accs.append(sum(val_2_acc)/len(val_2_acc))
					print(
						'Epoch {:>3}/{} Batch {:>4}/{} - Train Loss: {:>6.3f}  - V_Acc (IAS-A): {:>6.3f} {:>6.3f} (max)  - V_Acc (IAS-B) {:>6.3f} {:>6.3f} (max)'
							.format(epoch_i,
						            epochs,
						            batch_i,
						            len(train_target) // batch_size,
						            loss,
						            sum(val_acc) / len(val_acc),
                                    max(val_accs),
						            sum(val_2_acc) / len(val_2_acc),
                                    max(val_2_accs)
						            ))
				else:
					# if seq_acc_t > 0.5 and seq_2_acc_t > 0.5:
					pre_1 = max(val_accs)
					pre_2 = max(val_2_accs)
					if seq_acc_t > pre_1:
						saver.save(sess, new_dir + "/A_trained_model.ckpt")
					if seq_2_acc_t > pre_2:
						saver.save(sess, new_dir + "/B_trained_model.ckpt")
					# print(max(pre_1, seq_acc_t), max(seq_2_acc_t, pre_2))
					if (seq_acc_t > pre_1 and  seq_acc_t + seq_2_acc_t > pre_1 + pre_2) \
							or (seq_2_acc_t > pre_2 and seq_acc_t + seq_2_acc_t > pre_1 + pre_2):
						max_accs[0] = seq_acc_t
						max_accs[1] = seq_2_acc_t
						saver.save(sess, new_dir + "/trained_model.ckpt")
						# np.save(new_dir + '/val_X.npy', valid_source)
						# np.save(new_dir + '/val_y.npy', valid_target)
					val_accs.append(seq_acc_t)
					val_2_accs.append(seq_2_acc_t)
					print(
						'Epoch {:>3}/{} Batch {:>4}/{} - Train Loss: {:>6.3f}  - V_Acc (IAS-A): {:>6.3f} {:>6.3f} (max) - V_Acc (IAS-B) {:>6.3f} {:>6.3f} (max)'
							.format(epoch_i,
						            epochs,
						            batch_i,
						            len(train_target) // batch_size,
						            loss,
						            seq_acc_t,
						            max(pre_1, seq_acc_t),
						            seq_2_acc_t,
						            max(seq_2_acc_t, pre_2)
						            ))
			train_loss.append(loss)
			test_loss.append(sum(val_loss) / len(val_loss))
			test_2_loss.append(sum(val_2_loss) / len(val_2_loss))
			step += 1
		saver = tf.train.Saver()
		# new_dir += '/' + ps
		# try:
		# 	os.mkdir(new_dir)
		# except:
		# 	pass
		# saver.save(sess, new_dir + "/trained_model.ckpt")
		np.save(new_dir + '/train_X.npy', train_source)
		np.save(new_dir + '/train_y.npy', train_target)
		# np.save(new_dir + '/train_preds', train_preds)
		np.save(new_dir + '/val_X.npy', valid_source)
		np.save(new_dir + '/val_y.npy', valid_target)
		np.save(new_dir + '/val_2_X.npy', valid_2_source)
		np.save(new_dir + '/val_2_y.npy', valid_2_target)
		# np.save(new_dir + 'val_preds.npy', valid_preds)
		print('Model Trained and Saved')
		np.save(new_dir + '/train_loss.npy', np.array(train_loss))
		np.save(new_dir + '/test_A_loss.npy', np.array(test_loss))
		np.save(new_dir + '/test_B_loss.npy', np.array(test_loss))
		np.save(new_dir + '/acc.npy', np.array(accs))
		disc_str = ''
		disc_str += str(train_loss[-1]) + '-' + str(np.min(train_loss)) + ' ' + str(test_loss[-1]) + '-' + str(
			np.min(test_loss)) + ' ' \
		            + str(np.max(acc))
		f = open(ps + '.txt', 'w')
		f.write(disc_str)
		f.close()
	return 1

def evaluate_reid(model_dir):
	# print('Print the Validation Loss and Rank-1 Accuracy for each testing bacth: ')
	global batch_size, dataset, manner
	X = np.load(model_dir + '/val_X.npy')
	y = np.load(model_dir + '/val_y.npy')
	print(X.shape, y.shape)
	if dataset == 'IAS':
		X_2 = np.load(model_dir + '/val_2_X.npy')
		y_2 = np.load(model_dir + '/val_2_y.npy')
	if dataset == 'BIWI':
		classes = [i for i in range(28)]
	elif dataset == 'KGBD':
		classes = [i for i in range(164)]
	elif dataset == 'IAS':
		classes = [i for i in range(11)]
	elif dataset == 'KinectReID':
		classes = [i for i in range(71)]
	elif dataset == 'KS20':
		classes = [i for i in range(20)]
	checkpoint = model_dir + "/trained_model.ckpt"
	if dataset == 'IAS':
		checkpoint = model_dir + "/A_trained_model.ckpt"
	loaded_graph = tf.get_default_graph()
	from sklearn.preprocessing import label_binarize
	from sklearn.metrics import roc_curve, auc, confusion_matrix
	nAUC = 0
	def cal_AUC(score_y, pred_y, ps, draw_pic=False):
		score_y = np.array(score_y)
		pred_y = label_binarize(np.array(pred_y), classes=classes)
		# Compute micro-average ROC curve and ROC area
		fpr, tpr, thresholds = roc_curve(pred_y.ravel(), score_y.ravel())
		roc_auc = auc(fpr, tpr)
		y_true = np.argmax(pred_y, axis=-1)
		y_pred = np.argmax(score_y, axis=-1)
		print('\n### Re-ID Confusion Matrix: ')
		print(confusion_matrix(y_true, y_pred))
		return roc_auc
		if draw_pic:
			fig = plt.figure()
			lw = 2
			plt.plot(fpr, tpr, color='darkorange',
			         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
			plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
			plt.xlim([0.0, 1.0])
			plt.ylim([0.0, 1.05])
			plt.xlabel('False Positive Rate')
			plt.ylabel('True Positive Rate')
			plt.title('Receiver operating characteristic: ' + ps)
			plt.legend(loc="lower right")
			fig.savefig('30 epoch ROC')
			plt.close()
	with tf.Session(graph=loaded_graph, config=config) as sess:
		loader = tf.train.import_meta_graph(checkpoint + '.meta')
		loader.restore(sess, checkpoint)
		X_input = loaded_graph.get_tensor_by_name('X_input:0')
		y_input = loaded_graph.get_tensor_by_name('y_input:0')
		lr = loaded_graph.get_tensor_by_name('learning_rate:0')
		pred = loaded_graph.get_tensor_by_name('add_1:0')
		cost = loaded_graph.get_tensor_by_name('new_train/Mean:0')
		accuracy = loaded_graph.get_tensor_by_name('new_train/Mean_1:0')
		correct_num = 0
		total_num = 0
		rank_acc = {}
		ys = []
		preds = []
		accs = []
		cnt = 0
		Rank_1 = 0
		if dataset == 'IAS':
			print('### Validation Results on IAS-A: ')
		if manner == 'sc':
			for batch_i, (y_batch, X_batch) in enumerate(
				get_new_train_batches(y, X, batch_size)):
				loss, acc, pre = sess.run([cost, accuracy, pred],
				                   {X_input: X_batch,
				                    y_input: y_batch,
				                    lr: learning_rate})
				ys.extend(y_batch.tolist())
				preds.extend(pre.tolist())
				accs.append(acc)
				cnt += 1
				for i in range(y_batch.shape[0]):
					for K in range(1, len(classes) + 1):
						if K not in rank_acc.keys():
							rank_acc[K] = 0
						t = np.argpartition(pre[i], -K)[-K:]
						if np.argmax(y_batch[i]) in t:
							rank_acc[K] += 1
				correct_num += acc * batch_size
				total_num += batch_size
				print(
					'Testing Bacth: {:>3} - Validation Loss: {:>6.3f} - Validation Rank-1 Accuracy {:>6.3f}'
						.format(cnt,
					            loss,
					            acc,
					            ))
			for K in rank_acc.keys():
				rank_acc[K] /= total_num
			total_acc = correct_num / total_num
			Rank_1 = total_acc
			# print('Rank-1 Accuracy: %f' % total_acc)
			nAUC = cal_AUC(score_y=preds,pred_y=ys, ps='nAUC')
		else:
			all_frame_preds = []
			for batch_i, (y_batch, X_batch) in enumerate(
				get_new_train_batches(y, X, batch_size)):
				loss, acc, pre = sess.run([cost, accuracy, pred],
				                   {X_input: X_batch,
				                    y_input: y_batch,
				                    lr: learning_rate})
				ys.extend(y_batch.tolist())
				preds.extend(pre.tolist())
				all_frame_preds.extend(pre)
				accs.append(acc)
				cnt += 1
				# for i in range(y_batch.shape[0]):
				# 	for K in range(1, len(classes) + 1):
				# 		if K not in rank_acc.keys():
				# 			rank_acc[K] = 0
				# 		t = np.argpartition(pre[i], -K)[-K:]
				# 		if np.argmax(y_batch[i]) in t:
				# 			rank_acc[K] += 1
				# correct_num += acc * batch_size
				# total_num += batch_size
				# print(
				# 	'Testing Bacth: {:>3} - Validation Loss: {:>6.3f} - Validation Rank-1 Accuracy {:>6.3f}'
				# 		.format(cnt,
				# 	            loss,
				# 	            acc,
				# 	            ))
			# for K in rank_acc.keys():
			# 	rank_acc[K] /= total_num
			sequence_pred_correct = 0
			sequence_num = 0
			sequence_preds = []
			sequence_ys = []
			rank_acc = {}
			for k in range(len(all_frame_preds) // time_steps):
				sequence_labels = np.argmax(y[k * time_steps: (k + 1) * time_steps], axis=1)
				# print(sequence_labels)
				if (sequence_labels == np.tile(sequence_labels[0], [sequence_labels.shape[0]])).all():
					frame_predictions = np.array(all_frame_preds[k * time_steps: (k + 1) * time_steps])
					sequence_pred = np.argmax(np.average(frame_predictions, axis=0))
					temp_pred = np.average(frame_predictions, axis=0)
					for K in range(1, len(classes) + 1):
						if K not in rank_acc.keys():
							rank_acc[K] = 0
						t = np.argpartition(temp_pred, -K)[-K:]
						if sequence_labels[0] in t:
							rank_acc[K] += 1
					if sequence_pred == sequence_labels[0]:
						sequence_pred_correct += 1
					sequence_num += 1
					sequence_ys.append(sequence_labels[0])
					aver = np.average(frame_predictions, axis=0)
					sequence_preds.append(aver)
			for K in rank_acc.keys():
				rank_acc[K] /= sequence_num
			seq_acc_t = sequence_pred_correct / sequence_num
			# total_acc = correct_num / total_num
			# print('(Frame) Rank-1 Accuracy: %f' % total_acc)
			Rank_1 = seq_acc_t
			sequence_ys = label_binarize(sequence_ys, classes=classes)
			# cal_AUC(score_y=preds,pred_y=ys, ps='nAUC')
			nAUC = cal_AUC(score_y=sequence_preds, pred_y=sequence_ys, ps='nAUC')
		print('### Rank-n Accuracy: ')
		print(rank_acc)
		print('### Rank-1 Accuracy: %f' % Rank_1)
		print('### nAUC: ' + str(nAUC))
	if dataset == 'IAS':
		checkpoint = model_dir + "/B_trained_model.ckpt"
		loaded_graph = tf.get_default_graph()
		nAUC = 0
		with tf.Session(graph=loaded_graph, config=config) as sess:
			loader = tf.train.import_meta_graph(checkpoint + '.meta')
			loader.restore(sess, checkpoint)
			X_input = loaded_graph.get_tensor_by_name('X_input:0')
			y_input = loaded_graph.get_tensor_by_name('y_input:0')
			lr = loaded_graph.get_tensor_by_name('learning_rate:0')
			pred = loaded_graph.get_tensor_by_name('add_1:0')
			cost = loaded_graph.get_tensor_by_name('new_train/Mean:0')
			accuracy = loaded_graph.get_tensor_by_name('new_train/Mean_1:0')
			correct_num = 0
			total_num = 0
			rank_acc = {}
			ys = []
			preds = []
			accs = []
			cnt = 0
			Rank_1 = 0
			print('### Validation Results on IAS-B: ')
			# IAS-B
			if manner == 'sc':
				correct_num = 0
				total_num = 0
				rank_acc = {}
				ys = []
				preds = []
				accs = []
				cnt = 0
				for batch_i, (y_batch, X_batch) in enumerate(
						get_new_train_batches(y_2, X_2, batch_size)):
					loss, acc, pre = sess.run([cost, accuracy, pred],
					                          {X_input: X_batch,
					                           y_input: y_batch,
					                           lr: learning_rate})
					ys.extend(y_batch.tolist())
					preds.extend(pre.tolist())
					accs.append(acc)
					cnt += 1
					for i in range(y_batch.shape[0]):
						for K in range(1, len(classes) + 1):
							if K not in rank_acc.keys():
								rank_acc[K] = 0
							t = np.argpartition(pre[i], -K)[-K:]
							if np.argmax(y_batch[i]) in t:
								rank_acc[K] += 1
					correct_num += acc * batch_size
					total_num += batch_size
					# print(
					# 	'Testing Bacth: {:>3} - Validation Loss: {:>6.3f} - Validation Rank-1 Accuracy {:>6.3f}'
					# 		.format(cnt,
					# 	            loss,
					# 	            acc,
					# 	            ))
				for K in rank_acc.keys():
					rank_acc[K] /= total_num
				total_acc = correct_num / total_num
				Rank_1 = total_acc
				# print('Rank-1 Accuracy: %f' % total_acc)
				nAUC = cal_AUC(score_y=preds, pred_y=ys, ps='nAUC')
			else:
				all_frame_preds = []
				for batch_i, (y_batch, X_batch) in enumerate(
						get_new_train_batches(y_2, X_2, batch_size)):
					loss, acc, pre = sess.run([cost, accuracy, pred],
					                          {X_input: X_batch,
					                           y_input: y_batch,
					                           lr: learning_rate})
					ys.extend(y_batch.tolist())
					preds.extend(pre.tolist())
					accs.append(acc)
					all_frame_preds.extend(pre)
					cnt += 1
					# for i in range(y_batch.shape[0]):
					# 	for K in range(1, len(classes) + 1):
					# 		if K not in rank_acc.keys():
					# 			rank_acc[K] = 0
					# 		t = np.argpartition(pre[i], -K)[-K:]
					# 		if np.argmax(y_batch[i]) in t:
					# 			rank_acc[K] += 1
					# # correct_num += acc * batch_size
					# total_num += batch_size
					# print(
					# 	'Testing Bacth: {:>3} - Validation Loss: {:>6.3f} - Validation Rank-1 Accuracy {:>6.3f}'
					# 		.format(cnt,
					# 	            loss,
					# 	            acc,
					# 	            ))
				# for K in rank_acc.keys():
				# 	rank_acc[K] /= total_num
				sequence_pred_correct = 0
				sequence_num = 0
				sequence_preds = []
				sequence_ys = []
				rank_acc = {}
				for k in range(len(all_frame_preds) // time_steps):
					sequence_labels = np.argmax(y_2[k * time_steps: (k + 1) * time_steps], axis=1)
					if (sequence_labels == np.tile(sequence_labels[0], [sequence_labels.shape[0]])).all():
						frame_predictions = np.array(all_frame_preds[k * time_steps: (k + 1) * time_steps])
						sequence_pred = np.argmax(np.average(frame_predictions, axis=0))
						temp_pred = np.average(frame_predictions, axis=0)
						for K in range(1, len(classes) + 1):
							if K not in rank_acc.keys():
								rank_acc[K] = 0
							t = np.argpartition(temp_pred, -K)[-K:]
							if sequence_labels[0] in t:
								rank_acc[K] += 1
						if sequence_pred == sequence_labels[0]:
							sequence_pred_correct += 1
						sequence_num += 1
						sequence_ys.append(sequence_labels[0])
						aver = np.average(frame_predictions, axis=0)
						sequence_preds.append(aver)
				for K in rank_acc.keys():
					rank_acc[K] /= sequence_num
				seq_acc_t = sequence_pred_correct / sequence_num
				Rank_1 = seq_acc_t
				# total_acc = correct_num / total_num
				# print('(Frame) Rank-1 Accuracy: %f' % total_acc)
				# print('Rank-1 Accuracy: %f' % seq_acc_t)
				sequence_ys = label_binarize(sequence_ys, classes=classes)
				# cal_AUC(score_y=preds, pred_y=ys, ps='nAUC')
				nAUC = cal_AUC(score_y=sequence_preds, pred_y=sequence_ys, ps='nAUC')
			print('### Rank-n Accuracy: ')
			print(rank_acc)
			print('### Rank-1 Accuracy: %f' % Rank_1)
			print('### nAUC: ' + str(nAUC))

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
def cal_AUC(scores, labels, ps):
	scores = np.array(scores)
	labels = np.array(labels)
	# pred_y = label_binarize(np.array(pred_y), classes=classes)
	# Compute micro-average ROC curve and ROC area
	fpr, tpr, thresholds = roc_curve(labels.ravel(), scores.ravel())
	roc_auc = auc(fpr, tpr)
	print(ps + ': ' + str(roc_auc))



if __name__ == '__main__':
	tf.app.run()
