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
# Keep all following default parameters unchanged to evaluate the best model
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
tf.app.flags.DEFINE_string('best_model', 'rev_rec', "rev_rec (Rev. Rec.) or rev_rec_plus(Rev. Rec. Plus)")
tf.app.flags.DEFINE_string('RN_dir', 'None', "Choose the model directory to evaluate")
FLAGS = tf.app.flags.FLAGS
config = tf.ConfigProto()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
temperature = 0.1
config.gpu_options.allow_growth = True
view = 'view_'
transfer = 'None'
Model = 'rev_rec'
IAS_test = 'A'
RN_dir = 'None'
def main(_):
	global attention, dataset, series_length, epochs, time_steps, gpu, manner, frames_ps, \
		temperature, Frozen, C_reid, temperature, train_flag, view, use_attention, transfer, Model, IAS_test
	attention, dataset, gpu, manner, length, Frozen, C_reid, temperature, train_flag, view_num, transfer, Model, RN_dir = FLAGS.attention, \
	                                                  FLAGS.dataset, FLAGS.gpu, FLAGS.manner, \
	                                                  FLAGS.length, FLAGS.frozen, FLAGS.c_reid, \
	                                                FLAGS.t, FLAGS.train_flag, FLAGS.view, FLAGS.transfer, FLAGS.best_model, FLAGS.RN_dir
	# Choose different datasets and models (Rev. Reconstruction or Rev. Reconstruction++) to evaluate
	if dataset not in ['BIWI', 'IAS', 'KGBD', 'KS20']:
		raise Exception('Dataset must be BIWI, IAS, KGBD, or KS20.')
	if Model not in ['prediction', 'sorting', 'rev_rec', 'rev_rec_plus']:
		raise Exception('Model must be rev_rec or rev_rec_plus')
	# Keep all following default parameters unchanged to evaluate the best model
	if attention not in ['BA', 'LA']:
		raise Exception('Attention must be BA or LA.')
	if manner not in ['sc', 'ap']:
		raise Exception('Training manner must be sc or ap.')
	if not gpu.isdigit() or int(gpu) < 0:
		raise Exception('GPU number must be a positive integer.')
	if length not in ['4', '6', '8', '10']:
		raise Exception('Length number must be 4, 6, 8 or 10.')
	if Frozen not in ['0', '1']:
		raise Exception('Frozen state must be 0 or 1.')
	if C_reid not in ['0', '1']:
		raise Exception('C_reid state must be 0 or 1.')
	if train_flag not in ['0', '1', '2']:
		raise Exception('Train_flag must be 0, 1 or 2 (Only evaluation).')
	if view_num not in ['0', '1', '2', '3', '4', 'None']:
		raise Exception('View_num must be 0, 1, 2, 3, 4 or None')
	if transfer not in ['BIWI', 'IAS', 'KGBD', 'KS20', 'None']:
		raise Exception('Transfer dataset must be BIWI, IAS, KGBD, KS20 or None')
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

	time_steps = int(length)
	temperature = float(temperature)
	frames_ps = dataset + '/' + str(time_steps) + '/'
	epochs = 400

	if dataset != 'KS20':
		view = ''

	if dataset == 'KGBD':
		temperature = 0.5
	else:
		temperature = 0.1
	# 	Rev. Reconstruction
	if RN_dir == 'None':
		print(
			' ## Dataset: %s\n ## Attention: %s\n ## Re-ID Manner: %s\n ## Sequence Length: %s\n ## Tempearture: %s\n ## Pretext Task: %s\n ## GPU: %s\n' %
			(dataset, attention, manner, str(time_steps), str(temperature), Model, str(gpu)))
		if Model == 'rev_rec':
			if dataset == 'IAS':
				IAS_test = 'A'
				evaluate_reid('./Models/CAGEs_RN_models/IAS-A_' + attention + '_RN_' + manner + '_' + str(time_steps)
			              + '_' + str(temperature) + '_' + Frozen + view + 'pre_' + Model)
				IAS_test = 'B'
				evaluate_reid(
					'./Models/CAGEs_RN_models/IAS-B_' + attention + '_RN_' + manner + '_' + str(time_steps)
					+ '_' + str(temperature) + '_' + Frozen + view + 'pre_' + Model)
			else:
				evaluate_reid(
					'./Models/CAGEs_RN_models/' + dataset + '_' + attention + '_RN_' + manner + '_' + str(time_steps)
					+ '_' + str(temperature) + '_' + Frozen + view + 'pre_' + Model)
		# Rev. Reconstruction ++
		elif Model == 'rev_rec_plus':
			if dataset == 'IAS':
				try:
					IAS_test = 'A'
					evaluate_reid('./Models/CAGEs_RN_models/IAS-A' + '_RN_' + manner + '_' + str(time_steps)
				              + '_' + str(temperature) + '_' + Frozen + view + 'pre_' + Model)
					IAS_test = 'B'
					evaluate_reid(
						'./Models/CAGEs_RN_models/IAS-B' + '_RN_' + manner + '_' + str(time_steps)
						+ '_' + str(temperature) + '_' + Frozen + view + 'pre_' + Model)
				except:
					IAS_test = 'A'
					evaluate_reid('./Models/CAGEs_RN_models/IAS' + '_BA_RN_' + manner + '_' + str(time_steps)
					              + '_' + str(temperature) + '_' + Frozen + view + 'pre_' + Model)
					IAS_test = 'B'
					evaluate_reid(
						'./Models/CAGEs_RN_models/IAS' + '_BA_RN_' + manner + '_' + str(time_steps)
						+ '_' + str(temperature) + '_' + Frozen + view + 'pre_' + Model)
			else:
				evaluate_reid(
					'./Models/CAGEs_RN_models/' + dataset + '_RN_' + manner + '_' + str(time_steps)
					+ '_' + str(temperature) + '_' + Frozen + view+ 'pre_' + Model)
		else:
			evaluate_reid(
				'./Models/CAGEs_RN_models/' + dataset + '_RN_' + manner + '_' + str(time_steps)
				+ '_' + str(temperature) + '_' + Frozen + view + 'pre_' + Model)
	else:
		try:
			settings = RN_dir.split('_')
			dataset, attention, manner, time_steps, temperature = settings[0], settings[1], settings[3], int(settings[4]), float(settings[5])
			settings = RN_dir.split('pre_')
			Model = settings[1]
			print(' ## Dataset: %s\n ## Attention: %s\n ## Re-ID Manner: %s\n ## Sequence Length: %s\n ## Tempearture: %s\n ## Pretext Task: %s\n' %
			      (dataset, attention, manner, str(time_steps), str(temperature), Model))
			evaluate_reid('./Models/CAGEs_RN_models/' + RN_dir)
		except:
			print('Running failed. Please check out your parameters.')



def get_new_train_batches(targets, sources, batch_size):
	if len(targets) < batch_size:
		yield targets, sources
	else:
		for batch_i in range(0, len(sources) // batch_size):
			start_i = batch_i * batch_size
			sources_batch = sources[start_i:start_i + batch_size]
			targets_batch = targets[start_i:start_i + batch_size]
			yield targets_batch, sources_batch

def evaluate_reid(model_dir):
	# print('Print the Validation Loss and Rank-1 Accuracy for each testing bacth: ')
	global batch_size, dataset, manner, IAS_test
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
		if (dataset == 'IAS' and IAS_test == 'A') or dataset != 'IAS':
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
		if dataset == 'IAS' and IAS_test == 'B':
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


if __name__ == '__main__':
	tf.app.run()
