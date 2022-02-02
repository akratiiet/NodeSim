import numpy as np
from sklearn import linear_model as sk_lm
from sklearn import metrics as sk_ms
from sklearn import exceptions as sk_exc
from sklearn.metrics import roc_auc_score
import math
from sklearn.neural_network import MLPClassifier
from gensim.models import KeyedVectors


def hadamard_op_comm(arg1, arg2,arg3):
	'''
	Generate features for a Node-pair using their vector representation and community information.
	'''
	length = len(arg1)
	feat = []
	for i in range(length):
		feat.append(arg1[i]*arg2[i])
	feat.append(int(arg3))
	return feat


def edge_embedding_comm(node_embedding, edges):
	'''
	Call hadamard_op_comm function to generate feature representation for all node pairs.
	'''
	edge_embs = []

	for (n1, n2,n3) in edges:
		n1_emb = node_embedding[n1]
		n2_emb = node_embedding[n2]
		edge_embs.append(hadamard_op_comm(n1_emb, n2_emb,n3))
	return edge_embs	

def load_data_comm(filename):
	'''
	Read train and test data
	'''
	edges = []
	with open(filename, 'r') as f:
		for line in f:
			tmp = line.strip().split()
			edges.append((tmp[0], tmp[1], tmp[2]))
	return edges


def link_prediction():
	emb = KeyedVectors.load_word2vec_format("Output/sample.emb",  binary=False)
	#x_train and y_train are the files containing training data
	x_train = edge_embedding_comm(emb, load_data_comm("Input/sample_trainx.txt")) 
	y_train = np.loadtxt("Input/sample_trainy.txt")
	
	log_reg = sk_lm.LogisticRegression(solver='liblinear')
	log_reg.fit(np.array(x_train), np.array(y_train))
	
	#x_test and y_test are the files containing training data
	x_test = edge_embedding_comm(emb, load_data_comm("Input/sample_testx.txt"))
	y_test = np.loadtxt("Input/sample_testy.txt")
	y_pred = log_reg.predict(x_test)
	#check accuracy
	accuracy = sk_ms.accuracy_score(y_test, y_pred)
	roc_auc = roc_auc_score(y_test, y_pred)

	print('Total Accuracy: ', accuracy)
	print('Total AUC: ', roc_auc)
	

link_prediction()

