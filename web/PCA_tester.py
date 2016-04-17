#!/usr/bin/env python
# coding: utf-8 

# Rosetta Consulting customer segmentation pipeline
# 1.20.2016
# Peter Niessen

'''
Segmentr: a customer segmentation toolkit
'''


import numpy as np
from sklearn.decomposition import PCA
import time
import flask
import pandas as pd
import os
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for, send_file, copy_current_request_context
from werkzeug import secure_filename
from collections import Counter, OrderedDict
import pprint
from numpy import loadtxt, eye, asarray, dot, sum, diag
from numpy.linalg import svd
from joblib import Parallel, delayed
import multiprocessing
import random
import xlsxwriter
import sys, getopt
import itertools
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, DBSCAN, AffinityPropagation, MeanShift, KMeans, Birch
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import uuid
import shutil
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
#import threading
#from flask.ext.socketio import SocketIO, emit
#from threading import Thread, Event
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from pymongo import MongoClient
import cPickle as pickle
from bson.binary import Binary
import gridfs
import StringIO

#def init():
# default options
interactive_mode = False
xls = False
web_mode = False
filename = 'test_raw_data_q_v1.csv' # (user x question) .csv with question labels in first row
weights_filename = 'test_raw_data_q_v1_weights.csv' # (user x question) .csv with question labels in first row
#filename = 'research_files/labelled_data/GP_Content_Seg_Input_File_092115_Full_Data.csv' # labeled data
grid_search = False
visual_mode = True

# now check for command line arguments + options
argv = sys.argv[1:]
print "Command line arguments:", str(argv)

try:
	opts, args = getopt.getopt(argv,"hiwxlsf:gv",)
except getopt.GetoptError:
	print 'PCA_tester.py -i (interactive_mode), -w (web_mode), -xls (excel file export), -f: infile_name.csv -v (no viusals)'
	sys.exit()

for opt, arg in opts:
	if opt == '-h':
		print 'PCA_tester.py -i (interactive_mode), -w (web_mode), -xls (excel file export), -f: infile_name.csv -v (no visuals)'
		sys.exit()
	elif opt in ('-i','-I'):
		interactive_mode = True
	elif opt in ('-w', '-W'):
		web_mode = True
	elif opt in ('-xls', '-xl','-x'):
		xls = True
	elif opt in ('-g', '-G',):
		grid_search = True
	elif opt in ('-v', '-V',):
		visual_mode = False
	elif opt == '-f':
		# http://stackoverflow.com/questions/5899497/checking-file-extension
		if arg.lower().endswith(('.csv')):
			filename = arg#[1:]
			print "infile name:", filename
		else:
			print "wrong file type"
			sys.exit()

app = Flask(__name__)
# app.config['SECRET_KEY'] = 'secret!'
# app.config['DEBUG'] = True
# socketio = SocketIO(app)
start_time = time.time()



# load db
client   = MongoClient()
db  = client.dsbc.segmentr
db.remove()
grid_db  = MongoClient().gridfs_segmentr
gridfs = gridfs.GridFS(grid_db)

basedir_for_upload = os.path.abspath(os.path.dirname(__file__))
basedir = os.path.abspath(os.path.dirname(__file__)) + "/"
print basedir_for_upload
print 'basedir:', basedir

from logging import Formatter, FileHandler
handler = FileHandler(os.path.join(basedir_for_upload, 'log.txt'), encoding='utf8')
handler.setFormatter(
    Formatter("[%(asctime)s] %(levelname)-8s %(message)s", "%Y-%m-%d %H:%M:%S")
)
app.logger.addHandler(handler)

# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'csv', 'xls', 'xlsx'])

# return basedir, basedir_for_upload, handler, xls, interactive_mode, web_mode, filename, grid_search, app, start_time
def new_session (session_id):
	z = func_name(); print "--------in function: ", z, " -------------"

	print 'creating new db entry with session_id: ', session_id

	db.insert_one({'session_id': session_id})
	return
    
def load_db(session_id):
	#z = func_name(); print "--------in function: ", z, " -------------"
	
	results = db.find_one({'session_id': session_id})
	
	X_file_id = results['X']
	X_rebucketed_file_id = results['X_rebucketed']
	X_rebucketed_df_file_id = results['X_rebucketed_df']	
	results_dict_file_id = results['results_dict']
	question_dict_file_id = results['question_dict']
	names = results['names']

	X = pickle.loads(gridfs.get(X_file_id).read())
	X_rebucketed = pickle.loads(gridfs.get(X_rebucketed_file_id).read())
	X_rebucketed_df = pickle.loads(gridfs.get(X_rebucketed_df_file_id).read())
	results_dict = pickle.loads(gridfs.get(results_dict_file_id ).read())
	question_dict = pickle.loads(gridfs.get(question_dict_file_id).read())

	print '---load_db scorecard-------'
	print 'Session_id:', session_id
	print 'All documents: ', list(results)
	print 'X.shape: ', X.shape
	print 'X_rebucketed.shape: ', X_rebucketed.shape
	print 'X_rebucketed_df.shape: ', X_rebucketed_df.shape
	print 'question_dict: ', len(question_dict)
	print 'results_dict: ', len(results_dict)
	print 'names: ', len(names)

	# X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names = load_db(session_id)
	return X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names

def save_db(session_id, X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names):
	#z = func_name(); print "--------in function: ", z, " -------------"

	X_pickle = Binary(pickle.dumps(X, protocol=2), subtype=128)
	X_file_id = gridfs.put(X_pickle)
	db.update_one({'session_id': session_id},{"$set":{'X': X_file_id}},upsert=True)

	X_rebucketed_pickle = Binary(pickle.dumps(X_rebucketed, protocol=2), subtype=128 )
	X_rebucketed_file_id = gridfs.put(X_rebucketed_pickle)
	db.update_one({'session_id': session_id},{"$set":{'X_rebucketed': X_rebucketed_file_id}},upsert=True) 

	X_rebucketed_df_pickle = Binary(pickle.dumps(X_rebucketed_df, protocol=2), subtype=128 )
	X_rebucketed_df_file_id = gridfs.put( X_rebucketed_df_pickle )
	db.update_one({'session_id': session_id},{"$set":{'X_rebucketed_df': X_rebucketed_df_file_id}},upsert=True)

	results_dict_pickle = Binary(pickle.dumps(results_dict, protocol=2), subtype=128 )
	results_dict_pickle_file_id = gridfs.put( results_dict_pickle )
	db.update_one({'session_id': session_id},{"$set":{'results_dict': results_dict_pickle_file_id}},upsert=True)

	question_dict_pickle = Binary(pickle.dumps(question_dict, protocol=2), subtype=128 )
	question_dict_pickle_file_id = gridfs.put( question_dict_pickle )
	db.update_one({'session_id': session_id},{"$set":{'question_dict': question_dict_pickle_file_id}},upsert=True)

	db.update_one({'session_id': session_id},{"$set":{'names':names}},upsert=True)

	results = db.find_one({'session_id': session_id})

	print '---save_db scorecard-------'
	print 'Session_id:', session_id
	print 'All documents: ', list(results)
	print 'X.shape: ', X.shape
	print 'X_rebucketed.shape: ', X_rebucketed.shape
	print 'X_rebucketed_df.shape: ', X_rebucketed_df.shape
	print 'question_dict: ', len(question_dict)
	print 'results_dict: ', len(results_dict)
	print 'names: ', len(names)

	#save_data(session_id, X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names)
	return

def clean_text(text):
	'''
	Removes non-UTF8 and other stop characters from text
	'''
	z = func_name(); print "--------in function: ", z, " -------------"
	
	#print 'now cleaning:', text
	
	# stop characters (could this be a dict?)
	a = chr(146) # single quotation mark
	
	# clean up text
	text = [t.replace("'","").replace("’","").replace(a,"").replace(" ","_") for t in text]
	text_2 = []
	for t in text:
	    try:
	        t2 = t.encode('utf-8').decode('utf-8','ignore').encode("utf-8")
	    except:
	        t2 = ''
	    text_2.append(t2)
	
	text_3 = ''.join(text_2)

	#print 'cleaned text: ', text_3

	return text_3

def convert_data(filename):
	'''
	Converts dataset with responses values and labels to separate response_dict and X (=response value matrix)
	'''

	z = func_name(); print "--------in function: ", z, " -------------"

	X = pd.DataFrame.from_csv(filename) # handles 'irregular' chars better than direct to matrx
	question_names = list(X.columns)
	#question_names = [q.replace("'","").replace("’","").replace(a,"").replace(" ","_") for q in question_names]
	question_names = [clean_text(q) for q in question_names]
	#print question_names
	#print '!! names type:', type(question_names)
	#print '!! names (list) type:', type(list(question_names))
	question_text = X.ix[0,:] # assumes first row is full question text
	X = X.ix[1:,] # assumes first row is full question text
	X = pd.DataFrame.as_matrix(X) #(row, col)

	# clean up, remove bad characters
	null_counter = 0
	for col in range(X.shape[0]):
		for index in range(X.shape[1]):
			val = X[col][index]
			if val == '#NULL!' or val == ' ':
				val = 'n/a'
				null_counter += 1
				a = chr(146) # single quotation mark
				X[col][index] = val.replace("'","").replace("’","").replace(a,"").replace(" ","_")

	# create response_dict
	response_dict = {}
	for index, question in enumerate(question_names):
	    response_dict[question] = {}
	    response_dict[question]['verbatim'] = sorted(list(set(X[:,index])))
	    for item in response_dict[question]['verbatim']:
	        response_dict[question][item] = response_dict[question]['verbatim'].index(item) + 1
	        response_dict[question][response_dict[question][item]] = item

	# convert X to numeric only
	for col in range(X.shape[0]):
	    for index in range(X.shape[1]):
	        val = X[col][index]
	        X[col][index] = int(response_dict[question_names[index]][val])
	
	X = X.astype(float)
	
	return X, question_names, question_text, response_dict

# Step 1: load datasets, namess
# load datafile

def get_PCA(filename, weights_filename, session_id):
	'''
	Uses sklearn.decomposition for principal componets analysis of survey response data
	'''
	z = func_name(); print "--------in function: ", z, " -------------"
	
	# reboot question and results containers
	#global question_dict
	question_dict = {}

	#global results_dict
	results_dict = {}

	#global X
	X = np.random.random((10,10))

	#global X_rebucketed
	#global X_rebucketed_df
	X_rebucketed = np.random.random((10,10))
	X_rebucketed_df = pd.DataFrame(X_rebucketed)

	#global names 
	names = '' 

	# initialize db
	# can we make this conditional, e.g. if session_id is already in db then pass?

	try:
		results = db.find_one({'session_id': session_id})
		filename = results['data_filename']
		weights_filename = results['weights_filename']
		print 'using uploaded data - loading data_file and weights_file from db'

	except:
		new_session (session_id)
		db.update_one({'session_id': session_id},{"$set":{'data_filename':filename}},upsert=True)
		db.update_one({'session_id': session_id},{"$set":{'weights_filename':weights_filename}},upsert=True)
		print 'initializing session_id, using default data_file and weights_file'
	
	save_db(session_id, X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names)

	# clean up /plots directory
	# d = 'static/plots/'
	# if os.path.exists(d):
	# 	shutil.rmtree(d); os.makedirs(d)

	X, names, question_text, response_dict = convert_data(filename)

	#names = list(np.genfromtxt(basedir+filename, delimiter=',', names=True).dtype.names)
	#question_text = list(np.genfromtxt(basedir+filename, delimiter=',', skip_header=1, names=True).dtype.names)
	
	for index, question_name in enumerate(names):
		question_dict[question_name] = {}
		print index, question_text[index]
		#question_dict[question_name]['question_text'] = question_text[index][:50].strip("'").encode('utf-8').decode('utf-8','ignore').encode("utf-8")
		question_dict[question_name]['question_text'] = clean_text(question_text[index][:50])

		'''
		# this is ugly; put in a function?
		a = chr(146) # single quotation mark
		question = [q.replace("'","").replace("’","").replace(a,"").replace(" ","_") for q in question_text[index][:50]]
		question_2 = []
		for q in question:
		    try:
		        q2 = q.encode('utf-8').decode('utf-8','ignore').encode("utf-8")
		    except:
		        q2 = ''
		    question_2.append(q2)
		question_dict[question_name]['question_text'] = ''.join(question_2)
		'''
				
		question_dict[question_name]['dimension'] = "n/a"
		question_dict[question_name]['run_tracker'] = []
	print "question_dict initialized with ", len(question_dict.keys()), " questions"

	print names

	# X = np.genfromtxt(basedir+filename, delimiter=',', skip_header=2)

	# create weights, save under standardized name with unique session_id

	X_weights = np.asmatrix(np.genfromtxt(basedir + weights_filename, delimiter=',', skip_header=1))
	print 'X_weights.shape', X_weights.shape
	filename = basedir + 'static/weights_file/X_weights_' + session_id +'.csv'
	np.savetxt(filename, X_weights.T, delimiter=',') 
	print 'weights_filename:', filename

	print filename, "loaded"
	print weights_filename, "loaded"
	print X.shape
	print X[0] # print one row
	#print X.dtype.names

	# transpose to get features on X axis and respondents on Y ax
	#X = X.T
	#print X.shape

	# sameple array for testing
	#X = np.array([[-1, -1, 2, 5], [-2, -1, 5, 7], [-3, -2, 6, 1], [1, 1, 4, 5], [2, 1, 5, 9], [3, 2, 6, 5]])

	# step 2: Principal Components Analysis

	#pca = PCA(n_components=n_components)
	#pca.fit(X)

	#print "number of components:" , pca.n_components_
	#print("Runtime: %s seconds ---" % (time.time() - start_time))
	#print pca.components_.shape

	# transpose to get compontents on X axis and questions on Y

	#global factor_matrix

	factor_matrix = homebrew_factor_matrix(X, X_weights) 
	print "factor matrix d_type:", type(factor_matrix)
	
	print "factor_matrix shape:", factor_matrix.shape
	print "sample row: ", factor_matrix[0]
	print "max absolute value:", np.absolute(factor_matrix)[0].max()

	save_db(session_id, X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names)

	factor_matrix_pickle = Binary(pickle.dumps(factor_matrix, protocol=2), subtype=128)
	factor_matrix_file_id = gridfs.put(factor_matrix_pickle)
	db.update_one({'session_id': session_id},{"$set":{'factor_matrix': factor_matrix_file_id}},upsert=True)


	return factor_matrix, names, X, question_dict, results_dict

def homebrew_factor_matrix(X_unweighted, X_weights):
	'''
	Inputs raw survey data + weights, yields rotated factor matrix using varimax method
	'''
	z = func_name(); print "--------in function: ", z, " -------------"

	# input raw data, center, apply weights
	# (raw_data(i) - raw_data(mean)) * sqrt(weight)
	# http://stats.stackexchange.com/questions/113485/weighted-principal-components-analysis

	# unweighted_filename = '/Users/pniessen/Rosetta_Desktop/Segmentation_2-point-0/sample_case_work/GoPro/gopro_raw_data_v1.csv'
	# weights_filename = '/Users/pniessen/Rosetta_Desktop/Segmentation_2-point-0/segmentr/web/gopro_weights.csv'

	#X_names = list(np.genfromtxt(unweighted_filename, delimiter=',', names=True).dtype.names)
	#X_unweighted = np.asmatrix(np.genfromtxt(unweighted_filename, delimiter=',', skip_header=1))
	#X_weights = np.asmatrix(np.genfromtxt(weights_filename, delimiter=',', skip_header=1))

	X_unweighted_mean = (np.mean(X_unweighted, axis = 0))
	X_unweighted_centered = X_unweighted - X_unweighted_mean
	X_weighted = np.multiply(X_unweighted_centered.T,np.sqrt(X_weights))
	X = X_weighted.T

	# following http://sebastianraschka.com/Articles/2014_pca_step_by_step.html
	# correlation matrix
	X_cor_mat = np.corrcoef(X.T)

	# get eigenvectors and eigenvalues for the from the correlation matrix
	eig_val_cor, eig_vec_cor = np.linalg.eig(X_cor_mat)

	# initialize empty loading_factor_matrix of (0 x length of eigenvector)
	eig_vec_cors_length = eig_vec_cor[:,1].shape[0]
	loading_factors_matrix = np.asmatrix(np.zeros((0,eig_vec_cors_length))) # matrix shape must be tupple

	# filter for eigvec > 1.0, generate loading factor, add to loading_factors_matrix
	# eigenvec < 1.0 contribute less than original variables
	for i in range(len(eig_val_cor)):
	    one_eigenvector = eig_vec_cor[:,i].reshape(1,eig_vec_cors_length).T # unit vector, direction
	    one_eigenvalue = eig_val_cor[i] # scale
	    
	    if one_eigenvalue >=1:
	        # http://stats.stackexchange.com/questions/143905/loadings-vs-eigenvectors-in-pca-when-to-use-one-or-another
	        loading_factor = one_eigenvector * np.sqrt(one_eigenvalue)
	        loading_factors_matrix = np.r_[loading_factors_matrix,loading_factor.T]

	print loading_factors_matrix.shape
	print loading_factors_matrix.T[:,1].sum()
	print 'loading factors matrix sum:', abs(loading_factors_matrix.T).sum()

	loading_factors_matrix_rotated = varimax(loading_factors_matrix.T) # <class 'numpy.matrixlib.defmatrix.matrix'>
	print 'loading factors matrx rotated sum: ', abs(loading_factors_matrix_rotated).sum()
	loading_factors_matrix_rotated = np.asarray(loading_factors_matrix_rotated) # backwards compatability with pca.components_ output
	print loading_factors_matrix_rotated.shape

	'''
	tested vs SPSS output using gopro data:
	for explanation of SPSS output
	see http://www.ats.ucla.edu/stat/spss/output/principal_components.htm
	----------------
	SPSS loading factor matrix files:
	spss_unrotated: (148, 32) 468.803663037
	spss_rotated: (148, 32) sum: 381.386018769
	----------------
	Homebrew loading factor matrix files:
	homebrew_unrotated: (148, 32) sum: 468.845148151
	homebrew_rotated: (148, 32) sum: 380.83076567
	-----pairwise comparison-------------
	unrotated loading_factors_matrix are similar using np.isclose @ atol = 1e-04
	rotated loading_factors_matrix are similar using np.isclose @ atol = 1e-04
	'''
	return loading_factors_matrix_rotated
	        
def varimax(Phi, gamma = 1.0, q = 100, tol = 1e-10):
	'''
	Orthogonal factor matrix rotation - this helps to filter out 'midrange' factors
	# http://stackoverflow.com/questions/17628589/perform-varimax-rotation-in-python-using-numpy
    # gamma = 1.0: varimax
    # gamma = 0.0: quartimax  (https://github.com/rossfadely/consomme/blob/master/consomme/rotate_factor.py)
    '''

	z = func_name(); print "--------in function: ", z, " -------------"

	p,k = Phi.shape
	R = eye(k) # identity matrix
	d=0

	for i in xrange(q):
		
		d_old = d
		Lambda = dot(Phi, R)
		u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
		R = dot(u,vh)
		d = sum(s)

		delta = d / d_old
		#print i, delta
		
		if d_old !=0 and delta <(1 + tol):
			print 'varimax covergence in:', i, 'iterations'
			break

	result = dot(Phi, R)
	
	print 'result: ', abs(result).sum()

	return result

def top_n_factors (factor_matrix, top_n, question_dict, names, session_id):
	'''
	Uses PCA to group survey questions by PCA factors: 
		(a) rank order questions by largest PCA factor (factor_matrix) 
		(b) take largest (top_n)
	'''
	z = func_name(); print "--------in function: ", z, " -------------"
	
	X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names = load_db(session_id)

	#print "length of question_dict: ", len(question_dict)
	#print "question_dict keys: ", question_dict.keys()
	#print "length of question names list: ", len(names)
	#print "question names list: ", names
	#global question_dict
	
	# factor_matrix, n, question_dict = factor_matrix, top_n, question_dict
	# Step 3: identify largest PC factor for each question
	# add two rows of zeros for top 2 factors
	# np.append(a, z, axis=1)
	# stackoverflow.com/questions/8486294/how-to-add-an-extra-column-to-an-numpy-array

	# add extra cols for row (=question) index, #1 factor, #2 factor, best_rh
	factor_matrix_size = factor_matrix.shape # = C x R
	#print factor_matrix_size
	num_rows = factor_matrix_size[0]
	num_cols = factor_matrix_size[1]
	print "number of rows: " , num_rows
	print "nunber of columns: ", num_cols
	question_number = num_cols
	best_factor = num_cols + 1
	second_best_factor = num_cols + 2
	rh = num_cols + 3

	zero_matrix = np.zeros((num_rows,4), dtype=np.int64)
	print zero_matrix.shape

	factor_matrix = np.append(factor_matrix, zero_matrix, axis=1)
	print factor_matrix.shape
	#print "sample row: ", factor_matrix[0]
	#print "sample of last four cells in row", factor_matrix[0][num_cols:]

	for row in range(num_rows):
		#row = 0
		one_question_vector = np.absolute(factor_matrix[row])
		#print "one_question_vector:", len(one_question_vector), one_question_vector
		#print "max_value: ", one_question_vector.max()
		#print "max value is with component number", one_question_vector.tolist().index(one_question_vector.max())+1

		# find top n factors for each question
		n=2
		top_n_factors = one_question_vector.argsort()[-n:][::-1] #[::-1] to sort H-L, returns index (=component number) of n highest factors
		#print "top 2 factors:", top_n_factors + 1 # add +1 because top_n_factors is index (first value = 0)	
		top_n_factor_values = one_question_vector[top_n_factors] # use index to find actual values
		#print "top 2 factor values: ", top_n_factor_values

		# add back to factor_matrix
		factor_matrix[row][question_number] = row # index for question
		factor_matrix[row][best_factor], factor_matrix[row][second_best_factor] = top_n_factors
		question_dict[names[row]]['first_factor'], question_dict[names[row]]['second_factor'] = top_n_factors
		question_dict[names[row]]['first_factor_value'], question_dict[names[row]]['second_factor_value'] = top_n_factor_values

		#print "after: ", factor_matrix[row]

		# check to see that successfully inserted back into factor_matrix
		if not (factor_matrix[row][best_factor:second_best_factor+1]  == top_n_factors).all():
			print "error in row", row
			print factor_matrix[row][best_factor:second_best_factor+1]
			print top_n_factors

	print "found top 2 factors for", num_rows, " questions (=rows)"
	print "question_dict keys:", question_dict[question_dict.keys()[0]].keys()

	print("Runtime: %s seconds ---" % (time.time() - start_time))
	
	save_db(session_id, X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names)
	return factor_matrix, num_rows, num_cols, question_number, rh, best_factor, second_best_factor, question_dict

def rebucket(factor_matrix, names, X, rh, question_dict, session_id):
	'''
	Tries different bucketing techniques: [(2,3,2),(3,2,2), (2,2,3), (3,1,3), (1,3,3), (1,2,4)]
	Chooses technique that best proxies normal distribution 
	"Rosetta Heuristic" is (approxminately) a ranking of how far from standard deviation - lower is better
	Rosetta_heuristic = np.absolute((top_bucket - .26)) + np.absolute((bottom_bucket - .26)) + np.absolute((middle_bucket - .48)) + np.absolute((top_bucket - bottom_bucket)) * 100
	'''
	z = func_name(); print "--------in function: ", z, " -------------"
	X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names = load_db(session_id)
	print 'X.shape', X.shape
	
	print "length of question_dict: ", len(question_dict)
	#print "question_dict keys: ", question_dict.keys()
	#print "names: ", names
	#global question_dict

	# factor_matrix, names, X, rh, question_dict = factor_matrix, names, X, rh, question_dict
	# Step 4: calulate Rosetta Heuristic, identify optimal bucketing scheme, rebucket response matrix
	# shape = row x columns
	# get a row of X (= one set of responses to all questions from a single respondee)
	# print X[0]
	# get a column of X (= one set of responses from all respondeees to a single question)
	# print X[:,0]
	best_schemes = []

	for column in range(X.shape[1]):
		one_column = X[:,column]

		from collections import Counter
		responses = [1,2,3,4,5,6,7]
		
		# check for alternative response ranges (binary, 1-5, etc)
		responses_2 = list(set(one_column.tolist()))
		# print 'responses_2:', responses_2
		
		if len(responses_2) == 2:
			print 'Alert: question {0} has {1} buckets'.format(names[column],len(responses_2))
			print responses_2
			
			question_dict[names[column]]['best_rh'] = 0
			question_dict[names[column]]['bucket_scheme'] = (8,8,8) # dichotomous
			best_schemes.append((8,8,8))
			factor_matrix[column][rh] = 0


		elif len(responses_2) > len(responses):# + 1:
			print 'Alert: question {0} has {1} buckets'.format(names[column],len(responses_2))
			print responses_2

			question_dict[names[column]]['best_rh'] = 0
			question_dict[names[column]]['bucket_scheme'] = (9,9,9) # continuous
			best_schemes.append((9,9,9))
			factor_matrix[column][rh] = 0

			# test for >7, then insert 'pass' into best_scheme list, set X_rebucketed[:,col] = X[:,col]
		
		else:
			response_counts = [(Counter(one_column))[r] for r in responses]
			#print response_counts
			response_shares = [float(response_counts[x]) / sum(response_counts) for x in range(len(response_counts))]
			#print response_shares

			# Mapping heuristic: XYZ: (% top X boxes - .26) + (% bottom Z boxes - .26) + (% middle Y boxes - .48) + (% top X boxes - + % bottom Z boxes)
			bucket_schemes = [(2,3,2),(3,2,2), (2,2,3), (3,1,3), (1,3,3), (1,2,4)]
			rosetta_heuristics = []

			# next step: for col in colums, or wrap as function then use map() to apply to each column?
			for bucket_scheme in bucket_schemes:
			    top_bucket = sum(response_shares[0:bucket_scheme[0]])
			    #print top_bucket
			    middle_bucket = sum(response_shares[bucket_scheme[0]:bucket_scheme[0]+ bucket_scheme[1]])
			    #print middle_bucket
			    bottom_bucket = sum(response_shares[bucket_scheme[0]+ bucket_scheme[1]:bucket_scheme[0]+ bucket_scheme[1] + bucket_scheme[2]])
			    #bottom_bucket = 1 - top_bucket - middle_bucket
			    #print bottom_bucket
			    #if not top_bucket + middle_bucket + bottom_bucket == 1:
			    #	print "bucket weights wrong!"
			    #	print float(top_bucket + middle_bucket + bottom_bucket)
			    rosetta_heuristic = np.absolute((top_bucket - .26)) + np.absolute((bottom_bucket - .26)) + np.absolute((middle_bucket - .48)) + np.absolute((top_bucket - bottom_bucket)) * 100
			    #print "RH:", rosetta_heuristic
			    rosetta_heuristics.append(rosetta_heuristic)
			
			best_rh = min(rosetta_heuristics)
			best_scheme = rosetta_heuristics.index(best_rh)
			question_dict[names[column]]['best_rh'] = best_rh
			question_dict[names[column]]['bucket_scheme'] = bucket_schemes[best_scheme]
			best_schemes.append(bucket_schemes[best_scheme])
			factor_matrix[column][rh] = best_rh
			#print "after after: ", factor_matrix[column]
			#print column, " :", bucket_schemes[best_scheme], "Best RH:", min(rosetta_heuristics)
			#print best_schemes[:10]

	# now rebucket response matrix
	# could this all be redone using map()?

	X_rebucketed = np.zeros((X.shape), dtype=np.int64)
	print 'X_rebucketed.shape: ', X_rebucketed.shape
	for col in range(X_rebucketed.shape[1]):
		
		if best_schemes[int(col)] == (9,9,9): # skip continuous
			X_rebucketed[:,col] = X[:,col]

		elif best_schemes[int(col)] == (8,8,8): # dichotomous
			vars = list(set(X[:,col]))
			X_rebucketed[:,col] = [1 if x == min(vars) else 2 for x in X[:,col]]

		else:
			mapping_scheme = reduce(lambda x,y: x+y,[a*[b] for a,b in zip(best_schemes[int(col)],[1,2,3])])
			#mapping_scheme = [x+y for x,y in a*[b] for a,b in [zip(best_schemes[int(col)],[1,2,3])]]
			#X_rebucketed[:,col] = map(lambda x: mapping_scheme[int(x)-1], X[:,col])
			print mapping_scheme, col, names[col], question_dict[names[col]]['bucket_scheme'], set(X[:,col])
			X_rebucketed[:,col] = [mapping_scheme[int(x)-1] for x in X[:,col]]
			# http://stackoverflow.com/questions/3459098/create-list-of-single-item-repeated-n-times-in-python
			# http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python

		rebucket_counts = Counter(X_rebucketed[:,col]).values()
		
		# if len(rebucket_counts)<3:
		# 	rebucket_counts.append(0) # binary data

		#question_dict[names[col]]['rebucket_counts_1'], question_dict[names[col]]['rebucket_counts_2'], question_dict[names[col]]['rebucket_counts_3'] = rebucket_counts
		#question_dict[names[col]]['rebucket_shares_1'], question_dict[names[col]]['rebucket_shares_2'], question_dict[names[col]]['rebucket_shares_3'] = [float(value) / sum(rebucket_counts) for value in rebucket_counts]

		question_dict[names[col]]['rebucket_counts'] = rebucket_counts
		question_dict[names[col]]['rebucket_shares'] = [float(value) / sum(rebucket_counts) for value in rebucket_counts]

	X_rebucketed_df = pd.DataFrame(X_rebucketed, columns=names)
	#print "X_rebucketed_df.columns:", X_rebucketed_df.columns.tolist()

	# rebucketed_filename = 'X_rebucketed.csv'
	# np.savetxt(basedir + rebucketed_filename, X_rebucketed, fmt='%.2f', delimiter=",")
	# print rebucketed_filename, "saved to:", basedir

	rebucketed_filename = 'X_rebucketed_' + session_id + '.csv'
	X_rebucketed_df.to_csv(basedir + 'static/data_file/' + rebucketed_filename, index=False)
	print rebucketed_filename, "saved to:", basedir + 'static/data_file/' 

	#http://stackoverflow.com/questions/7701429/efficient-evaluation-of-a-function-at-every-cell-of-a-numpy-array
	#def rebucket(x):
	#	return mapping_scheme[(int(x))-1]
	#
	#rebucket_v = np.vectorize(rebucket)  # or use a different name if you want to keep the original f
	#X_rebucketed_2 = rebucket_v(X) 

	print "rebucketing check:"
	print "response matrix rebucketed (row, column):"
	print X_rebucketed.shape
	random_col = np.random.randint(0,X.shape[1])
	print "original responses:", X[:,random_col]
	print "bucketing scheme: ", best_schemes[random_col]
	print "rebucketed responses:", X_rebucketed[:,random_col]

	# print question_dict

	save_db(session_id, X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names)
	return rebucketed_filename, X_rebucketed, question_dict, X_rebucketed_df, rh

def make_cluster_seed(factor_matrix, best_factor, question_number, num_cols, num_rows, rh, session_id):
	'''
	Builds cluster_seed (size n_factors), grouping questions by highest PCA value then choosing lowest RH as cluster_seed

	'''
	z = func_name(); print "--------in function: ", z, " -------------"

	X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names = load_db(session_id)
	print 'X_rebucketed.shape: ', X_rebucketed.shape
	
	#global cluster_seed
	cluster_seed=[]

	# Step 5: now group questions by highest factor (=factor_matrix[best_factor], remember col #1 = index 0!)
	# http://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
	
	#factor_matrix, best_factor, question_number, num_cols, num_rows = factor_matrix, best_factor, question_number, num_cols, num_rows

	#print factor_matrix.shape
	sorted_factor_matrix = factor_matrix[factor_matrix[:,best_factor].argsort()]
	#print "verify question index:", sorted_factor_matrix[:,question_number]
	#print "verify sort by highest factor", sorted_factor_matrix[:,best_factor]

	# determine unique factor numbers:
	unique_factor_list= list(set(sorted_factor_matrix[:,num_cols+1]))
	# print unique_factor_list

	# now group questions by primary factor and sort by RH to form cluster 'seed'

	# print ("[question number, index value, rosetta heuristic]")
	for factor in unique_factor_list:
		# filter by primary factor, returns question indices, then list questions 
		question_index = sorted_factor_matrix[sorted_factor_matrix[:,best_factor]==factor][:,question_number] # filter == factor
		rh_list = sorted_factor_matrix[sorted_factor_matrix[:,best_factor]==factor][:,rh] # filter == factor
		question_index_list = list(question_index) # convert to list from np.array
		question_index_int_list = [int(x) for x in question_index_list] # convert from dtype = numpy.64 to int
		question_names = [names[q] for q in question_index_int_list] # finally filter question list by question index
		rh_list = list(rh_list)
		rh_list = [round(x) for x in rh_list]
		#print int(factor), ":", [names[q] for q in question_index_int_list] # finally filter question list by question index
		#print rh_list
		question_index_rh_list = sorted(zip(question_names, question_index_list, rh_list), key = lambda x: x[2])
		# print int(factor+1), ":", question_index_rh_list #this prints sorted RH by question
		cluster_seed.append(int(question_index_rh_list[0][1]))

	print "cluster seed:", cluster_seed
	print "number of items in cluster seed", len(cluster_seed)

	save_db(session_id, X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names)
	db.update_one({'session_id': session_id},{"$set":{'cluster_seed':cluster_seed}},upsert=True)
	return cluster_seed

# step 6: use cluster_seed as input to poLCA
#def poLCA(cluster_seed, num_seg, num_rep, rebucketed_filename, session_id):
# def poLCA(i, cluster_seed, num_seg, num_rep, rebucketed_filename, session_id):
def poLCA(i, cluster_seed, num_seg, num_rep, session_id):
	'''
	Runs poLCA script in R - see http://dlinzer.github.io/poLCA/
	'''
	z = func_name(); print "--------entering function: ", z, " -------------"

	X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names = load_db(session_id)

	import subprocess
	import base64
	import uuid

	#return 


	# cluster_seed, num_seg, num_rep, rebucketed_filename = cluster_seed, num_seg, num_rep, rebucketed_filename

	# # number of segments
	#num_seg = 6
	#num_rep = 1

	# # Define command and arguments
	command = 'Rscript'
	# #path2script = '/Users/pniessen/Rosetta_Desktop/Segmentation_2-point-0/poLCA_test_v2.R'
	# path2script = '/Users/pniessen/Rosetta_Desktop/Segmentation_2-point-0/simple_poLCA_test.R'
	path2script = basedir +'simple_poLCA_test.R'
	rebucketed_filename = 'static/data_file/X_rebucketed_' + session_id + '.csv'
	#infile = basedir + 'static/data_file/' + rebucketed_filename

	#cluster_seed_names = ['q06_1', 'q06_2', 'q06_3','q06_4','q06_5','q06_6','q06_7']

	# # Variable number of args in a list
	# #args = ['A', 'B', 'C', 'D']
	# #args = ['11', '3', '9', '42']
	cluster_seed_names = [names[int(value)] for value in cluster_seed]
	#timestamp = base64.b64encode(str(time.time() + np.random.random_integers(0,1000000000000)))
	timestamp = str(uuid.uuid4()) # http://stackoverflow.com/questions/534839/how-to-create-a-guid-in-pytho

	#weights_filename = 'test_raw_data_q_v1_weights.csv'
	weights_filename = 'static/weights_file/X_weights_' + session_id + '.csv'

	cluster_seeds = [num_seg] + [num_rep] + [basedir] + [rebucketed_filename] + [timestamp] + cluster_seed_names + [weights_filename]
	cluster_seeds = [str(seed) for seed in cluster_seeds]
	#print cluster_seeds
	# #args = ['74','68','75','73','162','182','168','69','78','70','113','181','179','81','201']
	args = cluster_seeds
	# #print args
	# # Build subprocess command
	cmd = [command, path2script] + args

	#print cmd

	# # check_output will run the command and store to result
	print "Running poLCA in R, please wait:"
	print cluster_seeds
	x = subprocess.check_output(cmd, universal_newlines=True)

	print x

	#return
	#scorecard(cluster_seeds, cluster_seed_names, num_seg)
	print("------- Runtime: %.2f seconds -------" % (time.time() - start_time))

	save_db(session_id, X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names)
	
	z = func_name(); print "--------exiting function: ", z, " -------------"
	
	return timestamp, cluster_seeds, cluster_seed_names, num_seg

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route("/")
def hello():
    #return "Hello World!"
    #"/Users/pniessen/Rosetta_Desktop/Segmentation_2-point-0/startbootstrap-grayscale-1.0.6/index.html"
    #"/Users/pniessen/Rosetta_Desktop/Segmentation_2-point-0/table.html"
    with open(basedir + 'index.html', 'r') as viz_file:
        return viz_file.read()

@app.route("/data", methods=["GET"])
def scorecard_2():
	z = func_name(); print "--------in function: ", z, " -------------"

	session_id = flask.request.args.get('session_id')
	print 'session id: ', session_id

	X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names = load_db(session_id)
	
	training_list = []

	print "results_dict size: ", len(results_dict.keys())
	print "results_dict keys: ", results_dict.keys()

	for key in results_dict:
		#train_dict = {'a':key,'b':results_dict[key]['cluster_number'],'c':results_dict[key]['num_vars'],'d':esults_dict[key]['model_stats'],'e':16,'f':17}
		train_dict = {'a':key[:10],'b':results_dict[key]['cluster_number'],'c':results_dict[key]['num_vars'],'d':round(results_dict[key]['model_stats'][0]),'e':round(results_dict[key]['model_stats'][1]),'f':round(results_dict[key]['model_stats'][2]), 'g':round(results_dict[key]['weighted_average_cluster_polarity'],4) ,'h':round(results_dict[key]['average_cross_question_polarity'],4),'i':results_dict[key]['method']}
		
		#print "one train dict entry:", train_dict
		#sorted(d, key=d.get)

		# print key
		# print "number of clusters:", results_dict[key]['cluster_number'] 
		# print "number of variables", results_dict[key]['num_vars'] 
		# print "model stats:", results_dict[key]['model_stats']
		# print "weighted average cluster polarity: ", results_dict[key]['weighted_average_cluster_polarity']
		# print "average cross-question polarity: ", results_dict[key]['average_cross_question_polarity']

		training_list.append(train_dict)
		#print training_list

	training_results = {'training': training_list}

	save_db(session_id, X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names)
	return flask.jsonify(training_results)

def scorecard_3(keys_to_upload, session_id):

	z = func_name(); print "--------in function: ", z, " -------------"
	X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names = load_db(session_id)

	run_scorecards = []

	print "keys_to_upload size: ", len(keys_to_upload)
	print "keys_to_upload: ", keys_to_upload

	for key in keys_to_upload:
		#train_dict = {'a':key,'b':results_dict[key]['cluster_number'],'c':results_dict[key]['num_vars'],'d':esults_dict[key]['model_stats'],'e':16,'f':17}
		run_scorecard = {'a':key,'b':results_dict[key]['cluster_number'],'c':results_dict[key]['num_vars'],
		'd':round(results_dict[key]['model_stats'][0]),'e':round(results_dict[key]['model_stats'][1]),
		'f':round(results_dict[key]['model_stats'][2]), 'g':round(results_dict[key]['weighted_average_cluster_polarity'],4) ,
		'h':round(results_dict[key]['average_cross_question_polarity'],4),'i':results_dict[key]['method'],
		'j':results_dict[key]['date']}
		
		#print "one train dict entry:", train_dict
		#sorted(d, key=d.get)

		# print key
		# print "number of clusters:", results_dict[key]['cluster_number'] 
		# print "number of variables", results_dict[key]['num_vars'] 
		# print "model stats:", results_dict[key]['model_stats']
		# print "weighted average cluster polarity: ", results_dict[key]['weighted_average_cluster_polarity']
		# print "average cross-question polarity: ", results_dict[key]['average_cross_question_polarity']

		run_scorecards.append(run_scorecard)
		#print training_list

	#save_db(session_id, X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names)
	return run_scorecards

@app.route("/create_tracker", methods=["GET"])
def create_tracker():
	z = func_name(); print "--------entering function: ", z, " -------------"

	session_id = flask.request.args.get('session_id')
	print 'session id: ', session_id
	
	factor_matrix, names, X, question_dict, results_dict = get_PCA(filename, weights_filename, session_id)
	factor_matrix, num_rows, num_cols, question_number, rh, best_factor, second_best_factor, question_dict = top_n_factors(factor_matrix, top_n, question_dict, names, session_id)
	rebucketed_filename, X_rebucketed, question_dict, X_rebucketed_df, rh = rebucket(factor_matrix, names, X, rh, question_dict, session_id)
	cluster_seed = make_cluster_seed(factor_matrix, best_factor, question_number, num_cols, num_rows, rh, session_id)

	tracker_list = []

	#print "X_rebucketed.shape:", X_rebucketed.shape
	#print "X.shape", X.shape

	#print question_dict.keys()
	#print question_dict[question_dict.keys()[0]]
	#print question_dict[question_dict.keys()[0]].keys()

	# if len(results_dict) == 0 and 'run_tracker' not in question_dict[question_dict.keys()[0]].keys():
	# can this be written with a try: [add data] except [data=0] for each var to add crashproofness?
	for key in question_dict:
		try:
			tracker_dict = OrderedDict([('001_Q#',key),('002_Q_name',question_dict[key]['question_text']),('003_Dim','n/a'),('0031Dimension','n/a'),('004_LF', round(question_dict[key]['first_factor_value'],3)),('005_#1 Factor', question_dict[key]['first_factor']),('006_#2 Factor', question_dict[key]['second_factor']),('007_Bucket', question_dict[key]['bucket_scheme']),('008_%T(=1)', round(question_dict[key]['rebucket_shares'][0]*100)),('009_%M(=2)', round(question_dict[key]['rebucket_shares'][1]*100)),('010_%L(=3)', round(question_dict[key]['rebucket_shares'][2]*100)), ('011_RH',round(question_dict[key]['best_rh'],2))])
		except:
			tracker_dict = OrderedDict([('001_Q#',key),('002_Q_name',question_dict[key]['question_text']),('003_Dim','n/a'),('0031Dimension','n/a'), ('004_LF', round(question_dict[key]['first_factor_value'],3)),('005_#1 Factor', question_dict[key]['first_factor']),('006_#2 Factor', question_dict[key]['second_factor']),('007_Bucket', question_dict[key]['bucket_scheme']),('008_%T(=1)', round(question_dict[key]['rebucket_shares'][0]*100)),('009_%M(=2)', round(question_dict[key]['rebucket_shares'][1]*100)),('010_%L(=3)', 0), ('011_RH',round(question_dict[key]['best_rh'],2))])	
		tracker_list.append(tracker_dict)
	
	corr_matrix = np.corrcoef(X.T)
	corr_matrix_list = dict(zip(names,[dict(zip(names, corr_matrix[:,row].tolist())) for row in range(corr_matrix.shape[1])]))
	print 'correlation matrix shape: ', corr_matrix.shape

	cluster_seed_question_list = [names[seed] for seed in cluster_seed]
	print 'cluster_seed_question_list being sent to front end: ', cluster_seed_question_list

	visual_flag_list = [visual_mode]

	tracker = {'tracker': tracker_list, 'corr_matrix': corr_matrix_list, 'cluster_seed': cluster_seed_question_list, 'visual_mode': visual_flag_list}
	#print "create_run_tracker: ", tracker

	tracker_json = flask.jsonify(tracker)
	print tracker_json

	save_db(session_id, X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names)

	z = func_name(); print "--------exiting function: ", z, " -------------"
	return flask.jsonify(tracker)


@app.route("/create_tracker_from_saved_session", methods=["GET"])
def make_tracker_data():
	z = func_name(); print "--------entering function: ", z, " -------------"
	
	session_id = flask.request.args.get('session_id')
	print 'session id: ', session_id

	X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names = load_db(session_id)
	cluster_seed = db.find_one({'session_id': session_id})['cluster_seed']

	tracker_list = []

	for key in question_dict:
		try:
			tracker_dict = OrderedDict([('001_Q#',key),('002_Q_name',question_dict[key]['question_text']),('003_Dim','n/a'),('0031Dimension','n/a'),('004_LF', round(question_dict[key]['first_factor_value'],3)),('005_#1 Factor', question_dict[key]['first_factor']),('006_#2 Factor', question_dict[key]['second_factor']),('007_Bucket', question_dict[key]['bucket_scheme']),('008_%T(=1)', round(question_dict[key]['rebucket_shares'][0]*100)),('009_%M(=2)', round(question_dict[key]['rebucket_shares'][1]*100)),('010_%L(=3)', round(question_dict[key]['rebucket_shares'][2]*100)), ('011_RH',round(question_dict[key]['best_rh'],2))])
		except:
			tracker_dict = OrderedDict([('001_Q#',key),('002_Q_name',question_dict[key]['question_text']),('003_Dim','n/a'),('0031Dimension','n/a'), ('004_LF', round(question_dict[key]['first_factor_value'],3)),('005_#1 Factor', question_dict[key]['first_factor']),('006_#2 Factor', question_dict[key]['second_factor']),('007_Bucket', question_dict[key]['bucket_scheme']),('008_%T(=1)', round(question_dict[key]['rebucket_shares'][0]*100)),('009_%M(=2)', round(question_dict[key]['rebucket_shares'][1]*100)),('010_%L(=3)', 0), ('011_RH',round(question_dict[key]['best_rh'],2))])	
		tracker_list.append(tracker_dict)
	
	corr_matrix = np.corrcoef(X.T)
	corr_matrix_list = dict(zip(names,[dict(zip(names, corr_matrix[:,row].tolist())) for row in range(corr_matrix.shape[1])]))
	#print 'correlation matrix shape: ', corr_matrix.shape

	cluster_seed_question_list = [names[seed] for seed in cluster_seed]
	#print 'cluster_seed_question_list being sent to front end: ', cluster_seed_question_list

	visual_flag_list = [visual_mode]

	tracker = {'tracker': tracker_list, 'corr_matrix': corr_matrix_list, 'cluster_seed': cluster_seed_question_list, 'visual_mode': visual_flag_list}
	z = func_name(); print "--------exiting function: ", z, " -------------"
	
	return flask.jsonify(tracker)

	
@app.route("/tracker", methods=["GET"])
def update_run_tracker():
	z = func_name(); print "--------entering function: ", z, " -------------"

	session_id = flask.request.args.get('session_id')
	print 'session id: ', session_id

	X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names = load_db(session_id)

	# could we use intersection of two dictionaries (resuts_dict_old, resuts_dict_current) and update difference?
	# http://code.activestate.com/recipes/576644-diff-two-dictionaries/
	# or dict.viewitems() - see http://zetcode.com/lang/python/dictionaries/

	#global results_dict #updated when new values uploaded
	#global cluster_seed
	#global X_rebucketed_df

	tracker_list = []
	r_and_q_list = []
	#print "in update_run_tracker function"
	#print question_dict.keys()
	#print question_dict[question_dict.keys()[0]]
	#print question_dict[question_dict.keys()[0]].keys()

	keys_to_upload = [key for key in results_dict if results_dict[key]['upload_state'] == False]
	starting_run = len(results_dict.keys()) - len(keys_to_upload)
	ending_run = len(results_dict.keys())
	
	if starting_run == ending_run: # no updates
		print "keys to upload", len(keys_to_upload), keys_to_upload
		print "starting_run", starting_run
		print "ending_run", ending_run

		return

	else: # updates
		print "keys to upload", len(keys_to_upload), keys_to_upload
		print "starting_run", starting_run
		print "ending_run", ending_run

		for key in keys_to_upload:
			results_dict[key]['upload_state'] = True
		
		for key2 in question_dict:
			r_list = ['R'] * len(keys_to_upload)
			run_num_list = range(starting_run, ending_run)
			r_dict_keys = [one_r + str(one_run).zfill(3) for one_r, one_run in zip(r_list,run_num_list)]
			r_dict = OrderedDict(zip(r_dict_keys, question_dict[key2]['bool_run_tracker'][starting_run:]))
			r_q_v = (r_dict_keys, key2, question_dict[key2]['bool_run_tracker'][starting_run:])
			r_dict = OrderedDict(sorted(r_dict.items(), key=lambda x: x[0]))
			#print r_dict
			#tracker_dict = OrderedDict('a':key,'b':"coming soon",'c':"coming soon")#,'d':round(question_dict[key]['first_factor_value'],3),'e':question_dict[key]['first_factor'],'f':question_dict[key]['second_factor'],'g':question_dict[key]['bucket_scheme'], 'h': round(question_dict[key]['rebucket_shares_1']*100),'i': question_dict[key]['rebucket_shares_2'],'j': question_dict[key]['rebucket_shares_3'],'k': round(question_dict[key]['best_rh']))
			# tracker_dict = OrderedDict([('001_Q#',key),('002_Q_name','n/a'),('003_Dim','n/a'), ('004_PCA', round(question_dict[key]['first_factor_value'],3)),('005_#1 Factor', question_dict[key]['first_factor']),('006_#2 Factor', question_dict[key]['second_factor']),('007_Bucket', question_dict[key]['bucket_scheme']),('008_%T(=1)', round(question_dict[key]['rebucket_shares_1']*100)),('009_%M(=2)', round(question_dict[key]['rebucket_shares_2']*100)),('010_%L(=3)', round(question_dict[key]['rebucket_shares_3']*100)), ('011_RH',round(question_dict[key]['best_rh'],2))])
			# #print tracker_dict
			# tracker_dict.update(r_dict)
			#pprint.pprint(tracker_dict, width=1)
			# tracker_list.append(tracker_dict)
			tracker_list.append(r_dict)
			r_and_q_list.append(r_q_v)
			#print tracker_list
		
		#tracker = {'tracker': tracker_list}
		#print tracker
		#print r_and_q_list

		run_reports_list = run_report(keys_to_upload, session_id)
		run_scorecards_list = scorecard_3(keys_to_upload, session_id)
		rfe_list = feature_importances(X_rebucketed_df, keys_to_upload, session_id)
		# cluster_seed_question_list = [names[seed] for seed in cluster_seed]
		# print 'cluster_seed_list being sent to front end:', cluster_seed_question_list
		tracker = {'tracker': tracker_list, 'run_reports': run_reports_list, 
		'run_scorecard': run_scorecards_list, 'feature_importance': rfe_list}

		save_db(session_id, X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names)
		z = func_name(); print "--------exiting function: ", z, " -------------"

		return flask.jsonify(tracker)

@app.route("/submit_data", methods=["POST"])
def submit_data():
	z = func_name(); print "--------entering function: ", z, " -------------"

	#print 'X_rebucketed.shape:', X_rebucketed.shape

    # read the data that came with the POST request as a dict
    # inbound request example: http://127.0.0.1:5000/predict -X POST -H 'Content-Type: application/json' -d '{"example": [154]}'
    # my example: http://127.0.0.1:5000/test_predict -X POST -H 'Content-Type: application/json' -d '{'keywords': ['rabbit','goat','wine'],'rating_metric': ['lda'],'distance_metric': ['pearsonr']}'
    # simple example: curl http://127.0.0.1:5000/test_predict -X POST -H 'Content-Type: applcation/json' -d '{"keywords": '66666'}'
	# curl http://127.0.0.1:5000/submit_data -X POST -H 'Content-Type: applcation/json' -d '{u'segments': [6], u'questions': u'[q39_8,q39_4,q39_6,q48_28,q31_20,q07_18,q07_17,q07_11,q35_5,q16_8,q06_2,q08_5,q08_6,q06_8,q06_16,q35_8,q48_25,q48_27]'}'
	# {"questions":"[q39_8,q39_4,q39_6,q48_28,q31_20,q07_18,q07_17,q07_11,q35_5,q16_8,q06_2,q08_5,q08_6,q06_8,q06_16,q35_8,q48_25,q48_27]", "segments":[6]} 
	# 	
	#global cluster_seed
	#global names
	#global X_rebucketed_df
	#global X_rebucketed

	counter = 0
	inbound_data = flask.request.json
	counter +=1
	print "counter:", counter
	print "inbound data", inbound_data
	keys = inbound_data.keys(); print keys
	for key in inbound_data:
		print key, inbound_data[key], type(inbound_data[key])

	if 'session_id' in keys:
		print "--session id-----"
		session_id = inbound_data['session_id']
		print 'session_id', session_id

		X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names = load_db(session_id)
		print 'X_rebucketed.shape:', X_rebucketed.shape

	if 'questions' in keys:
		print "----questions---"
		questions = inbound_data['questions']
		
		if questions != None:
			#is not None and questions.strip("[]").encode('ascii', 'ignore') is not 'null':
			#questions = questions.split(',')
			questions = [q.encode('ascii', 'ignore') for q in questions]
			print '!! Questions:', questions
			print '!! names:', names
			cluster_seed_inbound = [names.index(question) for question in questions] # convert questions to index #
			print "questions:", len(questions), questions

		else:
			cluster_seed = db.find_one({'session_id': session_id})['cluster_seed']
			cluster_seed_inbound = cluster_seed
			print "no questions provided; using cluster seed"

	if "segments" in keys:
		print "---segments----"
		num_segments = inbound_data['segments']
		num_segments = int(num_segments)
		# while type(num_segments) != 'int':
		# 	num_segments = int(num_segments.encode('ascii','ignore'))
		
		print "num_segments:", num_segments
		print 'num_segments type:', type(num_segments)

	if 'xls' in keys:
		xls = False
		print '---excel export----'
		xl_flag = inbound_data['xls']#.strip("[]")

		if xl_flag == 'true':
			xls = True

		print "excel export:", xls

	if 'grid_search' in keys:
		grid_search = False
		print '---grid search----'
		gs_flag = inbound_data['grid_search']#.strip("[]")
		
		if gs_flag == True:
			grid_search = True

		print "grid_search:", grid_search

	if 'method' in keys:
		print "----method---"
		method = inbound_data['method']
		if len(method) == 0:
			method = 'poLCA'
		
		if method != None:
			method = method.encode('ascii', 'ignore')

		print "method:", method	

	#cluster_seed_inbound = [names.index(question) for question in questions]
	print "cluster_seed_inbound:", cluster_seed_inbound

	#cluster_seed = [60.0, 90.0, 24.0, 125.0, 72.0, 61.0, 30.0, 68.0, 14.0, 123.0, 144.0, 49.0, 65.0, 53.0, 50.0, 12.0, 58.0, 146.0, 112.0, 81.0, 48.0, 43.0, 134.0, 139.0, 51.0, 86.0, 4.0, 94.0, 111.0, 99.0]
	#cluster_seed = cluster_seed_2
	num_seg = num_segments
	#grid_search = False
	num_rep = 1
	#rebucketed_filename = 'X_rebucketed.csv'

	#print "----run number:", len(result_dict.keys()), "-----------"
 	if method.lower() in ['agclust', 'meanshift', 'dbscan', 'kmeans', 'affinityprop', 'birch', 'spectral']:
 		results = get_segments(X_rebucketed_df, names, cluster_seed_inbound, method.lower(), num_seg, session_id)
 		print method, ' results:', results
	
	else:
		print "---------now running poLCA from submit_data function--------------"
		print 'X_rebucketed.shape: ', X_rebucketed.shape
		results = run_poLCA (grid_search,cluster_seed_inbound,num_seg,num_rep, session_id)	
	
	timestamps = update_results_dict(results, X_rebucketed, names, method, session_id)

	#print "---question_dict:-----", len(question_dict)
	#print "---results_dict:------", len(results_dict)
	
	update_question_dict(timestamps, session_id)
	run_report(timestamps, session_id)
	# threading.Thread(target=make_visual, args=(X_rebucketed_df, timestamps), kwargs={}).start()
	# print 'this is after threading started'
	if visual_mode == True:
		Parallel(n_jobs=num_cores,verbose=5)(delayed(make_visual)(i,X_rebucketed_df, timestamps[i]) for i in range(len(timestamps)))
	#rfe_list = feature_importances(X_rebucketed_df, timestamps)
	clean_up(timestamps, session_id)

	if xls:
		make_xls(session_id)


	results2 = {"example": [155]}
	z = func_name(); print "--------exiting function: ", z, " -------------"
	#save_db(session_id, X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names)

	return flask.jsonify(results2)

@app.route('/submit_objective_function', methods=['POST'])
def objective_function():

	cookies = flask.request.cookies
	session_id = cookies['session_id']
	X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names = load_db(session_id)

	#global question_dict
	print "questions: ", question_dict.keys()

	z = func_name(); print "--------entering function: ", z, " -------------"
	inbound_data = flask.request.json

	print "inbound data", inbound_data
	keys = inbound_data.keys(); print keys

	objective_functions_dict = dict([(key.encode('ascii', 'ignore'), [value.encode('ascii', 'ignore') for value in values]) for key, values in inbound_data.items()])

	print objective_functions_dict

	for key, questions in objective_functions_dict.iteritems():
		for question in questions:
			 question_dict[question]['dimension'] = key

	print 'question_dict[dimension] updated:'

	for key, questions in objective_functions_dict.iteritems():
		for question in questions:
			 print question, ": ", question_dict[question]['dimension']

	z = func_name(); print "--------exiting function: ", z, " -------------"
	save_db(session_id, X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names)

	results3 = {"example": [666]}
	return flask.jsonify(results3)

@app.route('/submit_session_id', methods=['POST'])
def session_id():

	z = func_name(); print "--------in function: ", z, " -------------"

	#global results_dict
	#global question_dict

	inbound_data = flask.request.json

	print "user_id, session_id:", inbound_data
	print type(inbound_data)
	
	user_id = inbound_data[0].encode('ascii', 'ignore')
	print type(user_id)
	print user_id

	session_id = inbound_data[1].encode('ascii', 'ignore')
	print type(session_id)
	print session_id

	# create subdirectory with user_id
	d = 'static/saved_sessions/' + user_id
	print 'user_id directory:', d
	if not os.path.exists(d):
		os.makedirs(d)
		print 'created user_id directory:', d
	
	results4 = {"session_id": [session_id]}
	
	return flask.jsonify(results4)

@app.route('/submit_dimension', methods=['POST'])
def submit_dimension():

	z = func_name(); print "--------entering function: ", z, " -------------"

	cookies = flask.request.cookies
	session_id = cookies['session_id']
	X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names = load_db(session_id)

	#global results_dict
	#global question_dict

	dimension = flask.request.json
	cookies = flask.request.cookies

	print "dimension", dimension
	print type(dimension)
	
	print "cookies:", cookies
	print type(cookies)

	print "cookies[session_id]:", cookies['session_id']
	session_id = cookies['session_id']
	print type(session_id)

	session_id = dimension[0]
	print session_id, type(session_id)

	new_dimension = [value.encode('ascii', 'ignore') for value in dimension[1]]
	print type(new_dimension)
	print new_dimension
	question, dim = new_dimension
	question_dict[question]['dimension'] = dim

	print 'added to question_dict: ', question, question_dict[question]['dimension']

	results5 = {"new_dimension": new_dimension}
	
	z = func_name(); print "--------exiting function: ", z, " -------------"
	save_db(session_id, X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names)

	return flask.jsonify(results5)

@app.route('/save', methods=['POST'])
def save_session():

	z = func_name(); print "--------in function: ", z, " -------------"

	inbound_data = flask.request.json

	print "user_id, session_id:", inbound_data
	print type(inbound_data)
	
	user_id = inbound_data[0].encode('ascii', 'ignore')
	print type(user_id)
	print user_id

	session_id = inbound_data[1].encode('ascii', 'ignore')
	print type(session_id)
	print session_id
	
	filename = save_results(user_id, session_id)
	print 'filename:', filename

	# testing loop to see if load_results works
	
	# global results_dict
	# global question_dict
	# global X
	# global X_rebucketed
	
	# results_dict = {}
	# question_dict = {}
	# X = np.zeros(X.shape)
	# X_rebucketed = np.zeros(X_rebucketed.shape)

	# print '-------data containers emptied after save_results()-----------'
	# print "results_dict:", len(results_dict)
	# print "question_dict:", len(question_dict)
	# print 'X: size, all zeros?', X.shape, np.all(X==0)
	# print 'X_rebucketed: size, all zeros?', X_rebucketed.shape, np.all(X_rebucketed==0)

	#load_results(filename)
	
	results5 = {"save_results": filename}
	
	return flask.jsonify(results5)

@app.route('/load_session', methods=['POST'])
def load_session():

	z = func_name(); print "--------in function: ", z, " -------------"

	inbound_data = flask.request.json

	print "user_id, filename:", inbound_data
	print type(inbound_data)
	
	user_id = inbound_data[0].encode('ascii', 'ignore')
	print type(user_id)
	print user_id

	filename = inbound_data[1].encode('ascii', 'ignore')
	print type(filename)
	print filename

	cookies = flask.request.cookies
	print "cookies:", cookies
	print type(cookies)
	
	session_id = cookies['session_id']
	print 'session_id:', session_id
	
	new_session(session_id)
	load_results(user_id, filename,session_id)
	
	results5 = {"save_results": filename}
	
	return flask.jsonify(results5)

@app.route('/filenames', methods=['POST'])
def get_filenames():
	z = func_name(); print "--------in function: ", z, " -------------"
	# create directory on submission of session_id

	basedir = os.path.abspath(os.path.dirname(__file__)) + "/static/saved_sessions/"
	#mypath = '/Users/pniessen/Rosetta_Desktop/Segmentation_2-point-0/segmentr/web/static/saved_sessions'
	user_id = flask.request.json
	print user_id
	print type(user_id)
	#user_id = user_id.encode('ascii', 'ignore')
	#print user_id
	mypath = basedir + user_id + "/"
	filenames = os.listdir(mypath)
	# http://stackoverflow.com/questions/3207219/how-to-list-all-files-of-a-directory-in-python?rq=1
	print filenames
	
	results5 = {"filenames": filenames}
	
	return flask.jsonify(results5)

# file upload section
# see http://code.runnable.com/UiPcaBXaxGNYAAAL/how-to-upload-a-file-to-the-server-in-flask-for-python
# https://github.com/moremorefor/flask-fileupload-ajax-example/blob/master/app.py
# http://stackoverflow.com/questions/18334717/how-to-upload-a-file-using-an-ajax-call-in-flask

'''
# Route that will process the file upload
@app.route('/upload', methods=['POST'])
def upldfile():
	z = func_name(); print "--------in function: ", z, " -------------"

	#global X_rebucketed
	#global X
	#global names
	#global results_dict
	#global question_dict

	#print 'X_rebucketed.shape:', X_rebucketed.shape

	if request.method == 'POST':
		files = request.files['file']
        print files
        if files and allowed_file(files.filename):
            filename = secure_filename(files.filename)
            print filename
            app.logger.info('FileName: ' + filename)
            updir = os.path.join(basedir_for_upload, 'uploads/')
            files.save(os.path.join(updir, filename))
            file_size = os.path.getsize(os.path.join(updir, filename))
            filename = 'uploads/' + filename

            global cluster_seed
            #global question_dict
            #global names
            #global X_rebucketed_df

            factor_matrix, names, X, question_dict, results_dict = get_PCA(filename)
            factor_matrix, num_rows, num_cols, question_number, rh, best_factor, second_best_factor, question_dict = top_n_factors(factor_matrix, top_n, question_dict, names)
            rebucketed_filename, X_rebucketed, question_dict, X_rebucketed_df = rebucket(factor_matrix, names, X, rh, question_dict)
            
            print 'X_rebucketed.shape:', X_rebucketed.shape

            cluster_seed = make_cluster_seed(factor_matrix, best_factor, question_number, num_cols, num_rows)
            
            return jsonify(name=filename, size=file_size)#, question_dict
'''
# Route that will process the file upload
# http://code.runnable.com/UiPeYmdVjZlYAAAf/how-to-upload-multiple-files-in-flask-for-python
# this is for weights
@app.route('/upload_2', methods=['POST'])
def upldfile_2():
	#z = func_name(); print "--------in function: ", z, " -------------"

	#global X_rebucketed
	#global X
	#global names
	#global results_dict
	#global question_dict
	#global weights_filename

	#print 'X_rebucketed.shape:', X_rebucketed.shape

	if request.method == 'POST':
		
		z = func_name(); print "--------entering function: ", z, " -------------"
		
		files = request.files['file']
        print files

        cookies = flask.request.cookies
        print "cookies:", cookies
        print type(cookies)

        session_id = cookies['session_id']
        print 'session_id:', session_id

    	if files and allowed_file(files.filename):
            filename = secure_filename(files.filename)
            print filename
            app.logger.info('FileName: ' + filename)
            updir = os.path.join(basedir_for_upload, 'uploads/')
            files.save(os.path.join(updir, filename))
            file_size = os.path.getsize(os.path.join(updir, filename))
            filename = 'uploads/' + filename

            # create new db record, save filenames
            new_session(session_id)
            db.update_one({'session_id': session_id},{"$set":{'weights_filename': filename }},upsert=True)

        #     global cluster_seed
        #     global question_dict
        #     global names
        #     global X_rebucketed_df

        #     factor_matrix, names, X, question_dict, results_dict = get_PCA(filename)
        #     factor_matrix, num_rows, num_cols, question_number, rh, best_factor, second_best_factor, question_dict = top_n_factors(factor_matrix, top_n, question_dict, names)
        #     rebucketed_filename, X_rebucketed, question_dict, X_rebucketed_df = rebucket(factor_matrix, names, X, rh, question_dict)
            
        #     print 'X_rebucketed.shape:', X_rebucketed.shape

        #     cluster_seed = make_cluster_seed(factor_matrix, best_factor, question_number, num_cols, num_rows)
        
        weights_filename = filename
        print 'weights_filename:', filename

        return jsonify(name=filename, size=file_size)#, question_dict

# this is for survey data 
@app.route('/upload_3', methods=['POST'])
def upldfile_3():
	z = func_name(); print "--------entering function: ", z, " -------------"

	#global X_rebucketed
	#global X
	#global names
	#global results_dict
	#global question_dict
	#global weights_filename

	#print 'X_rebucketed.shape:', X_rebucketed.shape

	if request.method == 'POST':

		z = func_name(); print "--------entering function: ", z, " -------------"
		
		files = request.files['file']
        print files

        cookies = flask.request.cookies
        print "cookies:", cookies
        print type(cookies)

        session_id = cookies['session_id']
        print 'session_id:', session_id

    	if files and allowed_file(files.filename):
            filename = secure_filename(files.filename)
            print filename
            app.logger.info('FileName: ' + filename)
            updir = os.path.join(basedir_for_upload, 'uploads/')
            files.save(os.path.join(updir, filename))
            file_size = os.path.getsize(os.path.join(updir, filename))
            filename = 'uploads/' + filename

            db.update_one({'session_id': session_id},{"$set":{'data_filename': filename }},upsert=True)

            top_n = 2 # fix this!

            #global cluster_seed
            #global question_dict
            #global names
            #global X_rebucketed_df

            #factor_matrix, names, X, question_dict, results_dict = get_PCA(filename, weights_filename, session_id)
            #factor_matrix, num_rows, num_cols, question_number, rh, best_factor, second_best_factor, question_dict = top_n_factors(factor_matrix, top_n, question_dict, names, session_id)
            #rebucketed_filename, X_rebucketed, question_dict, X_rebucketed_df, rh = rebucket(factor_matrix, names, X, rh, question_dict, session_id)
            
            #print 'X_rebucketed.shape:', X_rebucketed.shape

            #cluster_seed = make_cluster_seed(factor_matrix, best_factor, question_number, num_cols, num_rows, rh, session_id)
        
        print 'data_filename:', filename

        z = func_name(); print "--------exiting function: ", z, " -------------"

        return jsonify(name=filename, size=file_size)#, question_dict
'''
# Route that will process the file upload
@app.route('/upload_weights', methods=['POST'])
def upload_weights():
	z = func_name(); print "--------in function: ", z, " -------------"

	global X_rebucketed
	global X
	global names
	global results_dict
	global question_dict

	print 'X_rebucketed.shape:', X_rebucketed.shape

	if request.method == 'POST':
		files = request.files['file']
        print files
        if files and allowed_file(files.filename):
            filename = secure_filename(files.filename)
            print filename
            app.logger.info('FileName: ' + filename)
            files.save(os.path.join(basedir, filename))
            file_size = os.path.getsize(os.path.join(basedir, filename))

	    calculate_weights(filename)

	    return jsonify(name=filename, size=file_size)
'''

# download excel file
# http://stackoverflow.com/questions/30024948/flask-download-a-csv-file-on-clicking-a-button
@app.route('/download') # this is a job for GET, not POST
def download_file():
	z = func_name(); print "--------in function: ", z, " -------------"
	
	cookies = flask.request.cookies
	print "cookies:", cookies
	print type(cookies)

	session_id = cookies['session_id']
	print 'session_id:', session_id

    #session_id = flask.request.args.get('session_id')
    #print 'session id: ', session_id

	filename = make_xls(session_id)

	return send_file(filename, attachment_filename=filename, as_attachment=True) #mimetype='text/csv'

def scorecard(timestamp, cluster_seeds, cluster_seed_names, num_seg, session_id):
	'''
	Yields scorecard for single segmentation run - legacy module, does not run off results_dict
	'''
	z = func_name(); print "--------in function: ", z, " -------------"

	X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names = load_db(session_id)

	# step 7: load predicted clusters from file, add back to original data matrix (X)	
	# also calculate cluster scorecard metrics
	#timestamp, cluster_seeds, cluster_seed_names, num_seg = results

	# load datafile
	filenames = ['predicted_segment_','posterior_probabilities_','model_stats_']
	predicted_clusters, posterior_probabilities, model_stats = (np.genfromtxt(basedir + "/static/model/" + filename + timestamp + ".txt", delimiter=',', skip_header=0) for filename in filenames)


	cluster_size = [len(predicted_clusters[predicted_clusters == x+1]) for x in range(num_seg)]
	mean_posterior_probabilities = [np.mean(posterior_probabilities[:,x][predicted_clusters ==x+1]) for x in range(num_seg)] 
	cluster_shares = [float(cluster_size[x]) / sum(cluster_size) for x in range(num_seg)]

	# checksums
	if not sum(cluster_shares)==1:
		print "cluster shares do not sum to 1"
		print "cluster share sum:", sum(cluster_shares)

	if not sum(cluster_size)==predicted_clusters.shape[0]:
		print "cluster sizes do not sum to total"
		print "cluster size sum:", sum(cluster_size)
		print "total number of observations:", predicted_clusters.shape[0]

	# cluster scorecard
	print "-------  Cluster Scorecard  ---------"
	#print "Run number:", results_dict[timestamp]['run_number']
	print "Timestamp:", time.asctime(time.localtime(start_time))
	print "Observations (n=):", len(predicted_clusters)
	print "Segments (n=):", num_seg
	print "Number of reps:", num_rep
	print "Clustering variables (n=):", len(cluster_seed_names)
	print "Clustering variables:", cluster_seed_names
	print "Model stats (MLE, Chi Sq, BIC):", ['%.2f' % (ms*1) for ms in list(model_stats)]
	print "Cluster size (n=):", cluster_size
	print "Cluster shares (%):", ['%.2f' % (cluster_share * 100) for cluster_share in cluster_shares]
	print "Mean posterior probabilities (%):", ['%.2f' % (mpp * 100) for mpp in mean_posterior_probabilities]
	print("------- Runtime: %.2f seconds -------" % (time.time() - start_time))

	save_db(session_id, X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names)

	return

# @socketio.on('connect', namespace='/test')
# def test_connect():
# 	z = func_name(); print "--------in function: ", z, " -------------"
#     # need visibility of the global thread object
# 	global thread
# 	emit('my response', {'data': '------connected---------'})
# 	print('--------------------Client connected----------------------------')

def update_results_dict(results, X_rebucketed, names, method, session_id):
	''''
	Updates results_dict after poLCA run
	'''
	z = func_name(); print "--------in function: ", z, " -------------"
	X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names = load_db(session_id)

	print 'X_rebucketed.shape: ', X_rebucketed.shape

	#global results_dict
	#global question_dict

	starting_results_dict_length = len(results_dict.keys())
	print "results_dict size when entering make_results_dict function:", starting_results_dict_length

	# parse results data
	timestamps = [results[num][0] for num in range(len(results))]
	cluster_numbers = [results[num][3] for num in range(len(results))]
	num_vars = [len(results[num][2]) for num in range(len(results))]
	quest_list = [results[num][2] for num in range(len(results))]
	model_stats = [list(loadtxt(basedir+"/static/model/model_stats_"+timestamp+".txt", delimiter=",", unpack=False)) for timestamp in timestamps]
	predicted_clusters = [list(loadtxt(basedir+"/static/model/predicted_segment_"+timestamp+".txt", delimiter=",", unpack=False)) for timestamp in timestamps]
	posterior_probabilities = [list(loadtxt(basedir+"/static/model/posterior_probabilities_"+timestamp+".txt", delimiter=",", unpack=False)) for timestamp in timestamps]
	#rov_list = [list(loadtxt(basedir+"/static/model/rov_"+timestamp+".txt", delimiter=",", unpack=False)) for timestamp in timestamps]
	rov_list = [dict(csv.reader(open(basedir+"/static/model/rov_"+timestamp+".txt", 'r'))) for timestamp in timestamps]

	if len(set(timestamps)) != len(timestamps):
		print "warning: duplicate timestamps!"
	
	scorecard = zip(timestamps, cluster_numbers, num_vars, model_stats)
	#print sorted(scorecard, key=lambda x: x[3][2])[:10]
	
	# now store all in results_dict
	new_results_dict_keys = 0
	
	for timestamp in timestamps:
		if timestamp not in results_dict.keys():
			results_dict[timestamp] = {}
			new_results_dict_keys +=1

	print new_results_dict_keys, " new results_dict keys added"

	run_num = starting_results_dict_length

	for num, key in enumerate(timestamps):#results_dict.keys():
		results_dict[key]['run_number'] = starting_results_dict_length + num + 1
		results_dict[key]['date'] = time.asctime(time.localtime(time.time()) ) #time.time()
		results_dict[key]['method'] = method
		results_dict[key]['cluster_number'] = cluster_numbers[num]
		results_dict[key]['cluster_counts'] = Counter(predicted_clusters[num]).values()
		results_dict[key]['cluster_shares'] = [ float(cluster) / sum(results_dict[key]['cluster_counts']) for cluster in results_dict[key]['cluster_counts']]
 		results_dict[key]['num_vars'] = num_vars[num]
		results_dict[key]['quest_list'] = quest_list[num]
		results_dict[key]['model_stats'] = model_stats[num]
		results_dict[key]['predicted_clusters'] = predicted_clusters[num]
		results_dict[key]['posterior_probabilities'] = posterior_probabilities[num]
		results_dict[key]['upload_state'] = False
		results_dict[key]['rov'] = rov_list[num]

	print "results_dict size after adding new keys:", len(results_dict.keys())

	# within-segment polarity = ((max bucket share) - average bucket share) for each question and cluster
	# cross-question polarity = max bucket share - average bucket share for each question bucket across clusters
	
	print("------- Runtime: %.2f seconds -------" % (time.time() - start_time))
	
	#results_dict = dict(Parallel(n_jobs=num_cores,verbose=5)(delayed(make_one_results_dict_entry_mp)(i,results_dict.keys()[i],results_dict[results_dict.keys()[i]], X_rebucketed, names) for i in range(len(results_dict.keys()))))
	new_results_dict = dict(Parallel(n_jobs=num_cores,verbose=5)(delayed(make_one_results_dict_entry_mp)(i,timestamps[i],results_dict[timestamps[i]], X_rebucketed, names, question_dict) for i in range(len(timestamps))))
	results_dict.update(new_results_dict)

	print "sample results_dict entry:",results_dict[results_dict.keys()[0]].keys()
	print 'sample results_dict[cluster_counts] entry:', results_dict[results_dict.keys()[0]]['cluster_counts']
	print 'sample results_dict[cluster_shares] entry:', results_dict[results_dict.keys()[0]]['cluster_shares']

	print "results_dict complete with", len(results_dict.keys()), "entries"
	print("------- Runtime: %.2f seconds -------" % (time.time() - start_time))

	save_db(session_id, X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names)

	return timestamps#, results_dict

def make_one_results_dict_entry_mp(i, key, results_dict_entry, X_rebucketed, names, question_dict):
	'''
	Breaks apart results_dict update into threads if multiprocessing possible
	'''
	z = func_name(); print "--------in function: ", z, " -------------"
	print 'X_rebucketed.shape: ', X_rebucketed.shape
	
	one_results_dict_entry = dict(results_dict_entry)
	
	pred_clusters = np.array(one_results_dict_entry['predicted_clusters'])
	pred_clusters = np.reshape(pred_clusters, (pred_clusters.shape[0],1))
	clustered_responders = np.append(X_rebucketed, pred_clusters, axis=1)
	
	df_names = names + ['segments']
	print 'clustered_responders.shape', clustered_responders.shape
	print 'df_names length:', len(df_names)
	print 'df_names', df_names
	print 'set(X_rebucketed[:,-1])', set(X_rebucketed[:,-1])
	print 'set(X_rebucketed[:,-2])', set(X_rebucketed[:,-2])

	df = pd.DataFrame(clustered_responders, columns=df_names)
	df.to_csv('clustered_responders', header=True, sep=',')
	
	one_results_dict_entry['response_shares'] = {}; one_results_dict_entry['response_counts'] = {}; one_results_dict_entry['response_polarity'] = {}; one_results_dict_entry['cross_question_polarity'] = {}
	questions = one_results_dict_entry['quest_list']
	
	#for question in results_dict['quest_list']:
	# for question in question_dict.keys():
	for question in names:
		#print 'now trying question: ', question 
		#print 'this is element: ', names.index(question), ' in names'
		response_shares = []; response_counts = []
		
		for cluster in set(one_results_dict_entry['predicted_clusters']):
			response_count = [sum(clustered_responders[clustered_responders[:,-1] == cluster][:,names.index(question)] == bucket) for bucket in set(clustered_responders[:,names.index(question)])]
			
			# if names.index(question) > 145: #bugcheck
			# 	print 'question:', question
			# 	print 'cluster', cluster
			# 	print 'names.index(question):', names.index(question)
			# 	print 'response_count:', response_count
			# 	print 'set(clustered_responders[:,names.index(question)])', set(clustered_responders[:,names.index(question)])
			# 	print 'set(X_rebucketed[:,names.index(question)])', set(X_rebucketed[:,names.index(question)])
			
			response_counts.append(response_count)
			response_share = [float(response_count[i])/sum(response_count) for i in range(len(response_count))]
			response_shares.append(response_share)

		#print question, "response counts:", response_counts	
		#print question, "response shares:", response_shares	
		one_results_dict_entry['response_counts'][question] = response_counts
		one_results_dict_entry['response_shares'][question] = response_shares
		one_results_dict_entry['response_polarity'][question] = [max(item) - np.mean(item) for item in response_shares]

		# if names.index(question) > 145: #bugcheck
		# 	print 'results_dict[response_shares]', question, ': ', results_dict['response_shares'][question]
		# 	print 'results_dict[cluster_number]: ', results_dict['cluster_number']
		# 	print 'len(question_dict[question][rebucket_counts]: ', len(question_dict[question]['rebucket_counts'])
		# 	print 'question_dict[question][rebucket_counts]: ', question_dict[question]['rebucket_counts']

		#### below line creates error with <>3 buckets
		#cross_question_polarity = np.reshape(results_dict['response_shares'][question],(results_dict['cluster_number'],3))
		
		try:
			cross_question_polarity = np.reshape(one_results_dict_entry['response_shares'][question],(one_results_dict_entry['cluster_number'],len(question_dict[question]['rebucket_counts']))) # multibucket
		except:
			print '----error checking info------'
			print 'question: ', question
			print 'set(results_dict[predicted_clusters]):', set(one_results_dict_entry['predicted_clusters'])
			print 'results_dict[response_counts][question]: ', one_results_dict_entry['response_counts'][question]
			print '# of values in question: set(clustered_responders[:,names.index(question)]):', set(clustered_responders[:,names.index(question)])
			print 'names.index(question):', names.index(question)
			print '------do number of values in question match X_rebucketed manual check?----------'
			print 'X_rebucketed.csv saved in line 597'
			print 'clustered_responders saved in line 1457, col names do not match X_rebucketed.csv!'
			# clustered_responders is given name from names - is this refreshed on load of saved session file?
			print 'results_dict.keys():', one_results_dict_entry.keys()
			print 'question_dict.keys():', question_dict.keys()
			print 'question_dict[question]:', question_dict[question].keys()
			print 'names:', names
			print 'results_dict[response_shares]', question, ': ', type(one_results_dict_entry['response_shares'][question]), one_results_dict_entry['response_shares'][question]
			print 'results_dict[cluster_number]: ', one_results_dict_entry['cluster_number']
			print 'len(question_dict[question][rebucket_counts]: ', len(question_dict[question]['rebucket_counts'])
			print 'question_dict[question][rebucket_counts]: ', question_dict[question]['rebucket_counts']

			'''
			error is inxed >146???
			question: q31_8
			cluster 1.0
			names.index(question): 146
			response_count: [19, 251, 91]
			set(clustered_responders[:,names.index(question)]) set([1.0, 2.0, 3.0])
			set(X_rebucketed[:,names.index(question)]) set([1, 2, 3])
			question: q31_8
			cluster 2.0
			names.index(question): 146
			response_count: [94, 98, 37]
			set(clustered_responders[:,names.index(question)]) set([1.0, 2.0, 3.0])
			set(X_rebucketed[:,names.index(question)]) set([1, 2, 3])
			question: q31_8
			cluster 3.0
			names.index(question): 146
			response_count: [25, 125, 276]
			set(clustered_responders[:,names.index(question)]) set([1.0, 2.0, 3.0])
			set(X_rebucketed[:,names.index(question)]) set([1, 2, 3])
			question: q31_8
			cluster 4.0
			names.index(question): 146
			response_count: [96, 264, 48]
			set(clustered_responders[:,names.index(question)]) set([1.0, 2.0, 3.0])
			set(X_rebucketed[:,names.index(question)]) set([1, 2, 3])
			question: q31_8
			cluster 5.0
			names.index(question): 146
			response_count: [101, 228, 227]
			set(clustered_responders[:,names.index(question)]) set([1.0, 2.0, 3.0])
			'''
		#### above line creates error with <>3 buckets
		
		one_results_dict_entry['cross_question_polarity'][question] = [max(cross_question_polarity[:,col]) - np.mean(cross_question_polarity[:,col]) for col in range(cross_question_polarity.shape[1])]

	print "results_dict[response_shares] length:", len(one_results_dict_entry['response_shares'])
	
	one_results_dict_entry['polarity_scores'] = [sum(one_results_dict_entry['response_polarity'][question][cluster] / len (one_results_dict_entry['quest_list']) for question in one_results_dict_entry['quest_list']) for cluster in range(one_results_dict_entry['cluster_number'])]
	one_results_dict_entry['weighted_average_cluster_polarity'] = sum([a*b for a,b in zip(one_results_dict_entry['polarity_scores'],one_results_dict_entry['cluster_shares'])])
	one_results_dict_entry['mean_cluster polarity'] = np.mean(one_results_dict_entry['polarity_scores'])
	one_results_dict_entry['average_cross_question_polarity'] = sum(sum(value) for value in one_results_dict_entry['cross_question_polarity'].values()) / len(one_results_dict_entry['cross_question_polarity'].values())	

	print "number of entries for results_dict", key, len(one_results_dict_entry.keys())
	#print results_dict
	one_entry = dict(one_results_dict_entry.items())

	print "number of entries for one_entry", key, len(one_entry.keys())

	return key, one_entry

def update_question_dict(timestamps, session_id):#question_dict, results_dict):
	'''
	Updates question_dict after poLCA run, adding back a boolean run inclusion tracker for each question 
	'''
	z = func_name(); print "--------in function: ", z, " -------------"
	X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names = load_db(session_id)

	#global results_dict
	#global question_dict

	z = func_name(); print "--------in function: ", z, " -------------"
	print "number of keys in question_dict:", len(question_dict.keys())
	
	# # initialize question_dict['run_tracker'] with null for all vars, since not all are used
	# q = 0
	# for key in question_dict:
	# 	if 'run tracker' not in question_dict[key].keys(): 
	# 		question_dict[key]['run_tracker'] = []
	# 		question_dict[key]['bool_run_tracker'] = list(np.array([(run+1) in question_dict[key]['run_tracker'] for run in range(len(results_dict))]).astype(int))
	# 		q = q + 1
	# print q, "question_dict[key][run_tracker] initialized"

	# now update question_dict with run results
	counter = 0
	for key in timestamps:
		for question in results_dict[key]['quest_list']:
			# if 'run_tracker' in question_dict[question].keys():
			# 	question_dict[question]['run_tracker'].append(results_dict[key]['run_number'])
			# else:
			# 	question_dict[question]['run_tracker'] = [results_dict[key]['run_number']]

			question_dict[question]['run_tracker'].append(results_dict[key]['run_number'])
			counter +=1
	print "total updates to queston_dict['run_tracker']:", counter
	
	# now create boolean string for all runs (1 = included, 0 = not included)
	for question in question_dict.keys():
		question_dict[question]['bool_run_tracker'] = list(np.array([(run+1) in question_dict[question]['run_tracker'] for run in range(len(results_dict))]).astype(int))
		# should this be outside function to do boolean for all questions, not just those in run?

		# checksum
		if len(question_dict[question]['run_tracker']) != sum(question_dict[question]['bool_run_tracker']):
			print question, "# of runs included:", len(question_dict[question]['run_tracker'])
			print question, "# of bool entries:", len(question_dict[question]['bool_run_tracker'])
			print question, "sum of bool entries", sum(question_dict[question]['bool_run_tracker'])

	save_db(session_id, X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names)
	return #question_dict, results_dict

def make_xls(session_id):#results_dict):
	'''
	Excel export - currently only detailed run results
	'''
	z = func_name(); print "--------in function:", z, "-------------"
	X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names = load_db(session_id)

	results = db.find_one({'session_id': session_id})
	factor_matrix_file_id = results['factor_matrix']
	factor_matrix = pickle.loads(gridfs.get(factor_matrix_file_id).read())

	#global results_dict
	#lobal question_dict
	#global X
	#global X_rebucketed
	#global factor_matrix


	# uses http://xlsxwriter.readthedocs.org/getting_started.html
	outfile = 'clustering_data_export.xlsx'
	workbook = xlsxwriter.Workbook(basedir+outfile)

	tables = ['X', 'X_rebucketed', 'factor_matrix']
	worksheet_names = ['original_data', 'rebucketed_data', 'factor_matrix']
	x_axis_labels = ['names', 'names', 'range(x_cs)']
	y_axis_labels = ['range(x_rs)', 'range(x_rs)', 'names']

	for i,t in enumerate(tables):
		row = 0; col = 0 # reset
		table = eval(t)

		print '----writing .xls table: ', t, '-----'
		worksheet_name = worksheet_names[i]
		worksheet = workbook.add_worksheet(worksheet_name)
		x_rs,x_cs = table.shape # rows, cols
		x_axis_label = eval(x_axis_labels[i])
		y_axis_label = eval(y_axis_labels[i])

		print "table.shape:",  table.shape

		for x_r in xrange(x_rs):
			col = 0

			if x_r == 0:
				for label in x_axis_label: 
					worksheet.write(row, col+1, label)
					col = col +1 
				col = 0
			#col = col + 1
			for x_c in xrange(x_cs):
				if col == 0:
					#print y_axis_label[row]
					worksheet.write(row+1,col,y_axis_label[row])
				data = table[row,col]
				worksheet.write(row+1, col+1, data)
				col = col + 1
			row = row + 1

	run_number = 0

	timestamps = results_dict.keys()
	run_reports = run_report(timestamps, session_id)

	for run in run_reports:
		run_number += 1
		worksheet_name = 'R' + str(run_number)
		worksheet = workbook.add_worksheet(worksheet_name)
		num_columns = len(run[0])
		num_rows = len(run)

		#row = 0
		#col = 0

		for row in range(num_rows):
			for col in range(num_columns):
				data = run[row][col]
				if type(data) is list: # trap lists!
					data = data[0]
				#print data
				#print type(data)
				#print row
				#print col
				worksheet.write(row, col, data)

	workbook.close()
	'''
	for run in results_dict:
		run_number += 1
		worksheet_name = 'R' + str(run_number)
		worksheet = workbook.add_worksheet(worksheet_name)
		number_of_clusters = results_dict[run]['cluster_number'] 
		bold = workbook.add_format({'bold': True, 'align': 'center'})
		percent = workbook.add_format({'num_format': '0.00%', 'align': 'center'})

		row = 0
		col = 0

		worksheet.write(row, col + 0, 'question', bold)
		worksheet.write(row, col + 1, 'bucket', bold)

		for cluster in range(number_of_clusters):
			label = 'cluster ' + str(cluster+1)
			worksheet.write(row, col + cluster + 2, label,bold)

		row += 1

		for survey_question in results_dict[run]['response_shares'].keys():
			worksheet.write(row + 0, col + 0, survey_question, bold)
			worksheet.write(row + 0, col + 1, 'T=1', bold)
			worksheet.write(row + 1, col + 1, 'M=2', bold)
			worksheet.write(row + 2, col + 1, 'B=3', bold)

			for cluster in range(number_of_clusters):
				top, middle, bottom = results_dict[run]['response_shares'][survey_question][cluster]
				worksheet.write(row + 0, col + cluster + 2, top, percent)
				worksheet.write(row + 1, col + cluster + 2, top, percent)
				worksheet.write(row + 2, col + cluster + 2, top, percent)
			row += 4
	'''

	workbook.close()

	print "workbook exported: ", basedir+outfile

	save_db(session_id, X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names)

	return basedir+outfile

def clean_up (timestamps, session_id):
	'''
	Deletes working files from poLCA run (easiest way R>Python)
	'''
	z = func_name(); print "--------in function: ", z, " -------------"
	
	if visual_mode:
		X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names = load_db(session_id)
	
	# now clean up!
	for timestamp in timestamps:
		os.remove(basedir+"/static/model/model_stats_"+timestamp+".txt")
		os.remove(basedir+"/static/model/predicted_segment_"+timestamp+".txt")
		os.remove(basedir+"/static/model/posterior_probabilities_"+timestamp+".txt")
		os.remove(basedir+"/static/model/rov_"+timestamp+".txt")

		if visual_mode:
			num_clusters = results_dict[timestamp]['cluster_number']
			for cluster in range(num_clusters + 1):
				os.remove(basedir + "/static/plots/seg_graph_" + timestamp + "_" + cluster + ".png")

	print len(timestamps) * 4, "scoring files cleaned up from ", len(timestamps), "runs"

	return

def save_results(user_id, session_id):
	'''
	Writes save_dict to file
	'''
	import cPickle as pickle
	import time

	z = func_name(); print "--------in function: ", z, "--------------"
	X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names = load_db(session_id)

	results = db.find_one({'session_id': session_id})

	cluster_seed = results['cluster_seed']
	weights_filename = results['weights_filename']
	data_filename = results['data_filename']

	factor_matrix_file_id = results['factor_matrix']
	print factor_matrix_file_id

	factor_matrix = pickle.loads(gridfs.get(factor_matrix_file_id).read())

	#global results_dict
	#global question_dict
	#global X
	#global X_rebucketed
 	
	time_dict = {}
	load_time_dict = {}

	save_dict = {}
	save_dict['results_dict'] = results_dict
	save_dict['question_dict'] = question_dict
	save_dict['X'] = X
	save_dict['X_rebucketed'] = X_rebucketed
	save_dict['names'] = names
	save_dict['X_rebucketed_df'] = X_rebucketed_df
	save_dict['cluster_seed'] = cluster_seed
	save_dict['weights_filename'] = weights_filename
	save_dict['data_filename'] = data_filename
	save_dict['factor_matrix'] = factor_matrix

	print 'results_dict keys:',  save_dict['results_dict'].keys()
	print 'question_dict keys:', save_dict['question_dict'].keys()
	print 'X shape:', save_dict['X'].shape
	print 'X_rebucketed shape: ', save_dict['X_rebucketed'].shape
	print 'Names:', len(save_dict['names'])
	print 'X_rebucketed_df shape: ', save_dict['X_rebucketed_df'].shape
	print 'cluster_seed length:', len(save_dict['cluster_seed'])
	print 'weights_filename: ', save_dict['weights_filename']
	print 'data_filename: ', save_dict['data_filename']
	print 'factor_matrix.shape: ', save_dict['factor_matrix'].shape
	
	# now load weights, add to save_dict
	weights_filename = 'static/weights_file/X_weights_' + session_id + '.csv'
	X_weights = np.asmatrix(np.genfromtxt(basedir + weights_filename, delimiter=',', skip_header=0))
	save_dict['weights']= X_weights

	#np.savetxt(filename, X_weights, delimiter=',') 


	print os.path.abspath(os.path.dirname(__file__))

	# cPickle
	method = 'cPickle'
	print "trying :", method
	try:
		outfile = 'static/saved_sessions/' + user_id + '/save_dict_' + session_id + '_' + str(len(results_dict.keys())) + '.pkl'
		with open(outfile, 'wb') as handle:
		    pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)		
		print method,": ", outfile, ' saved with ', len(results_dict), ' entries'

		return outfile

		# pickle.load(open(outfile, 'rb'))
	
	except:
		print 'no file saved'
		return
	

def load_results(user_id, filename, session_id):

	'''
	Loads save_dict from file
	'''
	z = func_name(); print "--------in function: ", z, "--------------"

	#global results_dict
	#global question_dict
	#global X
	#global X_rebucketed
	#global names
	#global X_rebucketed_df
 
	import cPickle as pickle
	
	save_dict = {}
	print os.path.abspath(os.path.dirname(__file__))

	print 'user_id:', user_id
	print 'filename', filename

	# cPickle
	method = 'cPickle'
	print "trying :", method
	try:
		infile = 'static/saved_sessions/' + user_id + "/" + filename  
		print infile

		save_dict = pickle.load(open(infile, 'rb'))
		print method,": ", infile, ' loaded with ', len(save_dict), ' entries'

		# now reconstitute dictionaries
		results_dict = save_dict['results_dict']
		question_dict = save_dict['question_dict']
		X = save_dict['X']
		X_rebucketed = save_dict['X_rebucketed']
		names = save_dict['names']
		X_rebucketed_df = save_dict['X_rebucketed_df']

		cluster_seed = save_dict['cluster_seed']
		weights_filename = save_dict['weights_filename']
		data_filename = save_dict['data_filename']
		factor_matrix = save_dict['factor_matrix']
		X_weights = save_dict['weights']

		# now save working data_file and weights_file to their private directories
		data_filename = 'X_rebucketed_' + session_id + '.csv'
		X_rebucketed_df.to_csv(basedir + 'static/data_file/' + data_filename, index=False)
		print data_filename, "saved to:", basedir + 'static/data_file/' 

		weights_filename = 'X_weights_' + session_id + '.csv'
		np.savetxt(basedir + 'static/weights_file/' + weights_filename, X_weights.T, delimiter=',') 
		print weights_filename, "saved to:", basedir + 'static/weights_file/' 

		# reset upload_state for bulk upload
		for key in results_dict.keys():
			results_dict[key]['upload_state'] = False
			print key, results_dict[key]['upload_state']
		
		# update db
		save_db(session_id, X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names) 

		factor_matrix_pickle = Binary(pickle.dumps(factor_matrix, protocol=2), subtype=128 )
		factor_matrix_pickle_file_id = gridfs.put( factor_matrix_pickle )
		db.update_one({'session_id': session_id},{"$set":{'factor_matrix': factor_matrix_pickle_file_id}},upsert=True)

		db.update_one({'session_id': session_id},{"$set":{'cluster_seed':cluster_seed}},upsert=True)
		db.update_one({'session_id': session_id},{"$set":{'weights_filename':weights_filename}},upsert=True)
		db.update_one({'session_id': session_id},{"$set":{'data_filename':data_filename}},upsert=True)

		print 'results_dict keys:',  results_dict.keys()
		print 'question_dict keys:', question_dict.keys()
		print 'X shape:', X.shape
		print 'X_rebucketed shape: ', X_rebucketed.shape
		print 'Names:', len(names)
		print 'X_rebucketed_df shape: ', X_rebucketed_df.shape
		print 'cluster_seed length:', len(cluster_seed)
		print 'weights_filename: ', weights_filename
		print 'data_filename: ', data_filename
		print 'factor_matrix.shape: ', factor_matrix.shape

	except:
		print '----------save_dict load error!!!!!----------------'
		pass

	return


def run_poLCA (grid_search,cluster_seed,num_seg,num_rep, session_id):
	'''
	Runs single or gridsearch, also uses multiprocessing for gridsearch (if possible)
	'''
	z = func_name(); print "--------in function: ", z, " -------------"

	if grid_search == False:
		print '--------in grid_search == False function--------------'
		timestamp, cluster_seeds, cluster_seed_names, num_seg = poLCA (1, cluster_seed,num_seg,num_rep,session_id)
		scorecard(timestamp, cluster_seeds, cluster_seed_names, num_seg, session_id)
		#timestamps = [timestamp]
		results = [(timestamp,cluster_seeds,cluster_seed_names,num_seg)]
		print "results:", results

	if grid_search == True:
		print '---------in grid search == True function---------'
		# run clustering in parallel if possible
		shortened_cluster_seeds = []
		num_remove = 3
		random.shuffle(cluster_seed)
		
		shortened_cluster_seeds = [cluster_seed[0:(len(cluster_seed)-num)] for num in range(num_remove+1) ]
		print "shortened cluster seed:", shortened_cluster_seeds

		# shortened_cluster_seeds = [x for x in itertools.combinations(cluster_seed, 28)]
		# print "total cluster seeds: ", len(shortened_cluster_seeds)

		num_seg = [5,6,7,8,9,10,11,12]#,13,14]
		
		num_cores = multiprocessing.cpu_count()
		#results = Parallel(n_jobs=num_cores,verbose=5)(delayed(poLCA)(i,cluster_seed, num_seg[i], num_rep, rebucketed_filename) for i in range(10))
		
		#scorecard(cluster_seeds, cluster_seed_names, num_seg)
		# see http://blog.dominodatalab.com/simple-parallelization/
		print "running grid search (seg x variables removed):", len(num_seg), ", ", num_remove
		results = Parallel(n_jobs=num_cores,verbose=5)(delayed(poLCA)(i,shortened_cluster_seeds[j], num_seg[i], num_rep, session_id) for i in range(len(num_seg)) for j in range (len(shortened_cluster_seeds)))
		#print results
	
	print "number of runs completed:", len(results)
	return results

def func_name():
	import traceback
	return traceback.extract_stack(None, 2)[0][2]

def run_report(timestamps, session_id):
	'''
	Detailed report from single segmenting run
	'''
	z = func_name(); print "------in function:", z, "---------------"
	X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names = load_db(session_id)
	
	#global results_dict
	#global question_dict

	#  sample results_dict entry keys: ['cluster_shares', 'run_number', 'model_stats', 'response_shares', 'posterior_probabilities', 'cluster_number',
	# 'cross_question_polarity', 'quest_list', 'predicted_clusters', 'response_counts', 'cluster_counts', 'polarity_scores', 'upload_state', 
	# 'mean_cluster polarity', 'num_vars', 'response_polarity', 'average_cross_question_polarity', 'weighted_average_cluster_polarity']

	print "timestamps: ", timestamps

	run_reports = []
	for run in timestamps:
		print "run id: ", run

		number_of_clusters = results_dict[run]['cluster_number'] 
		run_report = []	

		# header: question...bucket...segment_1,...segment_2...etc
		segment_list = 	 ['S' + str((num + 1)) for num in range(number_of_clusters)]

		segment_counts = ['# in seg',   '',''			   ]     + results_dict[run]['cluster_counts']  + [sum(results_dict[run]['cluster_counts'])] + [''] + ([''] * len(segment_list))
		segment_shares = ['Pct in seg', '',''			   ]     + ['{0:.0%}'.format(results_dict[run]['cluster_shares'][cluster]) for cluster in range(number_of_clusters)] + ['{0:.0%}'.format(sum(results_dict[run]['cluster_shares']))] + [''] + ([''] * len(segment_list)) 
		header_1 = 		 ['', '', ''			 		   ]     + segment_list + ['Total']        + [''] + ([''] * len(segment_list)) 
		header_2 = 		 ['Question', 'Dimension', 'Bucket'] + segment_list + ['Average'] + [''] + segment_list

		row_spacer = [''] * (len(header_2))
		
		header = [header_1] + [segment_counts] + [segment_shares] + [row_spacer] + [header_2] + [row_spacer] + [row_spacer]
		run_report += header


		# order responses: segmenting variables, objective functions, other
		clustering_variables = results_dict[run]['quest_list']

		print 'clustering_variables:', clustering_variables
		
		all_questions = results_dict[run]['response_shares'].keys()
		d = ['n/a']
		dimension_vars = [q for q in all_questions if question_dict[q]['dimension'] not in d and q not in clustering_variables]
		dimension_vars_text = [question_dict[q]['dimension'] for q in dimension_vars]
		dimension_variables = [a for a,b in sorted(zip(dimension_vars, dimension_vars_text), key = lambda z: z[1])]
		non_clustering_variables = [q for q in all_questions if q not in clustering_variables and q not in dimension_variables]
		ordered_variables = sorted(clustering_variables) + dimension_variables + sorted(non_clustering_variables)

		print 'ordered_variables: ', ordered_variables

		#for survey_question in results_dict[run]['response_shares'].keys():
		for survey_question in ordered_variables:
			
			# new method starts here
			number_of_buckets = len(results_dict[run]['response_shares'][survey_question][0])
			dimension = question_dict[survey_question]['dimension']

			for bucket in range(number_of_buckets):
				bucket_data = [results_dict[run]['response_shares'][survey_question][cluster][bucket] for cluster in range(number_of_clusters)]
				#bucket_shares = ['{0:.1%}'.format(item) for item in bucket_data]
				bucket_shares = ['{0:.0%}'.format(item) for item in bucket_data]
				bucket_average = sum([float(x)*float(y) for x,y in zip(bucket_data,results_dict[run]['cluster_shares'])])
				bucket_avg = ['{0:.1%}'.format(bucket_average)]
				bucket_index_scores = [int((float(share) / bucket_average) * 100) for share in bucket_data]
				rov = results_dict[run]['rov'][survey_question].split('.')[0]
				#print 'rov_ty[e:', type(rov)

				if bucket == 0:
					row =    [survey_question, dimension, bucket] + bucket_shares + bucket_avg + [''] + bucket_index_scores
				else:
					row = 	 ['', '', 		              bucket] + bucket_shares + bucket_avg + [''] + bucket_index_scores
				
				run_report.append(row)

				if bucket + 1 == number_of_buckets: # last bucket
					rov_row = ['', 'ROV: ', [rov]] + [''] * (len(row)-3)
					row_spacer = [''] * (len(row))
					run_report.append(rov_row)
					run_report.append(row_spacer)

			if survey_question == sorted(clustering_variables)[-1]: # and len(clustering_variables) == clustering_variables.index(survey_question):
				row_spacer = [''] * (len(row))
				run_report.append(row_spacer)
				run_report.append(row_spacer)
				run_report.append(row_spacer)	

			'''

			#print "result_dict entry: ", survey_question, len(results_dict[run]['response_shares'][survey_question]), results_dict[run]['response_shares'][survey_question]
			top_bucket_data = [results_dict[run]['response_shares'][survey_question][cluster][0] for cluster in range(number_of_clusters)]
			top_bucket = ['{0:.1%}'.format(item) for item in top_bucket_data]
			top_bucket_average = sum([float(x)*float(y) for x,y in zip(top_bucket_data,results_dict[run]['cluster_shares'])])
			top_bucket_avg = ['{0:.1%}'.format(top_bucket_average)]
			top_bucket_index_scores = [int((float(share) / top_bucket_average) * 100) for share in top_bucket_data]
			#print "top bucket:", top_bucket
			#print 'top bucket weighted average:', top_bucket_average
			#print "top bucket index scores:", top_bucket_index_scores

			middle_bucket_data = [results_dict[run]['response_shares'][survey_question][cluster][1] for cluster in range(number_of_clusters)]
			middle_bucket = ['{0:.1%}'.format(item) for item in middle_bucket_data]
			middle_bucket_average = sum([float(x)*float(y) for x,y in zip(middle_bucket_data,results_dict[run]['cluster_shares'])])
			middle_bucket_avg = ['{0:.1%}'.format(middle_bucket_average)]
			middle_bucket_index_scores = [int((float(share) / middle_bucket_average) * 100) for share in middle_bucket_data]

			bottom_bucket_data = [results_dict[run]['response_shares'][survey_question][cluster][2] for cluster in range(number_of_clusters)]
			bottom_bucket = ['{0:.1%}'.format(item) for item in bottom_bucket_data]
			bottom_bucket_average = sum([float(x)*float(y) for x,y in zip(bottom_bucket_data,results_dict[run]['cluster_shares'])])
			bottom_bucket_avg = ['{0:.1%}'.format(bottom_bucket_average)]
			bottom_bucket_index_scores = [int((float(share) / bottom_bucket_average) * 100) for share in bottom_bucket_data]
			
			#top_bucket_average = sum([float(x)*float(y) for x,y in zip(results_dict[run]['cluster_shares']])
			#print 'top bucket weighted average:', top_bucket_average

			# now format responses ('x' to be replaced by space)
			top_row =    [survey_question,'T=1'] + top_bucket   + top_bucket_avg    + [''] + top_bucket_index_scores
			middle_row = ['',            'M=2'] + middle_bucket + middle_bucket_avg + [''] + middle_bucket_index_scores
			bottom_row = ['',            'B=3'] + bottom_bucket + bottom_bucket_avg + [''] + bottom_bucket_index_scores
			row_spacer = [''] * (len(bottom_row))
			
			# make each row a dict, append dicts to run_report
			key_list = range(number_of_clusters + 2)
			# run_report.append(dict(zip(key_list,top_row)))
			# run_report.append(dict(zip(key_list, middle_row)))
			# run_report.append(dict(zip(key_list,bottom_row)))
			# run_report.append(dict(zip(key_list,row_spacer)))

			run_report.append(top_row)
			run_report.append(middle_row)
			run_report.append(bottom_row)
			run_report.append(row_spacer)
			'''

		run_reports.append(run_report)

	print 'total length of run_reports:', len(run_reports)
	print 'total length of one run_report', len(run_reports[0])
	print 'header for one run_report:', run_reports[0][0]
	print 'sample entry for one question in one run_report:', run_reports[0][1]
 	
 	save_db(session_id, X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names)

	return run_reports

def get_segments(X_rebucketed_df, names, cluster_seed, method, num_seg, session_id):
	'''
	Performs range of clustering methods 
	'''
	z = func_name(); print "------entering function:", z, "---------------"
	X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names = load_db(session_id)

	cluster_seed_names = [names[int(value)] for value in cluster_seed]
	timestamp = str(uuid.uuid4())

	#number_clusters = 10

	print "X_rebucketed_df_columns(): ", list(X_rebucketed_df)
 	# http://stackoverflow.com/questions/19482970/get-list-from-pandas-dataframe-column-headers
	question_names = X_rebucketed_df.columns.values.tolist()
	cluster_seed_names = [names[int(seed)] for seed in cluster_seed]
	# print "cluster_seed_names: ", cluster_seed_names

	methods = [AgglomerativeClustering(linkage='ward'), MeanShift(), DBSCAN(eps = 0.5, min_samples = 2, metric ='euclidean'), KMeans(n_clusters=num_seg, init='random'), AffinityPropagation(), Birch(n_clusters=num_seg), SpectralClustering(n_clusters=num_seg, n_init=10)]
	keys = ['agclust', 'meanshift', 'dbscan', 'kmeans', 'affinityprop', 'birch', 'spectral']
	methods_dict = dict(zip(keys,methods))

	clusters = []
	m = methods_dict[method]

	m.fit(X_rebucketed_df[cluster_seed_names])
	
	clusters = m.labels_.tolist()
	clusters = [cluster + 1 for cluster in clusters]

	# DBSCAN labels outliers -1; MeanShift, AC, AP all automatically pick num_seg
	print "clusters: ", set(clusters)
	num_seg = len(set(clusters)) - (1 if -1 in clusters else 0)
	
	if num_seg == 0: # error trap
		num_seg +=1

	# now write resuts out for sharing (backwards compatability)
	# predicted segments
	filename = 'predicted_segment_' + timestamp + '.txt'
	basedir_for_segs = os.path.join(basedir, 'static/model/')
	
	with open(basedir_for_segs + filename,'w') as f:
		f.writelines( "%s\n" % item for item in clusters)
		f.close()
	print 'clusters file written to:', basedir_for_segs + filename

	# posterior probabilities
	probability_list = [[float(1) / num_seg] * num_seg] * X_rebucketed_df.shape[0]
	print probability_list[:10]
	filename = 'posterior_probabilities_' + timestamp + '.txt'
	
	with open(basedir_for_segs + filename,'wb') as g:
		writer = csv.writer(g)
		writer.writerows(probability_list)
	print 'fake posterior_probabilities file written to:', basedir_for_segs + filename	

	# model stats [for poLCA = (MLE, Chi-sq, BIC); could be updated for other methods]
	model_stats_list = [1,1,1]
	filename = 'model_stats_' + timestamp + '.txt'
	
	with open(basedir_for_segs + filename, 'w') as f:
		f.writelines( "%s\n" % item for item in model_stats_list)
	print 'fake model_stats file written to:', basedir_for_segs + filename	

	# fake ROV
	fake_rov = [1] * len(question_names)
	fake_rov_list = zip(question_names, fake_rov)
	filename = 'rov_' + timestamp + '.txt'

	with open(basedir_for_segs + filename, 'w') as f:
		f.writelines( "%s,%s\n" % (item[0],item[1]) for item in fake_rov_list)
	print 'fake rov file written to:', basedir_for_segs + filename	

	results = [(timestamp,cluster_seed,cluster_seed_names,num_seg)]

	z = func_name(); print "------exiting function:", z, "---------------"
	save_db(session_id, X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names)

	return results

def feature_importances(X_rebucketed_df, timestamps, session_id):
	'''
	Calculates relative contribution of features using DecisionTreeClassifier
	'''
	z = func_name(); print "------in function:", z, "---------------"
	X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names = load_db(session_id)

	rfe_list = []

	for timestamp in timestamps:
		cluster_seed_names = results_dict[timestamp]['quest_list']
		print "cluster_seed_names: ", cluster_seed_names
		
		clusters = results_dict[timestamp]['predicted_clusters']
		print "clusters:", len(clusters)
		
		#clusters_df = pd.DataFrame(clusters, columns = ['cluster'], index = range(len(clusters))
		
		X2 = np.array(X_rebucketed_df[cluster_seed_names])#.join(clusters_df)

		rfe_scores = []
		for n in range(len(cluster_seed_names)):
			top_n = n + 1
			clf = DecisionTreeClassifier()
			rfe = RFE(clf, top_n).fit(X2, clusters)
			rfe_score = rfe.score(X2,clusters)
			var_names = list(np.array(cluster_seed_names)[np.array(rfe.support_)])
			rfe_scores.append(rfe_score - sum(rfe_scores))

			print 'RFE Mean accuracy: {0:.2%}'.format(rfe_score)

			if rfe_score > .99:
				#rfe_scores = ['{0:.2%}'.format(item) for item in rfe_scores]
				print type(var_names)
				print type(rfe_scores)
				one_rfe_list = zip(var_names, rfe_scores)
				print type(one_rfe_list)

				one_rfe_list.sort(key=lambda x: x[1], reverse=True)
				print type(one_rfe_list)

				one_rfe_list = [(a,'{0:.2%}'.format(b)) for a,b in one_rfe_list] 
				one_rfe_list.append(('Total (>99%)', '{0:.2%}'.format(sum(rfe_scores))))

				one_rfe_list.insert(0, ('---- Question ----', '----  Accuracy (higher = better) ----'))
				one_rfe_list.insert(0, ('------------------', '-------------------------------------'))
				one_rfe_list.insert(0, ('  ','  '))

				# rfe_scores.append(sum(rfe_scores))
				# rfe_scores = ['{0:.2%}'.format(item) for item in rfe_scores]
				# rfe_scores.insert(0, '  ')
				# rfe_scores.insert(0, '----  Accuracy (higher = better) ----')
				
				# print var_names
				# one_rfe_list = zip(var_names, rfe_scores)
				#one_rfe_list  = one_rfe_list.sort(key=lambda x: x[1])
				rfe_list.append(one_rfe_list)
				
				print one_rfe_list
				
				break

	save_db(session_id, X, X_rebucketed, X_rebucketed_df, results_dict, question_dict, names)			
	return rfe_list

def make_visual(i, X_rebucketed_df, timestamp):
	'''
	makes 3-d visual using PCA 
	'''
	z = func_name(); print "------in function:", z, "---------------"
	
	# visualize in 3D
	# 	https://zulko.wordpress.com/2012/09/29/animate-your-3d-plots-with-pythons-matplotlib/
	
	question_names = X_rebucketed_df.columns.values.tolist()
	#print timestamps
	# print "X_rebucketed_df.columns.values:", question_names
	


	#new_results_dict = dict(Parallel(n_jobs=num_cores,verbose=5)(delayed(make_one_results_dict_entry_mp)(i,timestamps[i],results_dict[timestamps[i]], X_rebucketed, names) for i in range(len(timestamps))))

	#for timestamp in timestamps:

	cluster_seed_names = results_dict[timestamp]['quest_list']
	print "cluster_seed_names: ", cluster_seed_names

	number_clusters = results_dict[timestamp]['cluster_number']
	print "number of clusters: ", number_clusters
	
	number_components = 3
	pca = PCA(n_components=number_components)
	c = pca.fit(X_rebucketed_df[cluster_seed_names].T).components_

	clusters = results_dict[timestamp]['predicted_clusters']
	print "clusters:", len(clusters)

	clusters_array = np.array(clusters)
	clusters_array_2 = clusters_array.reshape(len(clusters),1)
	print c.shape
	print clusters_array_2.shape
	d = np.concatenate((c.T, clusters_array_2), axis=1)
	print d.shape
	df = pd.DataFrame(d[:,:3])
	df.to_csv('graph_dataset_' + str(timestamp) +'.csv')
	
	# generate list of random colors for graph		
	#color_dict = {0:'red',1:'blue',2:'green',3:'black',4:'yellow', 5:'pink', 6:'orange',7:'grey', 8:'brown', 9:'purple', 10:'indigo', 11:'light blue', 12:'light green'}
	colors = [np.random.rand(3,) for x in range(number_clusters + 1)]

	#3D plot
	fig = plt.figure(figsize=(3.5,3.5), tight_layout=True)
	ax = fig.add_subplot(111, projection='3d')
	lines = ax.plot(d[:,0],d[:,1],d[:,2],'o', markersize=7, color='grey', alpha=0.05)
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	ax.set_zticks([])

	for num in range(number_clusters + 1):
		lines2 = ax.plot(d[d[:,3]==num][:,0],d[d[:,3]==num][:,1],d[d[:,3]==num][:,2],'o', markersize=7, color=colors[num], alpha=0.3)
		filename = 'static/plots/seg_graph_' + str(timestamp) + '_' + str(num) + '.png'
		fig.savefig(filename)
		print 'file saved: ', filename
		#print ax.lines
		ax.lines.pop(1)
		#print ax.lines		
	
	plt.close(fig)

		#timestamp_and_seg_number = [timestamp, number_clusters]
		#socketio.emit('newnumber', {'number': timestamp_and_seg_number}, namespace='/test')
		# plt.show()

		#2D plot
		#for num in range(cluster):
	    #   lines2 = ax.plot(d[d[:,3]==num][:,0],d[d[:,3]==num][:,1],'o', markersize=7, color=colors[num], alpha=0.5)#, label = labels)
		#	ax.lines.pop(1)
	
	#timestamps_and_seg_nums = []
	#number_of_segments = [results_dict[timestamp]['cluster_number'] for timestamp in timestamps]
	#for i in range(len(timestamps)):
	#	timestamps_and_seg_nums.append(timestamps[i])
	#	timestamps_and_seg_nums.append(number_of_segments[i])
	
	#print timestamps_and_seg_nums

	#number_of_segments = [str(results_dict[timestamp]['cluster_number']) for timestamp in timestamps]
	#print 'number_of_segments:', number_of_segments
	
	#timestamps_and_seg_nums = dict(zip(timestamps, number_of_segments))
	#print timestamps_and_seg_nums = [[timestamp[i]]+[number_of_segments[i]] for i in len(timestamps)]
	
	#print 'timestamps:', timestamps, type(timestamps)
	#print 'str(timestamps:', str(timestamps)
	#print 'json.dumps(timestamps):', json.dumps(timestamps)
	#timestamps = str(timestamps)
	#timestamps_2 = timestamps.replace("'","")
	#print timestamps_2
	#timestamps_3 = "'{0}'".format(timestamps_2)
	#print timestamps_3, type(timestamps_3)
	#timestamps = [str(len(timestamps))] + timestamps
	#number = dict(zip(range(len(timestamps)),timestamps))
	#timestamps_2 = "'" + timestamps + "'"
	#number = 999999999 #json.dumps(timestamps)
	#print 'sleeping'
	#time.sleep(20)
	
	# numbers = []
	# for i in range(3):
	# 	number = str(uuid.uuid4())
	# 	numbers.append(number)
	# numbers = str(numbers)
	# print 'numbers:', numbers

	#socketio.emit('newnumber', {'number': '[9c9e7b59-1a6e-4065-9690-46e290bb51a7, c558e84d-ce89-4409-95f6-94e487911cd7, 10b77204-fd8e-45eb-9fba-0822df9a7633]'}, namespace='/test')
	#socketio.emit('newnumber', {'number': timestamps}, namespace='/test')
	#socketio.emit('newnumber', {'number': str(timestamps_2)}, namespace='/test')
	#socketio.emit('newnumber', {'number': number}, namespace='/test')
	#socketio.emit('newnumber', {'number': timestamps_and_seg_nums}, namespace='/test')
	#socketio.emit('clusternum', {'cluster': str(number_of_segments)}, namespace='/test')

	return

def calculate_weights(responder_infilename):
	#filename = '/Users/pniessen/Rosetta_Desktop/Segmentation_2-point-0/segmentr/web/test_weights_v1.csv'

	responder_df = pd.DataFrame.from_csv(responder_infilename)
	responder_data = np.asarray(responder_df)
	#responder_data = np.genfromtxt(filename, delimiter=',', skip_header=1)
	print 'responder data loaded: ', responder_data.shape

def calculate_weights(responder_data):

    weights = []

    # target populations (source: US Census)
    target_age_categories = ['18-24','25-34','35-44','45-54','55-65']
    target_age_distribution = [.15,.22,.20,.21,.22]
    target_age_dict = dict(zip(target_age_categories, target_age_distribution))
    print target_age_dict

    target_gender_categories = ['Male','Female']
    target_gender_distribution = [.4979, .5021]
    target_gender_dict = dict(zip(target_gender_categories, target_gender_distribution))
    print target_gender_dict

    # calculate age_share, gender_share
    responder_age_shares = [float(sum(responder_data[:,0]==c1))/len(responder_data) for c1 in target_age_categories]
    responder_age_dict = dict(zip(target_age_categories, responder_age_shares))
    responder_gender_shares = [float(sum(responder_data[:,1]==c2))/len(responder_data) for c2 in target_gender_categories]
    responder_gender_dict = dict(zip(target_gender_categories, responder_gender_shares))
    print responder_age_shares, responder_gender_shares
    print responder_age_dict, responder_gender_dict

    for responder in range(len(responder_data))[:10]:
        print responder,
        responder_age = responder_data[responder, 0]
        responder_gender = responder_data[responder,1]
        one_weight = (responder_age_dict[responder_age]/ target_age_dict[responder_age])* (responder_gender_dict[responder_gender] / target_gender_dict[responder_gender]) 
        print one_weight, 
        weights.append(one_weight)

    # now save file
    weights_outfilename = 'test_weights_X_v1.csv'
    with open(basedir + weights_outfilename,'w') as f:
    	f.writelines("%s\n" % item for item in weights)
    	f.close()
    
    print 'weights file written to:', weights_outfilename	
    
    return 


if __name__ == "__main__":

	#app = Flask(__name__)
	num_seg = 5
	num_rep = 1
	#n_components = 30
	top_n = 2
	num_cores = multiprocessing.cpu_count()
	results_dict = {}
	question_dict = {}
	cluster_seed = []
	method = 'poLCA'
	timestamp = 'test123'
	session_id = 'test_test_test_test'

	#basedir = '/Users/pniessen/Rosetta_Desktop/Segmentation_2-point-0/sample_case_work/GoPro/'
	# filename = 'test_raw_data_v1.csv' 
	# file should be (user x question) matrix .csv with question labels in first row

	# setup
	# basedir, basedir_for_upload, handler, xls, interactive_mode, web_mode, filename, grid_search, start_time = init()
	
	# data loading / cleaning pipeline
	# cleaning (module TBA)
		# -          Single Box
		# -          3 scale point check
		# -          High bucketed (Agree / Disagree)
		# -          High bucketed (Neutral)
		# -          Speed Demons
		# -          Outlier Check
		# -          Logic check
		# -          Final cleaning flag
	# weighting (module TBA)

	# pre-processing pipeline:
	#factor_matrix, names, X, question_dict, results_dict = get_PCA(filename, weights_filename, session_id)
	#factor_matrix, num_rows, num_cols, question_number, rh, best_factor, second_best_factor, question_dict = top_n_factors(factor_matrix, top_n, question_dict, names, session_id)
	#rebucketed_filename, X_rebucketed, question_dict, X_rebucketed_df, rh = rebucket(factor_matrix, names, X, rh, question_dict, session_id)
	#cluster_seed = make_cluster_seed(factor_matrix, best_factor, question_number, num_cols, num_rows, rh, session_id)
	# get_segments(X_rebucketed_df, names, cluster_seed, method, timestamp, num_seg)
	# make_visual(X_rebucketed_df, cluster_seed_inbound, timestamps, names)
	# rfe_list = feature_importances(X_rebucketed_df, timestamps, cluster_seed_names)
	# app.run()

	#print "cluster_seed: ", cluster_seed

	if grid_search:
	#analysis and reporting pipeline
		results = run_poLCA (grid_search,cluster_seed,num_seg,num_rep,session_id)#,filename)	
		timestamps = update_results_dict(results, X_rebucketed, names, method, session_id)
		update_question_dict(timestamps, session_id)
		clean_up(timestamps, session_id)
		save_results()
		run_report(timestamps, session_id)
	
	print("------- Runtime: %.2f seconds -------" % (time.time() - start_time))

	# feature switches
	if xls:
		make_xls(session_id)

	if interactive_mode:
		# app.debug = True
		# socketio.run(app)
		app.run(debug = True, threaded=True)
		# http://stackoverflow.com/questions/14814201/can-i-serve-multiple-clients-using-just-flask-app-run-as-standalone
		
	if web_mode:
		app.run(host='0.0.0.0', port=80)
	

	# issue backlog:
	# (x) handoff from Python > R
	# (x)	specify which variables to use in poLCA clustering
	# (x) parse return data from PoLCA - predicted clusters for each observation?
	# (x) remove col 32-35 reference, develop "question_number", "best_factor", "second_best_factor", "rh" labels
	# (x) send parameters to poLCA: number of clusters, number of iterations, etc
	# (x) add predicted posterior to return data, calculate cluster size counts and mean predicted posterior for each cluster
	# (c) i/o to same directory
	# (c) add AIC, BIC to model scorecard
	# (x) bucketing / formatting for poLCA custom input
	# (x) test different number of factors (!= 32)
	# (x) web front end beta
	# (x) reorder run_tracker
	# (x) add switches to control panel
	# (x) migrate start to top nav bar
	# (x) reorder run_tracker
	# (x) add switches to control panel
	# (x) migrate start to top nav bar
	# (x) multiprocessing for result_dictionary creation?
	# (x) user option: save .xls files
	# (x) broaden gridsearch with combinations (Y take X)
	# (x) drag / drop question chooser (with "rewind" feature to run number X) ("this run is x% similar")
	# (x) populate explorer with actual question names
	# (x) detect which questions are checked and/or selected in explorer
	# (x) progress bar for data loading
	# (x) update run_tracker by column baed on results_dict state change (how to handle batch mode?)
	# (x) other param connectivity (num_clusers, grid_search, xls)
	# (x) conditional formatting for run_report
	# (x) upload = empty file blocker
	# (x) 3-d visualization: 3D (users, PCA(questions)(:3), clusters)
	# (x) loading factor matrix - audit vs SPSS
	# (x) varimax rotation 
	# (x) objective function selector, connection to DB
	# (x) clean up /plots subdirectory on lauch?
	# (x) close matplotlib.plt files (memory issues - RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).)
	# (x) JSON.stringify for flask i/o
	# no duplicate question_ids in objective function (one or across objective functions - reduce left side by all elements of objective_functions)
	# drag back 
	# validation traps - # of variables > num_seg, etc
	# % of variance 3D PCA explains
	# audit how weights are used in rebucketing - X passed in, currently unweighted (2/25/2016)
	# draggable / tiled run_report?
	# reorder run_report with segmenting variables at top
	# refactor run_report boolean - add to end, not compeltely regenerate each run
	# grid search = largest
	# mongo db for grid search = large
	# other data types (continuous, dichotomous)
	# coloring of questions by factor relationships?
	# on mouseover support for run_tracker KPI?
	# input file validation - len(set(questions)) == len(questions)
	# confirm question assignment consistent (dict arbitrary order doesn't create errors in table)
	# logic check: num_quetions > num_segments
	# control panel: option for other methods, knn, DBSCAN, tensorflow
	# control panel: small vs large gridsearch
	# 2-d plot of runs - cohesive (X) vs differentiation (Y)?
	# timestamp resuls_dict and question_dict for web service to multiple cases?
	# Cython for frequently used modules?
	# data cleanup
	#	- time: (hcapturetotaltime	hedittotaltime	hmanagetotaltime	hsharetotaltime	henjoytotaltime	hpersonaltotaltime	hadditionaltotaltime	htotaltimeinminutes)
	#	- outlier (>3 standard deviations)
	# 	- logic check (= mutually contradictory questions)
	# 	- response patterns: single box, 3 scale point check, High Bucketed (Agree/disagree, neutral), Speed demon
	# 	- basics: unique question names, unique reponder IDS 
	# data weighting
	# how to rebucket binary and continuous data (vs 1-7)
	# audit trail - save X, X_rebucketed, factor_matrix, run_tracker, results_tracker to disk?
	# visual - radial graph with nearest neighbors?
	# recommendation - top X%?
	# synchronize explorer and run tracker checkboxes
	# audit boolean_run_tracker
	# compare run_reports: (a) convert label to integer; (b) formatting of tables in modal; (c) enlarge modal dynaically 
	# http://www.abeautifulsite.net/jquery-file-tree/#demo





