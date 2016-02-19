# Rosetta Consulting customer segmentation pipeline
# 1.20.2016
# Peter Niessen

"""
Segmentr: a customer segmentation toolkit
"""

import numpy as np
from sklearn.decomposition import PCA
import time
import flask
import pandas as pd
import os
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
from werkzeug import secure_filename
from collections import Counter, OrderedDict
import pprint
from numpy import loadtxt
from joblib import Parallel, delayed
import multiprocessing
import random
import xlsxwriter
import sys, getopt
import itertools

#def init():
# default options
interactive_mode = False
xls = False
web_mode = False
filename = 'test_raw_data_v1.csv' # (user x question) .csv with question labels in first row
grid_search = False

# now check for command line arguments + options
argv = sys.argv[1:]
print "Command line arguments:", str(argv)

try:
	opts, args = getopt.getopt(argv,"hiwxlsf:g",)
except getopt.GetoptError:
	print 'PCA_tester.py -i (interactive_mode), -w (web_mode), -xls (excel file export), -f: infile_name.csv'
	sys.exit()

for opt, arg in opts:
	if opt == '-h':
		print 'PCA_tester.py -i (interactive_mode), -w (web_mode), -xls (excel file export), -f: infile_name.csv'
		sys.exit()
	elif opt in ('-i','-I'):
		interactive_mode = True
	elif opt in ('-w', '-W'):
		web_mode = True
	elif opt in ('-xls', '-xl','-x'):
		xls = True
	elif opt in ('-g', '-G',):
		grid_search = True
	elif opt == '-f':
		# http://stackoverflow.com/questions/5899497/checking-file-extension
		if arg.lower().endswith(('.csv')):
			filename = arg[1:]
			print "infile name:", filename
		else:
			print "wrong file type"
			sys.exit()

app = Flask(__name__)
start_time = time.time()

basedir_for_upload = os.path.abspath(os.path.dirname(__file__))
basedir = os.path.abspath(os.path.dirname(__file__)) + "/"
print basedir_for_upload

from logging import Formatter, FileHandler
handler = FileHandler(os.path.join(basedir_for_upload, 'log.txt'), encoding='utf8')
handler.setFormatter(
    Formatter("[%(asctime)s] %(levelname)-8s %(message)s", "%Y-%m-%d %H:%M:%S")
)
app.logger.addHandler(handler)

# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'csv', 'xls'])

# return basedir, basedir_for_upload, handler, xls, interactive_mode, web_mode, filename, grid_search, app, start_time



# Step 1: load datasets, namess
# load datafile

def get_PCA(filename, n_components):
	'''
	Uses sclearn.decomposition for principal componets analysis of survey response data
	'''
	z = func_name(); print "in function:", z
	print "-----------in get_PCA function----------------"
	
	global question_dict
	
	names = list(np.genfromtxt(basedir+filename, delimiter=',', names=True).dtype.names)
	
	for question_name in names:
		question_dict[question_name] = {}
		question_dict[question_name]['question_text'] = "n/a"
		question_dict[question_name]['dimension'] = "n/a"
		question_dict[question_name]['run_tracker'] = []
	print "question_dict initialized with ", len(question_dict.keys()), " questions"

	print names

	X = np.genfromtxt(basedir+filename, delimiter=',', skip_header=1)

	print filename, "loaded"
	print X.shape
	print X[0] # print one row
	#print X.dtype.names

	# transpose to get features on X axis and respondents on Y ax
	#X = X.T
	#print X.shape

	# sameple array for testing
	#X = np.array([[-1, -1, 2, 5], [-2, -1, 5, 7], [-3, -2, 6, 1], [1, 1, 4, 5], [2, 1, 5, 9], [3, 2, 6, 5]])

	# step 2: Principal Components Analysis

	pca = PCA(n_components=n_components)
	pca.fit(X)

	print "number of components:" , pca.n_components_
	print("Runtime: %s seconds ---" % (time.time() - start_time))
	#print pca.components_.shape

	# transpose to get compontents on X axis and questions on Y
	global factor_matrix
	factor_matrix = pca.components_.T
	print factor_matrix.shape
	print "sample row: ", factor_matrix[0]
	print "max absolute value:", np.absolute(factor_matrix)[0].max()

	return factor_matrix, names, X#, question_dict

def top_n_factors (factor_matrix, top_n):
	'''
	Uses PCA to group survey questions by PCA factors: 
		(a) rank order questions by largest PCA factor (factor_matrix) 
		(b) take largest (top_n)
	'''
	z = func_name(); print "in function:", z
	print "-----------in top_n_factors function----------------------"
	global question_dict
	
	# factor_matrix, n, question_dict = factor_matrix, top_n, question_dict
	# Step 3: identify largest PC factor for each question
	# add two rows of zeros for top 2 factors
	# np.append(a, z, axis=1)
	# stackoverflow.com/questions/8486294/how-to-add-an-extra-column-to-an-numpy-array

	# add extra cols for row (=question) index, #1 factor, #2 factor, best_rh
	factor_matrix_size = factor_matrix.shape # = C x R
	print factor_matrix_size
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
		#print "before:", one_question_vector
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
	#print question_dict

	print("Runtime: %s seconds ---" % (time.time() - start_time))

	return factor_matrix, num_rows, num_cols, question_number, rh, best_factor, second_best_factor#, question_dict

def rebucket(factor_matrix, names, X, rh):
	'''
	Tries different bucketing techniques: [(2,3,2),(3,2,2), (2,2,3), (3,1,3), (1,3,3), (1,2,4)]
	Chooses technique that best proxies normal distribution 
	"Rosetta Heuristic" is (approxminately) a ranking of how far from standard deviation - lower is better
	Rosetta_heuristic = np.absolute((top_bucket - .26)) + np.absolute((bottom_bucket - .26)) + np.absolute((middle_bucket - .48)) + np.absolute((top_bucket - bottom_bucket)) * 100
	'''
	z = func_name(); print "in function:", z
	print "-----------in rebucket function------------------"
	global question_dict

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
		# how to handle alternative response ranges (binary, 1-5, etc)
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
	print best_schemes[:10]

	# now rebucket response matrix
	# could this all be redone using map()?

	X_rebucketed = np.zeros((X.shape), dtype=np.int64)
	for col in range(X_rebucketed.shape[1]):
		mapping_scheme = reduce(lambda x,y: x+y,[a*[b] for a,b in zip(best_schemes[int(col)],[1,2,3])])
		#mapping_scheme = [x+y for x,y in a*[b] for a,b in [zip(best_schemes[int(col)],[1,2,3])]]
		#X_rebucketed[:,col] = map(lambda x: mapping_scheme[int(x)-1], X[:,col])
		X_rebucketed[:,col] = [mapping_scheme[int(x)-1] for x in X[:,col]]
		# http://stackoverflow.com/questions/3459098/create-list-of-single-item-repeated-n-times-in-python
		# http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
		rebucket_counts = Counter(X_rebucketed[:,col]).values()
		question_dict[names[col]]['rebucket_counts_1'], question_dict[names[col]]['rebucket_counts_2'], question_dict[names[col]]['rebucket_counts_3'] = rebucket_counts
		question_dict[names[col]]['rebucket_shares_1'], question_dict[names[col]]['rebucket_shares_2'], question_dict[names[col]]['rebucket_shares_3'] = [float(value) / sum(rebucket_counts) for value in rebucket_counts]

	X_rebucketed_df = pd.DataFrame(X_rebucketed, columns=names)

	# rebucketed_filename = 'X_rebucketed.csv'
	# np.savetxt(basedir + rebucketed_filename, X_rebucketed, fmt='%.2f', delimiter=",")
	# print rebucketed_filename, "saved to:", basedir

	rebucketed_filename = 'X_rebucketed.csv'
	X_rebucketed_df.to_csv(basedir + rebucketed_filename, index=False)
	print rebucketed_filename, "saved to:", basedir

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

	return rebucketed_filename, X_rebucketed#, question_dict

def cluster_seed(factor_matrix, best_factor, question_number, num_cols, num_rows):
	'''
	Builds cluster_seed (size n_factors), grouping questions by highest PCA value then choosing lowest RH as cluster_seed

	'''
	z = func_name(); print "in function:", z
	print "----------in cluster_seed function-------------------"
	# Step 5: now group questions by highest factor (=factor_matrix[best_factor], remember col #1 = index 0!)
	# http://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
	
	factor_matrix, best_factor, question_number, num_cols, num_rows = factor_matrix, best_factor, question_number, num_cols, num_rows

	#print factor_matrix.shape
	sorted_factor_matrix = factor_matrix[factor_matrix[:,best_factor].argsort()]
	#print "verify question index:", sorted_factor_matrix[:,question_number]
	#print "verify sort by highest factor", sorted_factor_matrix[:,best_factor]

	# determine unique factor numbers:
	unique_factor_list= list(set(sorted_factor_matrix[:,num_cols+1]))
	# print unique_factor_list

	# now group questions by primary factor and sort by RH to form cluster 'seed'
	cluster_seed=[]
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
		cluster_seed.append(question_index_rh_list[0][1])

	print "cluster seed:", cluster_seed
	print "number of items in cluster seed", len(cluster_seed)

	return cluster_seed

# step 6: use cluster_seed as input to poLCA
#def poLCA(cluster_seed, num_seg, num_rep, rebucketed_filename):
def poLCA(i, cluster_seed, num_seg, num_rep, rebucketed_filename):
	'''
	Runs poLCA script in R - see http://dlinzer.github.io/poLCA/
	'''
	z = func_name(); print "in function:", z

	print "------------in poLCA function---------------"
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
	#rebucketed_filename = "X_rebucketed.csv"
	infile = basedir + rebucketed_filename

	#cluster_seed_names = ['q06_1', 'q06_2', 'q06_3','q06_4','q06_5','q06_6','q06_7']

	# # Variable number of args in a list
	# #args = ['A', 'B', 'C', 'D']
	# #args = ['11', '3', '9', '42']
	cluster_seed_names = [names[int(value)] for value in cluster_seed]
	#timestamp = base64.b64encode(str(time.time() + np.random.random_integers(0,1000000000000)))
	timestamp = str(uuid.uuid4()) # http://stackoverflow.com/questions/534839/how-to-create-a-guid-in-python
	# Q: how is names() global?
	cluster_seeds = [num_seg] + [num_rep] + [basedir] + [rebucketed_filename] + [timestamp] + cluster_seed_names
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
	x = subprocess.check_output(cmd, universal_newlines=True)

	#return
	#scorecard(cluster_seeds, cluster_seed_names, num_seg)
	print("------- Runtime: %.2f seconds -------" % (time.time() - start_time))
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

	print "in scorecard_2 function"
	training_list = []

	print "results_dict size: ", len(results_dict.keys())
	print "results_dict keys: ", results_dict.keys()

	for key in results_dict:
		#train_dict = {'a':key,'b':results_dict[key]['cluster_number'],'c':results_dict[key]['num_vars'],'d':esults_dict[key]['model_stats'],'e':16,'f':17}
		train_dict = {'a':key[:10],'b':results_dict[key]['cluster_number'],'c':results_dict[key]['num_vars'],'d':round(results_dict[key]['model_stats'][0]),'e':round(results_dict[key]['model_stats'][1]),'f':round(results_dict[key]['model_stats'][2]), 'g':round(results_dict[key]['weighted_average_cluster_polarity'],4) ,'h':round(results_dict[key]['average_cross_question_polarity'],4)}
		
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

	return flask.jsonify(training_results)

@app.route("/create_tracker", methods=["GET"])
def create_tracker():

	tracker_list = []
	print "in create_tracker function"
	print question_dict.keys()
	print question_dict[question_dict.keys()[0]]
	print question_dict[question_dict.keys()[0]].keys()

	# if len(results_dict) == 0 and 'run_tracker' not in question_dict[question_dict.keys()[0]].keys():
	for key in question_dict:
		tracker_dict = OrderedDict([('001_Q#',key),('002_Q_name','n/a'),('003_Dim','n/a'), ('004_PCA', round(question_dict[key]['first_factor_value'],3)),('005_#1 Factor', question_dict[key]['first_factor']),('006_#2 Factor', question_dict[key]['second_factor']),('007_Bucket', question_dict[key]['bucket_scheme']),('008_%T(=1)', round(question_dict[key]['rebucket_shares_1']*100)),('009_%M(=2)', round(question_dict[key]['rebucket_shares_2']*100)),('010_%L(=3)', round(question_dict[key]['rebucket_shares_3']*100)), ('011_RH',round(question_dict[key]['best_rh'],2))])
		tracker_list.append(tracker_dict)
	
	tracker = {'tracker': tracker_list}
	#print "create_run_tracker: ", tracker

	tracker_json = flask.jsonify(tracker)
	print tracker_json

	return flask.jsonify(tracker)
	
@app.route("/tracker", methods=["GET"])
def update_run_tracker():
# could we use intersection of two dictionaries (resuts_dict_old, resuts_dict_current) and update difference?
# http://code.activestate.com/recipes/576644-diff-two-dictionaries/
# or dict.viewitems() - see http://zetcode.com/lang/python/dictionaries/

	global results_dict #updated when new values uploaded

	tracker_list = []
	r_and_q_list = []
	print "in update_run_tracker function"
	print question_dict.keys()
	print question_dict[question_dict.keys()[0]]
	print question_dict[question_dict.keys()[0]].keys()

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
		print r_and_q_list

		run_reports_list = run_report(keys_to_upload)
		tracker = {'tracker': tracker_list, 'run_reports': run_reports_list}

		return flask.jsonify(tracker)

@app.route("/submit_data", methods=["POST"])
def submit_data():

    # read the data that came with the POST request as a dict
    # inbound request example: http://127.0.0.1:5000/predict -X POST -H 'Content-Type: application/json' -d '{"example": [154]}'
    # my example: http://127.0.0.1:5000/test_predict -X POST -H 'Content-Type: application/json' -d '{'keywords': ['rabbit','goat','wine'],'rating_metric': ['lda'],'distance_metric': ['pearsonr']}'
    # simple example: curl http://127.0.0.1:5000/test_predict -X POST -H 'Content-Type: applcation/json' -d '{"keywords": '66666'}'
	# curl http://127.0.0.1:5000/submit_data -X POST -H 'Content-Type: applcation/json' -d '{u'segments': [6], u'questions': u'[q39_8,q39_4,q39_6,q48_28,q31_20,q07_18,q07_17,q07_11,q35_5,q16_8,q06_2,q08_5,q08_6,q06_8,q06_16,q35_8,q48_25,q48_27]'}'
	# {"questions":"[q39_8,q39_4,q39_6,q48_28,q31_20,q07_18,q07_17,q07_11,q35_5,q16_8,q06_2,q08_5,q08_6,q06_8,q06_16,q35_8,q48_25,q48_27]", "segments":[6]} 
	# 	
	global cluster_seed

	counter = 0
	inbound_data = flask.request.json
	counter +=1
	print "counter:", counter
	print "inbound data", inbound_data
	keys = inbound_data.keys(); print keys
	for key in inbound_data:
		print key, inbound_data[key], type(inbound_data[key])

	if 'questions' in keys:
		print "----questions---"
		questions = inbound_data['questions']
		
		if 'null' not in questions:
			#is not None and questions.strip("[]").encode('ascii', 'ignore') is not 'null':
			questions = questions.strip("[]").split(',')
			questions = [q.encode('ascii', 'ignore') for q in questions]
			cluster_seed_inbound = [names.index(question) for question in questions] # convert questions to index #

		else:
			cluster_seed_inbound = cluster_seed
			print "no questions provided; using cluster seed"

		print "questions:", len(questions), questions

	if "segments" in keys:
		print "---segments----"
		num_seg = inbound_data['segments']
		num_segments = num_seg[0]
		
		print "num_segments:", num_segments

	if 'xls' in keys:
		xls = False
		print '---excel export----'
		xl_flag = inbound_data['xls'].strip("[]")

		if xl_flag == 'true':
			xls = True

		print "excel export:", xls

	if 'grid_search' in keys:
		grid_search = False
		print '---grid search----'
		gs_flag = inbound_data['grid_search'].strip("[]")
		
		if gs_flag == 'true':
			grid_search = True

		print "grid_search:", grid_search	

	#cluster_seed_inbound = [names.index(question) for question in questions]
	print "cluster_seed_inbound:", cluster_seed_inbound

	#cluster_seed = [60.0, 90.0, 24.0, 125.0, 72.0, 61.0, 30.0, 68.0, 14.0, 123.0, 144.0, 49.0, 65.0, 53.0, 50.0, 12.0, 58.0, 146.0, 112.0, 81.0, 48.0, 43.0, 134.0, 139.0, 51.0, 86.0, 4.0, 94.0, 111.0, 99.0]
	#cluster_seed = cluster_seed_2
	num_seg = num_segments
	#grid_search = False
	num_rep = 1
	#rebucketed_filename = 'X_rebucketed.csv'

	#print "----run number:", len(results_dict.keys()), "-----------"
 
	print "---------now running poLCA from submit_data function--------------"
	results = run_poLCA (grid_search,cluster_seed_inbound,num_seg,num_rep,rebucketed_filename)	
	timestamps = update_results_dict(results, X_rebucketed, names)

	print "---question_dict:-----", len(question_dict)
	print "---results_dict:------", len(results_dict)
	
	update_question_dict(timestamps)
	run_report(timestamps)
	clean_up(timestamps)

	if xls:
		make_xls()


	results2 = {"example": [155]}
	return flask.jsonify(results2)

# file upload section
# see http://code.runnable.com/UiPcaBXaxGNYAAAL/how-to-upload-a-file-to-the-server-in-flask-for-python
# https://github.com/moremorefor/flask-fileupload-ajax-example/blob/master/app.py
# http://stackoverflow.com/questions/18334717/how-to-upload-a-file-using-an-ajax-call-in-flask

# Route that will process the file upload
@app.route('/upload', methods=['POST'])
def upldfile():
    if request.method == 'POST':
        files = request.files['file']
        if files and allowed_file(files.filename):
            filename = secure_filename(files.filename)
            app.logger.info('FileName: ' + filename)
            updir = os.path.join(basedir_for_upload, 'uploads/')
            files.save(os.path.join(updir, filename))
            file_size = os.path.getsize(os.path.join(updir, filename))
            return jsonify(name=filename, size=file_size)

def scorecard(timestamp, cluster_seeds, cluster_seed_names, num_seg):
	'''
	Yields scorecard for single segmentation run - legacy module, does not run off results_dict
	'''
	z = func_name(); print "in function:", z
	print "------------in scorecard function---------------------"
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

	return

def update_results_dict(results, X_rebucketed, names):
	''''
	Updates results_dict after poLCA run
	'''
	z = func_name(); print "in function:", z
	print "----------in make_results_dict function-------------"
	global results_dict
	global question_dict

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
	num = 0
	for key in timestamps:#results_dict.keys():
		results_dict[key]['run_number'] = starting_results_dict_length + num + 1
		results_dict[key]['cluster_number'] = cluster_numbers[num]
		results_dict[key]['cluster_counts'] = Counter(predicted_clusters[num]).values()
		results_dict[key]['cluster_shares'] = [ float(cluster) / sum(results_dict[key]['cluster_counts']) for cluster in results_dict[key]['cluster_counts']]
 		results_dict[key]['num_vars'] = num_vars[num]
		results_dict[key]['quest_list'] = quest_list[num]
		results_dict[key]['model_stats'] = model_stats[num]
		results_dict[key]['predicted_clusters'] = predicted_clusters[num]
		results_dict[key]['posterior_probabilities'] = posterior_probabilities[num]
		results_dict[key]['upload_state'] = False
		num +=1

	print "results_dict size after adding new keys:", len(results_dict.keys())

	# within-segment polarity = ((max bucket share) - average bucket share) for each question and cluster
	# cross-question polarity = max bucket share - average bucket share for each question bucket across clusters
	
	print("------- Runtime: %.2f seconds -------" % (time.time() - start_time))
	
	#results_dict = dict(Parallel(n_jobs=num_cores,verbose=5)(delayed(make_one_results_dict_entry_mp)(i,results_dict.keys()[i],results_dict[results_dict.keys()[i]], X_rebucketed, names) for i in range(len(results_dict.keys()))))
	new_results_dict = dict(Parallel(n_jobs=num_cores,verbose=5)(delayed(make_one_results_dict_entry_mp)(i,timestamps[i],results_dict[timestamps[i]], X_rebucketed, names) for i in range(len(timestamps))))
	results_dict.update(new_results_dict)

	print "sample results_dict entry:",results_dict[results_dict.keys()[0]].keys()
	print 'sample results_dict[cluster_counts] entry:', results_dict[results_dict.keys()[0]]['cluster_counts']
	print 'sample results_dict[cluster_shares] entry:', results_dict[results_dict.keys()[0]]['cluster_shares']

	print "results_dict complete with", len(results_dict.keys()), "entries"
	print("------- Runtime: %.2f seconds -------" % (time.time() - start_time))

	return timestamps#, results_dict

def make_one_results_dict_entry_mp(i, key, results_dict_entry, X_rebucketed, names):
	'''
	Breaks apart results_dict update into threads if multiprocessing possible
	'''
	z = func_name(); print "in function:", z
	print "---------in make_one_results_dict_entry_mp function-----------------"
	results_dict = dict(results_dict_entry) #not same as global var; should change name!
	
	pred_clusters = np.array(results_dict['predicted_clusters'])
	pred_clusters = np.reshape(pred_clusters, (pred_clusters.shape[0],1))
	clustered_responders = np.append(X_rebucketed, pred_clusters, axis=1)
	
	results_dict['response_shares'] = {}; results_dict['response_counts'] = {}; results_dict['response_polarity'] = {}; results_dict['cross_question_polarity'] = {}
	questions = results_dict['quest_list']
	
	#for question in results_dict['quest_list']:
	# for question in question_dict.keys():
	for question in names:
		response_shares = []; response_counts = []
		
		for cluster in set(results_dict['predicted_clusters']):
			response_count = [sum(clustered_responders[clustered_responders[:,clustered_responders.shape[1]-1] == cluster][:,names.index(question)] == bucket) for bucket in set(clustered_responders[:,names.index(question)])]
			response_counts.append(response_count)
			response_share = [float(response_count[i])/sum(response_count) for i in range(len(response_count))]
			response_shares.append(response_share)

		#print question, "response counts:", response_counts	
		#print question, "response shares:", response_shares	
		results_dict['response_counts'][question] = response_counts
		results_dict['response_shares'][question] = response_shares
		results_dict['response_polarity'][question] = [max(item) - np.mean(item) for item in response_shares]
		cross_question_polarity = np.reshape(results_dict['response_shares'][question],(results_dict['cluster_number'],3))
		results_dict['cross_question_polarity'][question] = [max(cross_question_polarity[:,col]) - np.mean(cross_question_polarity[:,col]) for col in range(cross_question_polarity.shape[1])]

	print "results_dict[response_shares] length:", len(results_dict['response_shares'])
	
	results_dict['polarity_scores'] = [sum(results_dict['response_polarity'][question][cluster] / len (results_dict['quest_list']) for question in results_dict['quest_list']) for cluster in range(results_dict['cluster_number'])]
	results_dict['weighted_average_cluster_polarity'] = sum([a*b for a,b in zip(results_dict['polarity_scores'],results_dict['cluster_shares'])])
	results_dict['mean_cluster polarity'] = np.mean(results_dict['polarity_scores'])
	results_dict['average_cross_question_polarity'] = sum(sum(value) for value in results_dict['cross_question_polarity'].values()) / len(results_dict['cross_question_polarity'].values())	

	print "number of entries for results_dict", key, len(results_dict.keys())
	#print results_dict
	one_entry = dict(results_dict.items())

	print "number of entries for one_entry", key, len(one_entry.keys())

	return key, one_entry

def update_question_dict(timestamps):#question_dict, results_dict):
	'''
	Updates question_dict after poLCA run, adding back a boolean run inclusion tracker for each question 
	'''
	z = func_name(); print "in function:", z

	global results_dict
	global question_dict

	print "--------in update_question_dict function--------------"
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

	return #question_dict, results_dict

def make_xls():#results_dict):
	'''
	Excel export - currently only detailed run results
	'''
	z = func_name(); print "in function:", z

	global results_dict

	# uses http://xlsxwriter.readthedocs.org/getting_started.html
	# sample of results_dict[key]['response_shares']:
	#q16_5 response shares: [[0.36829268292682926, 0.4121951219512195, 0.21951219512195122], [0.2006172839506173, 0.41975308641975306, 0.37962962962962965], [0.2997032640949555, 0.5252225519287834, 0.17507418397626112], [0.09090909090909091, 0.5844155844155844, 0.3246753246753247], [0.07824427480916031, 0.7519083969465649, 0.16984732824427481]]

	outfile = 'clustering_data_export.xlsx'
	workbook = xlsxwriter.Workbook(basedir+outfile)
	run_number = 0

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

	workbook.close()

	print "workbook exported: ", basedir+outfile
	return

def clean_up (timestamps):
	'''
	Deletes working files from poLCA run (easiest way R>Python)
	'''
	z = func_name(); print "in function:", z
	print "----in clean_up function----"
	# now clean up!
	for timestamp in timestamps:
		os.remove(basedir+"/static/model/model_stats_"+timestamp+".txt")
		os.remove(basedir+"/static/model/predicted_segment_"+timestamp+".txt")
		os.remove(basedir+"/static/model/posterior_probabilities_"+timestamp+".txt")
	print len(timestamps) * 3, "scoring files cleaned up from ", len(timestamps), "runs"

	return

def save_results():
	'''
	Writes results_dict to file
	'''
	z = func_name(); print "--------in function: ", z, "--------------"

	global results_dict
	import ujson
	import uuid
	import json
	import simplejson  
	import cPickle as pickle
	import marshal
	import time
	
	time_dict = {}
	load_time_dict = {}

	# outfile = 'results_dict_' + str(uuid.uuid4()) + '.json'

	# cPickle
	method = 'cPickle'
	print "trying :", method
	last_time = time.time()
	try:
		outfile = 'results_dict_' + str(uuid.uuid4()) + '.pkl'
		with open(outfile, 'wb') as handle:
		    pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)		
		this_time = time.time()
		elapsed_time = this_time - last_time
		print method,": ", outfile, ' saved with ', len(results_dict), ' entries, in :' , elapsed_time
		time_dict[method] = elapsed_time

		next_time = time.time()
		pickle.load(open(outfile, 'rb'))
		elapsed_time = next_time - this_time
		load_time_dict[method] = elapsed_time
	
	except:
		pass
	
	print '------------Results-----------'
	print "saving file:", sorted(time_dict.items(), key=lambda x: x[1])
	print "loading file:", sorted(load_time_dict.items(), key=lambda x: x[1])
	return


def run_poLCA (grid_search,cluster_seed,num_seg,num_rep,rebucketed_filename):
	'''
	Runs single or gridsearch, also uses multiprocessing for gridsearch (if possible)
	'''
	z = func_name(); print "in function:", z

	if grid_search == False:
		print '--------in grid_search == False function--------------'
		timestamp, cluster_seeds, cluster_seed_names, num_seg = poLCA (1, cluster_seed,num_seg,num_rep,rebucketed_filename)
		scorecard(timestamp, cluster_seeds, cluster_seed_names, num_seg)
		#timestamps = [timestamp]
		results = [(timestamp,cluster_seeds,cluster_seed_names,num_seg)]
		print "results:", results

	if grid_search == True:
		print '---------in grid search == True function---------'
		# run clustering in parallel if possible
		shortened_cluster_seeds = []
		num_remove = 5
		random.shuffle(cluster_seed)
		
		shortened_cluster_seeds = [cluster_seed[0:(len(cluster_seed)-num)] for num in range(num_remove+1) ]
		print "shortened cluster seed:", shortened_cluster_seeds

		# shortened_cluster_seeds = [x for x in itertools.combinations(cluster_seed, 28)]
		# print "total cluster seeds: ", len(shortened_cluster_seeds)

		num_seg = [5,6,7,8]#,9,10,11,12,13,14]
		
		num_cores = multiprocessing.cpu_count()
		#results = Parallel(n_jobs=num_cores,verbose=5)(delayed(poLCA)(i,cluster_seed, num_seg[i], num_rep, rebucketed_filename) for i in range(10))
		
		#scorecard(cluster_seeds, cluster_seed_names, num_seg)
		# see http://blog.dominodatalab.com/simple-parallelization/
		print "running grid search (seg x variables removed):", len(num_seg), ", ", num_remove
		results = Parallel(n_jobs=num_cores,verbose=5)(delayed(poLCA)(i,shortened_cluster_seeds[j], num_seg[i], num_rep, rebucketed_filename) for i in range(len(num_seg)) for j in range (len(shortened_cluster_seeds)))
		#print results
	
	print "number of runs completed:", len(results)
	return results

def func_name():
	import traceback
	return traceback.extract_stack(None, 2)[0][2]

def run_report(timestamps):
	'''
	Detailed report from single segmenting run
	'''
	z = func_name(); print "------in function:", z, "---------------"

	global results_dict

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
		
		segment_counts = ['# in seg',   '']     + results_dict[run]['cluster_counts']  + [sum(results_dict[run]['cluster_counts'])] + [''] + ([''] * len(segment_list))
		segment_shares = ['Pct in seg', '']     + ['{0:.1%}'.format(results_dict[run]['cluster_shares'][cluster]) for cluster in range(number_of_clusters)] + ['100.0%'] + [''] + ([''] * len(segment_list)) 
		header_1 = 		 ['', '', 		  ]     + segment_list + ['Total']        + [''] + ([''] * len(segment_list)) 
		header_2 = 		 ['Question', 'Bucket'] + segment_list + ['Average'] + [''] + segment_list
		row_spacer = [''] * (len(header_2))
		
		header = [header_1] + [segment_counts] + [segment_shares] + [row_spacer] + [header_2] + [row_spacer] + [row_spacer]
		run_report += header


		# data: segment response shares			
		for survey_question in results_dict[run]['response_shares'].keys():
			
			#print "result_dict entry: ", survey_question, len(results_dict[run]['response_shares'][survey_question]), results_dict[run]['response_shares'][survey_question]
			top_bucket_data = [results_dict[run]['response_shares'][survey_question][cluster][0] for cluster in range(number_of_clusters)]
			top_bucket = ['{0:.1%}'.format(item) for item in top_bucket_data]
			top_bucket_average = sum([float(x)*float(y) for x,y in zip(top_bucket_data,results_dict[run]['cluster_shares'])])
			top_bucket_avg = ['{0:.1%}'.format(top_bucket_average)]
			top_bucket_index_scores = [int((float(share) / top_bucket_average) * 100) for share in top_bucket_data]
			print "top bucket:", top_bucket
			print 'top bucket weighted average:', top_bucket_average
			print "top bucket index scores:", top_bucket_index_scores

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
			print 'top bucket weighted average:', top_bucket_average

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

		run_reports.append(run_report)

	print 'total length of run_reports:', len(run_reports)
	print 'total length of one run_report', len(run_reports[0])
	print 'header for one run_report:', run_reports[0][0]
	print 'sample entry for one question in one run_report:', run_reports[0][1]
 				
	return run_reports
	

if __name__ == "__main__":

	#app = Flask(__name__)
	num_seg = 5
	num_rep = 1
	n_components = 30
	top_n = 2
	num_cores = multiprocessing.cpu_count()
	results_dict = {}
	question_dict = {}

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
	factor_matrix, names, X = get_PCA(filename, n_components)
	factor_matrix, num_rows, num_cols, question_number, rh, best_factor, second_best_factor = top_n_factors(factor_matrix, top_n)
	rebucketed_filename, X_rebucketed = rebucket(factor_matrix, names, X, rh)
	cluster_seed = cluster_seed(factor_matrix, best_factor, question_number, num_cols, num_rows)
	# app.run()

	print "cluster_seed: ", cluster_seed

	if grid_search:
	#analysis and reporting pipeline
		results = run_poLCA (grid_search,cluster_seed,num_seg,num_rep,rebucketed_filename)	
		timestamps = update_results_dict(results, X_rebucketed, names)
		update_question_dict(timestamps)
		clean_up(timestamps)
		save_results()
		run_report(timestamps)
	
	print("------- Runtime: %.2f seconds -------" % (time.time() - start_time))

	# feature switches
	if xls:
		make_xls()

	if interactive_mode:
		app.run(debug = True)
		
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
	# conditional formatting for run_report
	# grid search = largest
	# mongo db for grid search = largge
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
	# visualization - 3D (users, PCA(questions)(:3), clusters)
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





