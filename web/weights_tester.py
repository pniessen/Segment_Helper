import numpy as np

basedir = os.path.abspath(os.path.dirname(__file__)) + "/"
filename = 'test_weights_v1.csv'

responder_data = pd.DataFrame.from_csv(path)

responder_data = np.genfromtxt(basedir+filename, delimiter=',', skip_header=1)
print 'responder data loaded: ', responder_data.shape
print responder_data

def calculate_weights(responder_data`):

	weights = []

	# target populations (source: US Census)
	target_age_categories = ['18-24','25-34','35-44','45-54','55-65']
	target_age_distribution = [15,22,20,21,22]
	target_age_dict = dict(zip(target_age_categories, target_age_distribution))

	target_gender_categories = ['male','female']
	target_gender_distribution = [49.79, 50.21]
	target_gender_dict = dict(zip(target_gender_categories, target_gender_distribution))

	# calculate age_share, gender_share
	responder_age_shares = [sum(responder_data[responder_data[:,1]==c1][:,1]) for c1 in target_age_categories]
	responder_age_dict = dict(zip(target_age_categories, responder_age_shares))
	responder_gender_shares = [sum(responder_data[responder_data[:,2]==c2][:,2]) for c2 in target_gender_categories]
	responder_gender_dict = dict(zip(target_age_categories, responder_age_shares))

	for responder in range(len(responder_data)):
		responder_age = responder_data[responder, 1]
		responder_gender = responder_data[responder,2]
		one_weight = responder_age_dict[responder_age]* responder_gender_dict[responder_gender] / target_age_dict[responder_age] * target_gender_dict[responder_gender]
		weights.append(one_weight)

	return weights