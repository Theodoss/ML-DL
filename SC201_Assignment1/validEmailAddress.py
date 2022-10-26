"""
File: validEmailAddress.py
Name: 
----------------------------
This file shows what a feature vector is
and what a weight vector is for valid email 
address classifier. You will use a given 
weight vector to classify what is the percentage
of correct classification.

Accuracy of this model: TODO: 0.5
"""

WEIGHT = [                           # The weight vector selected by Jerry
	[0.4],                           # (see assignment handout for more details)
	[0.4],
	[0.2],
	[0.2],
	[0.9],
	[-0.65],
	[0.1],
	[0.1],
	[0.1],
	[-0.7]
]

DATA_FILE = 'is_valid_email.txt'     # This is the file name to be processed


def main():
	maybe_email_list = read_in_data()
	is_vaild_mail_list = []
	for maybe_email in maybe_email_list:
		feature_vector = feature_extractor(maybe_email)
		# TODO:
		score = 0
		for i in range(len(WEIGHT)):
			score += feature_vector[i]*WEIGHT[i][0]
		is_vaild_mail_list.append(1) if score > 1 else is_vaild_mail_list.append(0)

	ans_vaild_mail = [0 for i in range(13)]
	[ans_vaild_mail.append(1) for i in range(13)]

	not_vaild_list = []
	train_ans = 0
	for i in range(len(is_vaild_mail_list)):
		if is_vaild_mail_list[i] == ans_vaild_mail[i]:
			train_ans += 1
		else:
			not_vaild_list.append(i+1)

	print(f'Ans list is ={ans_vaild_mail}')
	print(f'Test ans is ={is_vaild_mail_list}')
	print(f'Wrong mail is {not_vaild_list}')
	print(f'The accuracy is {train_ans/len(is_vaild_mail_list)*100}%')




def feature_extractor(maybe_email):
	"""
	:param maybe_email: str, the string to be processed
	:return: list, feature vector with 10 values of 0's or 1's
	"""
	# maybe_email = "e0987b0066@gmail.com"
	feature_vector = [0] * len(WEIGHT)
	for i in range(len(feature_vector)):
		if i == 0:
			feature_vector[i] = 1 if '@' in maybe_email else 0
		elif i == 1:
			if feature_vector[0]:
				feature_vector[i] = 1 if '.' not in maybe_email.split('@')[0] else 0

		###################################
		#                                 #
		#              TODO:              #
		#                                 #
		###################################
		elif i == 2:
			if feature_vector[0]:
				feature_vector[i] = 1 if len(maybe_email.split('@')[0]) >0 else 0
		elif i == 3:
			if feature_vector[0]:
				feature_vector[i] = 1 if len(maybe_email.split('@')[1]) >0 else 0
		elif i == 4:
			if feature_vector[0]:
				feature_vector[i] = 1 if '.' in maybe_email.split('@')[1] else 0
		elif i == 5:
			if feature_vector[0]:
				feature_vector[i] = 1 if " " not in maybe_email else 0
		elif i == 6:
			if feature_vector[0]:
				feature_vector[i] = 1 if 'com' in maybe_email.split('.')[-1] else 0
		elif i == 7:
			if feature_vector[0]:
				feature_vector[i] = 1 if 'edu' in maybe_email.split('.')[-1] else 0
		elif i == 8:
			if feature_vector[0]:
				feature_vector[i] = 1 if 'tw' in maybe_email.split('.')[-1] else 0
		elif i == 9:
			if feature_vector[0]:
				feature_vector[i] = 1 if len(maybe_email) > 10 else 0


	return feature_vector


def read_in_data():
	"""
	:return: list, containing strings that might be valid email addresses
	"""
	# TODO:
	with open(DATA_FILE,'r') as f:
		email_list = []
		for line in f :
			email_list.append(line)
	# print(email_list)
	return email_list

if __name__ == '__main__':
	main()
