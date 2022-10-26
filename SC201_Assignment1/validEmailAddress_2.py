"""
File: validEmailAddress_2.py
Name: 
----------------------------
Please construct your own feature vectors
and try to surpass the accuracy achieved by
Jerry's feature vector in validEmailAddress.py.
feature1:  TODO:
feature2:  TODO:
feature3:  TODO:
feature4:  TODO:
feature5:  TODO:
feature6:  TODO:
feature7:  TODO:
feature8:  TODO:
feature9:  TODO:
feature10: TODO:

Accuracy of your model: TODO:
"""
import re

WEIGHT = [                           # The weight vector selected by Jerry
	[0.5],          # feature0: 只有一個@字號              # (see assignment handout for more details)
	[0.1],			# feature1: 前段沒有.
	[0.2],			# feature2: 前段大於3
	[0.2],			# feature3: 後段大於5
	[0.4],			# feature4: 後段有.
	[-2],			# feature5: 有空白符號
	[0.1],			# feature6: 以.com結尾
	[0.1],			# feature7: 以.edu結尾
	[0.1],			# feature8: 以.tw結尾
	[-2],			# feature9: 兩個以上@
	[-2],			# feature10: 暫時沒有
	[-2],			# feature11: 連續.
	[-1],			# feature12: 非."".
	[-1],			# feature13: &
	[-1],			# feature14: |
	[-1],			# feature15: =
	[-2],			# feature16: _
	[-2],			# feature17: )
	[-2],			# feature18: (
	[-2],			# feature19: :
	[-2],			# feature20: ;
	[-2],			# feature21: <
	[-2],			# feature22: >
	[-2],			# feature23: %
	[-2],			# feature24: ^
	[-2]			# feature25: $

]

DATA_FILE = 'is_valid_email.txt'     # This is the file name to be processed
# DATA_FILE = '123.txt'     # This is the file name to be processed


def main():
	maybe_email_list = read_in_data()
	is_vaild_mail_list = []
	score_list = []
	for maybe_email in maybe_email_list:
		feature_vector = feature_extractor(maybe_email)
		score = 0
		for i in range(len(WEIGHT)):
			score += feature_vector[i]*WEIGHT[i][0]
		score_list.append(score)
		# 建立答案列
		is_vaild_mail_list.append(1) if score > 1 else is_vaild_mail_list.append(0)

	# 建立真實答案列
	ans_vaild_mail = [0 for i in range(13)]
	[ans_vaild_mail.append(1) for i in range(13)]

	not_vaild_list = []
	train_ans = 0
	# 計算正確率
	for i in range(len(is_vaild_mail_list)):
		if is_vaild_mail_list[i] == ans_vaild_mail[i]:
			train_ans += 1
		else:
			not_vaild_list.append(i+1)
	print(f'Score list is ={score_list}')
	print(f'Ans list is ={ans_vaild_mail}')
	print(f'Test ans is ={is_vaild_mail_list}')
	print(f'Wrong mail is {not_vaild_list}')
	print(f'The accuracy is {train_ans/len(is_vaild_mail_list)*100}%')




def feature_extractor(maybe_email):
	"""
	:param maybe_email: str, the string to be processed
	:return: list, feature vector with 10 values of 0's or 1's
	"""
	# maybe_email = "e098\"asd\"asdasd7b0066@@gmail.com"
	feature_vector = [0] * len(WEIGHT)
	# 監測雙括弧與忽略雙括弧內字串
	if "\"" in maybe_email:
		temp_mail = maybe_email.split("\"")
		for i in range(len(temp_mail)):
			if i % 2 == 1:
				temp_mail[i] = ""
		maybe_email = "\"".join(temp_mail)
		temp_mail = re.split("\.|@", maybe_email)
		for i in range(len(temp_mail)):
			if "\"" in temp_mail[i]:
				if temp_mail[i] != "\"\"":
					feature_vector[12] = 1

	for i in range(len(feature_vector)):
		# 只有一個@字號
		if i == 0:
			at_count = 0
			for j in maybe_email:
				at_count += 1 if j == "@" else 0
			if at_count == 1:
				feature_vector[i] = 1
			else:
				feature_vector[9] = 1
		elif i == 1:
			if feature_vector[0]:
				feature_vector[i] = 1 if '.' not in maybe_email.split('@')[0] else 0
		# 不能有 &=_"(),:;<>[\]'_+
		elif i == 2:
			if feature_vector[0]:
				feature_vector[i] = 1 if len(maybe_email.split('@')[0]) >3 else 0
		elif i == 3:
			if feature_vector[0]:
				feature_vector[i] = 1 if len(maybe_email.split('@')[1]) >5 else 0
		elif i == 4:
			if feature_vector[0]:
				feature_vector[i] = 1 if '.' in maybe_email.split('@')[1] else 0
		elif i == 5:
			if feature_vector[0]:
				feature_vector[i] = 1 if " " in maybe_email else 0
		elif i == 6:
			if feature_vector[0]:
				feature_vector[i] = 1 if 'com' in maybe_email.split('.')[-1] else 0
		elif i == 7:
			if feature_vector[0]:
				feature_vector[i] = 1 if 'edu' in maybe_email.split('.')[-1] else 0
		elif i == 8:
			if feature_vector[0]:
				feature_vector[i] = 1 if 'tw' in maybe_email.split('.')[-1] else 0

		# # 不能有 #
		# elif i == 10:
		# 	if feature_vector[0]:
		# 		temp_list = re.split('\.|@', maybe_email)
		# 		feature_vector[i] = 1 if '\"' in temp_list else 0

		# 不能有 連續 . & .@
		elif i == 11:
			if feature_vector[0]:
				temp_list = re.split('\.|@', maybe_email)
				feature_vector[i] = 1 if '' in temp_list else 0
		# 不能有 "&" "|" "=" "_" ")" "(" ":" ";" "<" ">" "%"
		elif i == 13:
			if feature_vector[0]:
				feature_vector[i] = 1 if '&' in maybe_email else 0
		elif i == 14:
			if feature_vector[0]:
				feature_vector[i] = 1 if '|' in maybe_email else 0
		elif i == 15:
			if feature_vector[0]:
				feature_vector[i] = 1 if '=' in maybe_email else 0
		elif i == 16:
			if feature_vector[0]:
				feature_vector[i] = 1 if '_' in maybe_email else 0
		elif i == 17:
			if feature_vector[0]:
				feature_vector[i] = 1 if ')' in maybe_email else 0
		elif i == 18:
			if feature_vector[0]:
				feature_vector[i] = 1 if '(' in maybe_email else 0
		elif i == 19:
			if feature_vector[0]:
				feature_vector[i] = 1 if ':' in maybe_email else 0
		elif i == 20:
			if feature_vector[0]:
				feature_vector[i] = 1 if ';' in maybe_email else 0
		elif i == 21:
			if feature_vector[0]:
				feature_vector[i] = 1 if '<' in maybe_email else 0
		elif i == 22:
			if feature_vector[0]:
				feature_vector[i] = 1 if '>' in maybe_email else 0
		elif i == 23:
			if feature_vector[0]:
				feature_vector[i] = 1 if '%' in maybe_email else 0
		elif i == 24:
			if feature_vector[0]:
				feature_vector[i] = 1 if '^' in maybe_email else 0
		elif i == 25:
			if feature_vector[0]:
				feature_vector[i] = 1 if '$' in maybe_email else 0

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


