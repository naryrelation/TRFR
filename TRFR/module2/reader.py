import json
import re
import random

import numpy as np
from keras.utils.np_utils import to_categorical

random.seed(1984)

INPUT_PADDING = 50
OUTPUT_PADDING = 100


class Vocabulary(object):

	def __init__(self, vocabulary_file, vector,padding=None):
		"""
			Creates a vocabulary from a file
			:param vocabulary_file: the path to the vocabulary
		"""
		self.vocabulary_file = vocabulary_file
		with open(vocabulary_file, 'r') as f:
			self.vocabulary = json.load(f)

		self.padding = padding
		self.reverse_vocabulary = {v: k for k, v in self.vocabulary.items()}
		self.vector=vector

	def size(self):
		"""
			Gets the size of the vocabulary
		"""
		return len(self.vocabulary.keys())

	def string_to_int_target(self, text):
		"""
			Converts a string into it's character integer
			representation
			:param text: text to convert
		"""
		##characters = list(text)
		strtext=str(text)
		characters=strtext.split("\t")
		integers = []

		if self.padding and len(characters) >= self.padding:
			# truncate if too long
			characters = characters[:self.padding - 1]

		# listunk = [float(0)] * 100
		# self.vector['<unk>'] = listunk
		# vectorkey = self.vector.keys()
		# for c in characters:
		# 	if c in self.vocabulary and c in vectorkey:
		# 		for i in range(len(self.vector[c])):
		# 			self.vector[c][i]=float(self.vector[c][i])
		# 		integers.append(self.vector[c])
		# 	else:
		# 		integers.append(self.vector['<unk>'])
		#characters.append('<eot>')

		for c in characters:
			if c in self.vocabulary:
				integers.append(self.vocabulary[c])
				b=self.vocabulary[c]
			else:
				integers.append(self.vocabulary['<unk>'])
				b=self.vocabulary['<unk>']


		# pad:
		if self.padding and len(integers) < 1:
			integers.extend([self.vocabulary['<unk>']]
							* (1 - len(integers)))

		if len(integers) != 1:
			print(text)
			raise AttributeError('target Length of text was not 5.')
		return integers
		#return b
	def string_to_int(self, text):
		"""
			Converts a string into it's character integer
			representation
			:param text: text to convert
		"""
		##characters = list(text)
		strtext=str(text)
		#strtext.rstrip('\n')
		characters=strtext[:-1].split("\t")
		integers = []

		if self.padding and len(characters) > self.padding:
			# truncate if too long
			characters = characters[:self.padding]

		#characters.append('<eot>')
		listunk=['0']*100
		self.vector['<unk>']=listunk
		vectorkey=self.vector.keys()
		for c in characters:
			if c in self.vocabulary and c in vectorkey:
				integers.append(self.vector[c])
			else:
				integers.append(self.vector['<unk>'])


		# pad:
		if self.padding and len(integers) < self.padding:
			integers.extend([self.vector['<unk>']]
							* (self.padding - len(integers)))

		if len(integers) != self.padding:
			print(text)
			raise AttributeError('Length of text was not padding.')
		return integers

	def int_to_string(self, integers):
		"""
			Decodes a list of integers
			into it's string representation
		"""
		# characters = []
		# for i in integers:
		# 	characters.append(self.reverse_vocabulary[i])
		#
		# return characters
		cha=[]
		for i in range(len(integers)):
			#print(integers[i][0])
			ch=self.reverse_vocabulary[integers[i][0]]
			#print(ch)
			#integers[i][1]=ch
			cha.append(ch)
		return  cha


class Data(object):
	def __init__(self, file_name, input_vocabulary, output_vocabulary_entity,output_vocabulary_relation):
		"""
			Creates an object that gets data from a file
			:param file_name: name of the file to read from
			:param vocabulary: the Vocabulary object to use
			:param batch_size: the number of datapoints to return
			:param padding: the amount of padding to apply to
							a short string
		"""

		self.input_vocabulary = input_vocabulary
		self.output_vocabulary_entity = output_vocabulary_entity
		self.output_vocabulary_relation = output_vocabulary_relation
		self.file_name = file_name

	def load(self):
		"""
			Loads data from a file
		"""
		self.inputs1 = []
		self.inputs2 = []
		self.inputs3 = []
		self.inputs4 = []
		self.inputs5 = []
		self.targets1 = []
		self.targets2 = []
		temp4=[]
		temp5=[]
		n=1
		with open(self.file_name, 'r') as f:
			for line in f.readlines():
				strinfo = re.compile(' ')
				line = strinfo.sub('\t', line)
				if line!="":
					if n%9==2:
						concepts = line.strip().split("\t")
						self.targets1.append(concepts[3])
						self.targets2.append(concepts[3])
						temp4.append(concepts[1])
						temp5.append(concepts[2])
					if n%9==3:
						temp4.append(line)
					if n % 9 == 4:
						temp5.append(line)
					if n%9==5:
						self.inputs1.append(line)
					if n%9==6:
						self.inputs2.append(line)
					if n%9==7:
						self.inputs3.append(line)
					n += 1
		j=0
		for i in range(int(len(temp4)/2)):
			self.inputs4.append(temp4[j]+"\t"+temp4[j+1])
			j=j+2
		k=0
		for i in range(int(len(temp5)/2)):
			self.inputs5.append(temp5[k]+"\t"+temp5[k+1])
			k=k+2



	def transform(self,vector):
		"""
			Transforms the data as necessary
		"""
		self.inputs1 = np.array(list(
			map(self.input_vocabulary.string_to_int, self.inputs1)),dtype=np.float64)
		print("inputs1")
		
		self.inputs2 = np.array(list(
			map(self.input_vocabulary.string_to_int, self.inputs2)),dtype=np.float64)
		print("inputs2")
		self.inputs3 = np.array(list(
			map(self.input_vocabulary.string_to_int, self.inputs3)),dtype=np.float64)
		print("inputs3")
		self.inputs4 = np.array(list(
			map(self.input_vocabulary.string_to_int, self.inputs4)),dtype=np.float64)
		print("inputs4")
		self.inputs5 = np.array(list(
			map(self.input_vocabulary.string_to_int, self.inputs5)),dtype=np.float64)
		print("inputs5")
		#self.inputs = np.array(list(map(self.input_vocabulary.string_to_int, self.inputs)))
		self.targets1 = np.array(list(map(self.output_vocabulary_relation.string_to_int_target, self.targets1)),dtype=np.float64)
		self.targets2 = np.array(list(map(self.output_vocabulary_entity.string_to_int_target, self.targets2)), dtype=np.float64)
		#target--:one-hot
		self.targets1 = np.array(list(map(lambda x: to_categorical(x,num_classes=self.output_vocabulary_relation.size()),self.targets1)))
		self.targets2 = np.array(list(map(lambda x: to_categorical(x, num_classes=self.output_vocabulary_entity.size()), self.targets2)))
		print(len(self.inputs1.shape))
		assert len(self.inputs1.shape) == 3, 'Inputs could not properly be encoded'
		assert len(self.inputs2.shape) == 3, 'Inputs could not properly be encoded'
		assert len(self.inputs3.shape) == 3, 'Inputs could not properly be encoded'
		assert len(self.inputs4.shape) == 3, 'Inputs could not properly be encoded'
		assert len(self.inputs5.shape) == 3, 'Inputs could not properly be encoded'
		assert len(self.targets1.shape) == 3, 'Targets could not properly be encoded'
		assert len(self.targets2.shape) == 3, 'Targets could not properly be encoded'
