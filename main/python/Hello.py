import sys, os
from pocketsphinx.pocketsphinx import *
from sphinxbase.sphinxbase import *
import wave
import time
import speech_recognition as sr
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils

SAMPLE_RATE = 44100
CHUNK_SIZE =  1024

modeldir = "/home/acer/Downloads/cmusphinx-en-in-5.2" 

config = Decoder.default_config()
config.set_string('-hmm', os.path.join(modeldir, 'en-in'))
config.set_string('-lm', os.path.join(modeldir, 'en-in.lm.bin'))
config.set_string('-dict', os.path.join(modeldir, 'en-in.dic'))
config.set_float('-kws_threshold', 1e-5)

decoder = Decoder(config)

wavfile_dir = '/home/acer/Downloads/'

wavfile_dir = '/home/acer/Pronunciation'

speaker = [
	'Sathya',
	'Shruti'
]

words = [
	'Australian',
	'Bag',
	'Bank',
	'Blue',
	'Beard',
	'Book',
	'Brother',
	'Bus',
	'Canadian',
	'Chair',
	'Cheese',
	'Classes',
	'Day',
	'Fair',
	'Father',
	'Fifteen1',
	'Fifteen2',
	'Fifty',
	'First',
	'Five',
	'Four',
	'Funny',
	'Glasses',
	'Green',
	'Hair',
	'Hundred',
	'Japanese',
	'Jeans',
	'Korean',
	'Mother',
	'Name',
	'Plate1',
	'Plate2',
	'Play',
	'Sandwich',
	'Saturday',
	'Secretary',
	'Shelf',
	'Shirt',
	'Shoes',
	'Shopping',
	'Short',
	'Subway',
	'Sunday',
	'Table',
	'Taiwanese',
	'Tea',
	'Their',
	'There',
	'These',
	'They',
	'Thirteen',
	'Thirty',
	'Train',
	'Waitress',
	'Wallet',
	'Watch',
	'Well built',
	'Where',
	'Worker'
]

no_of_files = 14

dataset_length = len(words) * len(speaker) * no_of_files

sequence_length = 50
embedding_vecor_length = 32

model = Sequential()
model.add(Embedding(50, embedding_vecor_length, input_length=sequence_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

chars = sorted(list([' ','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z' , '1', '2', '3','4','5','6','7','8','9','0','-' ]))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

y_train = [-1]
X_train = []

j= 0
count = 0

file  = open("Syllable.txt", 'w+')
				
speech_data = []
for person in speaker:
	for word in words:
		for i in range(1,no_of_files+1):
				file_name = person + "/" + word + "-" + str(i)
				wavfile = wavfile_dir + "/"+ file_name +'.wav'
				wf = wave.open(wavfile, 'rb');
				filesize = wf.getnframes()*2
				data = wf.readframes(filesize)
				decoder.start_utt()
				syllable = []
				index = 0
				detected_times = 0
				while index < filesize:
					if (index + CHUNK_SIZE) > filesize: count = (filesize - index)
					else: count = CHUNK_SIZE
					temp_data = data[index:(index+count)]
					decoder.process_raw(temp_data, False, False)
					if decoder.hyp() != None:
						detected_times = detected_times + 1
						syllable.append(decoder.lookup_word(decoder.hyp().hypstr))
						decoder.end_utt()
						decoder.start_utt()
					index = index + count
				speech_data.append([word,syllable])
				decoder.end_utt()

				y_train.append(j)
				try:
				   seq = "".join(syllable)
				except:
					seq = " "
				for i in range(len(seq),sequence_length):
					seq = seq + " "
				syl = [char_to_int[char] for char in seq]
				X_train.append(syl)
				count = count +1
				if count%2==0:
					j = j+1

				file.write("Syllable for word "+ word +" is" + str(syl))
				file.write("\n")

file.close()

X_test = []
				
speech_data_test = []
for person in speaker:
	for word in words:
		file_name = person + "/" + word + "-" + str(15)
		wavfile = wavfile_dir + "/"+ file_name +'.wav'
		wf = wave.open(wavfile, 'rb');
		filesize = wf.getnframes()*2
		data = wf.readframes(filesize)
		decoder.start_utt()
		syllable = []
		index = 0
		detected_times = 0
		while index < filesize:
			if (index + CHUNK_SIZE) > filesize: count = (filesize - index)
			else: count = CHUNK_SIZE
			temp_data = data[index:(index+count)]
			decoder.process_raw(temp_data, False, False)
			if decoder.hyp() != None:
				detected_times = detected_times + 1
				syllable.append(decoder.lookup_word(decoder.hyp().hypstr))
				decoder.end_utt()
				decoder.start_utt()
			index = index + count
		speech_data_test.append([word,syllable])
		decoder.end_utt()

		try:
		   seq = "".join(syllable)
		except:
			seq = " "
		for i in range(len(seq),sequence_length):
			seq = seq + " "
		syl = [char_to_int[char] for char in seq]
		X_test.append(syl)
		

X_train_1 = [i-i for i in range(1,sequence_length+1)]
X_test_1 = [i-i for i in range(1,sequence_length+1)]

for row in X_train:
	X_train_1 = np.vstack((X_train_1,row))

for row in X_test:
	X_test_1 = np.vstack((X_test_1,row))

all_model = []

for i in range(len(words)):

	Model = model
	y_train_1 = []

	for data in y_train:
		if data == i:
			y_train_1.append(0)
		else:
			y_train_1.append(1)

	y_train_1 = np_utils.to_categorical(y_train_1)

	print("Model for " + words[i] + "...")
	Model.fit(X_train_1 ,y_train_1, epochs=30, batch_size=64)
	scores = Model.evaluate(X_train_1,y_train_1, verbose=0)
	pred = Model.predict(X_train_1)
	print("Accuracy: %.2f%%" % (scores[1]*100))
	print("Pred" + " " + str(pred))
	pred = Model.predict(X_test_1)
	print("Pred" + " " + str(pred))
	Model.save(words[i]+".h5")
	all_model.append(Model)

