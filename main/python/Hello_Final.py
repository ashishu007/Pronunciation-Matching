import sys, os, time
import wave
import numpy as np
from pocketsphinx.pocketsphinx import *
from sphinxbase.sphinxbase import *
import speech_recognition as sr
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils

SAMPLE_RATE = 44100
CHUNK_SIZE = 1024

no_of_files = 15
training_percentage = 0.7
sequence_length = 50
embedding_vecor_length = 32

modeldir = "H:/Pronunciation/Downloads/cmusphinx-en-in-5.2" 
wavfile_dir = 'H:/Pronunciation/Downloads'

speaker = [
	'Sathya',
	'Shruti'
]

words = [
	'Australian',
	'Bag',
	# 'Bank',
	# 'Blue',
	# 'Beard',
	# 'Book',
	# 'Brother',
	# 'Bus',
	# 'Canadian',
	# 'Chair',
	# 'Cheese',
	# 'Classes',
	# 'Day',
	# 'Fair',
	# 'Father',
	# 'Fifteen1',
	# 'Fifteen2',
	# 'Fifty',
	# 'First',
	# 'Five',
	# 'Four',
	# 'Funny',
	# 'Glasses',
	# 'Green',
	# 'Hair',
	# 'Hundred',
	# 'Japanese',
	# 'Jeans',
	# 'Korean',
	# 'Mother',
	# 'Name',
	# 'Plate1',
	# 'Plate2',
	# 'Play',
	# 'Sandwich',
	# 'Saturday',
	# 'Secretary',
	# 'Shelf',
	# 'Shirt',
	# 'Shoes',
	# 'Shopping',
	# 'Short',
	# 'Subway',
	# 'Sunday',
	# 'Table',
	# 'Taiwanese',
	# 'Tea',
	# 'Their',
	# 'There',
	# 'These',
	# 'They',
	# 'Thirteen',
	# 'Thirty',
	# 'Train',
	# 'Waitress',
	# 'Wallet',
	# 'Watch',
	# 'Well built',
	# 'Where',
	# 'Worker'
]

config = Decoder.default_config()
config.set_string('-hmm', os.path.join(modeldir, 'en-in'))
config.set_string('-lm', os.path.join(modeldir, 'en-in.lm.bin'))
config.set_string('-dict', os.path.join(modeldir, 'en-in.dic'))
config.set_float('-kws_threshold', 1e-5)
decoder = Decoder(config)

training_files = int(no_of_files * training_percentage)
dataset_length = len(words) * len(speaker) * no_of_files

model = Sequential()
model.add(Embedding(50, embedding_vecor_length, input_length=sequence_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(2, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

chars = sorted(list([' ','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z' , '1', '2', '3','4','5','6','7','8','9','0','-' ]))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

Train = []
speech_data = []

label= 0
count = 0
counter = 0

file = open("Syllable_Training.txt", 'w+')
				
for word in words:
	for person in speaker:
		for i in range(1,training_files + 1):

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

				if (index + CHUNK_SIZE) > filesize: 
					count = (filesize - index)
				else: 
					count = CHUNK_SIZE

				temp_data = data[index:(index+count)]

				decoder.process_raw(temp_data, False, False)

				if decoder.hyp() != None:

					detected_times = detected_times + 1

					syllable.append(decoder.lookup_word(decoder.hyp().hypstr))

					decoder.end_utt()
					decoder.start_utt()

				index = index + count

			decoder.end_utt()

			speech_data.append([word,syllable])

			try:
				seq = "".join(syllable)
			except:
				seq = " "

			for i in range(len(seq),sequence_length):
				seq = seq + " "

			numeric_syllable = [char_to_int[char] for char in seq]

			Train.append([numeric_syllable, label])

			counter = counter +1
			
			file.write("Syllable for word "+ word +str(counter)+" is" + str(syllable) + " " + str(numeric_syllable) + " with label " + str(label))
			file.write("\n")

			if (counter%(training_files*2))==0:
				label = label+1
			
file.close()

Test = []
speech_data_test = []

label= 0
count = 0
counter = 0

file = open("Syllable_Testing1.txt", 'w+')

for word in words:
	for person in speaker:
		for i in range(training_files + 1, no_of_files + 1):

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

				if (index + CHUNK_SIZE) > filesize: 
					count = (filesize - index)
				else: 
					count = CHUNK_SIZE

				temp_data = data[index:(index+count)]

				decoder.process_raw(temp_data, False, False)

				if decoder.hyp() != None:

					detected_times = detected_times + 1

					syllable.append(decoder.lookup_word(decoder.hyp().hypstr))

					decoder.end_utt()
					decoder.start_utt()

				index = index + count

			decoder.end_utt()

			speech_data.append([word,syllable])

			try:
				seq = "".join(syllable)
			except:
				seq = " "

			for i in range(len(seq),sequence_length):
				seq = seq + " "

			numeric_syllable = [char_to_int[char] for char in seq]

			Test.append([numeric_syllable, label])

			counter = counter +1
			
			file.write("Syllable for word "+ word +str(counter)+" is" + str(syllable) + " " + str(numeric_syllable) + " with label " + str(label))
			file.write("\n")

			if (counter%(training_files*2))==0:
				label = label+1

file.close()

all_model = []
result = []

file = open("Result1.txt","w+")
for i in range(len(words)):
	
	y_train = []
	y_test = []
	X_train = []
	X_test = []

	for row in Train:

		if row[1]==i:

			y_train.append(1)

			if len(X_train) ==0:
				X_train = row[0]
			else:
				X_train = np.vstack((X_train,row[0]))

		else:

			y_train.append(0)

			if len(X_train) ==0:
				X_train = row[0]
			else:
				X_train = np.vstack((X_train,row[0]))

	for row in Test:

		if row[1] == i:

			y_test.append(1)

			if len(X_test) ==0:
				X_test = row[0]
			else:
				X_test = np.vstack((X_test,row[0]))

		else:

			y_test.append(0)

			if len(X_test) ==0:
				X_test = row[0]
			else:
				X_test = np.vstack((X_test,row[0]))
		
	print("Model for " + words[i] + "...")
	file.write("Model for " + words[i] + "...\n")

	Model = model
	Model.fit(X_train ,y_train, epochs=100, batch_size=64)

	train_scores = Model.evaluate(X_train,y_train, verbose=0)
	print("Train Accuracy: %.2f%%" % (train_scores[1]*100))
	file.write("Train Accuracy: %.2f%% \n" % (train_scores[1]*100))
	
	test_scores = Model.evaluate(X_test,y_test, verbose=0)
	print("Test Accuracy: %.2f%%" % (test_scores[1]*100))
	file.write("Test Accuracy: %.2f%% \n" % (test_scores[1]*100))

	Model.save(words[i]+"_test.h5")
	all_model.append(Model)