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
from keras.models import load_model

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
	# 'Bag',
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

no_of_files = 1

dataset_length = len(words) * len(speaker) * no_of_files

sequence_length = 50
embedding_vecor_length = 32

chars = sorted(list([' ','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z' , '1', '2', '3','4','5','6','7','8','9','0','-' ]))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
y_train = [-1]
X_train = []


j= 0
count = 0

file  = open("Prediction.txt", 'w+')
				
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

for row in X_train:
	model = load_model(word +'.h5')
	pred = model.fit(np.array(row),0,epochs=1, batch_size= 64)
	print(pred)
	file.write("prediction for word "+ word + " = " + str(pred))
	file.write("\n")
