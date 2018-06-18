import sys, os, time
import wave
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils
from bottle import Bottle, route, run, request, response, BaseRequest
import json

app = Bottle()
BaseRequest.MEMFILE_MAX = 1024000000

def pre():
	return "hello client!"

def predict(data):

	name = data['name']
	syllable = data['syllable']
	prediction_model_file = "C:\\jython\\Pronunciation\\Pronunciation-Matching\\main\\python\\H5\\" + name + ".h5"
	prediction_model_file = "C:\\jython\\Pronunciation\\Pronunciation-Matching\\main\\python\\H5\\" + "Bag" + ".h5"

	SAMPLE_RATE = 44100
	CHUNK_SIZE = 1024

	no_of_files = 15
	training_percentage = 0.7
	sequence_length = 50
	embedding_vecor_length = 32

	model = Sequential()
	model.add(Embedding(50, embedding_vecor_length, input_length=sequence_length))
	model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	model.add(LSTM(100))
	model.add(Dense(2, activation='softmax'))
	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.load_weights(prediction_model_file)

	s_array = np.asarray(syllable)
	s_array_new = np.reshape(s_array, (-1, 1))
	s_use = np.transpose(s_array_new)

	predictions = model.predict(s_use)

	true = predictions[0][0]*100
	false = predictions[0][1]*100
	return true

@app.hook('after_request')
def enable_cors():
	response.headers['Access-Control-Allow-Origin'] = '*'
	response.headers['Access-Control-Allow-Methods'] = 'PUT, GET, POST, DELETE, OPTIONS'
	response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'

@app.route('/')
def hello():
	return 'Hello World!'

@app.route('/api/test', method=['OPTIONS', 'POST'])
def apitest():
	if request.method == 'POST':
		data = request.json
		print(type(data))
		print(data)
		res = {"true": predict(data)}
		return res
	else:
		return 'ok'

app.run(port=5000, debug=True)