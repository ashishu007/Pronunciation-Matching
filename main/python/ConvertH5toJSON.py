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
model = load_model('Australian_Test.h5')
json_string = model.to_json()
text_file = open('Australian_Test.json', 'w')
text_file.write(json_string)
text_file.close()
model = load_model('Bag_Test.h5')
json_string = model.to_json()
text_file = open('Bag_Test.json', 'w')
text_file.write(json_string)
text_file.close()