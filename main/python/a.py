from pocketsphinx.pocketsphinx import *
from sphinxbase.sphinxbase import *
import speech_recognition as sr
import os

# obtain path to "english.wav" in the same folder as this script
AUDIO_FILE = "australian1.wav"

# use the audio file as the audio source
r = sr.Recognizer()
with sr.AudioFile(AUDIO_FILE) as source:
    audio = r.record(source)  # read the entire audio file
modeldir = "/home/acer/Downloads/cmusphinx-en-in-5.2" 


config = Decoder.default_config()
config.set_string('-hmm', os.path.join(modeldir, 'en-in'))
config.set_string('-lm', os.path.join(modeldir, 'en-in.lm.bin'))
config.set_string('-dict', os.path.join(modeldir, 'en-in.dic'))
config.set_float('-kws_threshold', 1e-5)
decoder = Decoder(config)

# recognize speech using Sphinx
try:
    print("Sphinx thinks you said " + r.recognize_sphinx(audio))
    data = r.recognize_sphinx(audio)
except sr.UnknownValueError:
    print("Sphinx could not understand audio")
except sr.RequestError as e:
    print("Sphinx error; {0}".format(e))
    
print(data)
print(decoder.lookup_word(data))
