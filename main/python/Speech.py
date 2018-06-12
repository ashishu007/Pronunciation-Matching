import os
from pocketsphinx import AudioFile, get_model_path, get_data_path

import subprocess

def play(audio_file_path):
    subprocess.call(["ffplay", "-nodisp", "-autoexit", audio_file_path])

play('/home/acer/Downloads/australian1.wav')
  
model_path = get_model_path()
data_path = get_data_path()

config = {
    'verbose': False,
    'audio_file': '/home/acer/Downloads/australian2.wav',
    'buffer_size': 2048,
    'no_search': False,
    'full_utt': False,
    'hmm': os.path.join(model_path, 'en-us'),
    'lm': os.path.join(model_path, 'en-us.lm.bin'),
    'dict': os.path.join(model_path, 'cmudict-en-us.dict')
}

audio = AudioFile(**config)
for phrase in audio:
    print(phrase)
    

