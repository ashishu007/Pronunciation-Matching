#s### Example code for a single keyword that gets me a timestamp: 
import sys, os
from pocketsphinx.pocketsphinx import *
from sphinxbase.sphinxbase import *

modeldir = "/usr/local/lib/python2.7/dist-packages/pocketsphinx/model" 
datadir  = ""
modeldir = "/usr/local/lib/python2.7/dist-packages/pocketsphinx/model" 

stream = open('australian1.wav', 'rb')

config = Decoder.default_config()
config.set_string('-hmm', os.path.join(modeldir, 'en-us'))
config.set_string('-lm', os.path.join(modeldir, 'en-us.lm.bin'))
config.set_string('-dict', os.path.join(modeldir, 'cmudict-en-us.dict'))
config.set_float('-kws_threshold', 1e-5)
decoder = Decoder(config)

decoder.start_utt()
while True:
    buf = stream.read(1024)
    if buf:
         decoder.process_raw(buf, False, False)
    else:
         break
    if decoder.hyp() != None:
        print ([(seg.word, seg.prob, seg.start_frame, seg.end_frame) for seg in decoder.seg()])
        print ("Detected keyphrase, restarting search")
        decoder.end_utt()
        decoder.start_utt()
