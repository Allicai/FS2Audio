'''
Script to train our tacotron2 model. 

Generally using libraries from class, with some optimizers we've discussed like Adam.
Additionally had to do some research on mel spectrograms which are very relevant to audio 
generation. Trying to use the train_test_split method still, but metadata.csv from the 
LJSpeech dataset is formatted very oddly.
'''

import tensorflow as tf
import numpy as np
import pandas as pd
from tacotron2_model import create_tacotron2_model
from tensorflow_tts.optimizers import AdamWeightDecay
from tensorflow_tts.losses import TacotronLoss
from tensorflow_tts.utils import Char2MelProcessor
from sklearn.model_selection import train_test_split

# loading the data
# metadata.csv has two transcriptions for each audio file - split the csv by '|'.
metadata = pd.read_csv("metadata.csv", sep='|', names=["ID", "transcription1", "transcription2"])

# creating a new column for audio paths 
# metadata.csv contains 'ID's, which are the names of the audio files in the wavs dir
# for example, the ID LJ001-0001 correlates to LJ001-0001.wav in the wavs folder, so we append
# .wav to the end of the ID and get the audio
metadata['audio_path'] = metadata['ID'].apply(lambda x: f'wavs/{x}.wav')

# using train_test_split to generate training vs testing data
train_metadata, val_metadata = train_test_split(metadata, test_size=0.1, random_state=42)

# hyperparameters
BATCH_SIZE = 8
EPOCHS = 50

# initializing a tacotron2 model
tacotron2 = create_tacotron2_model()

# defining our optimizer - read on the Adam optimizer from class 
optimizer = AdamWeightDecay(learning_rate=1e-4, weight_decay_rate=1e-6)

# compiling the model
tacotron2.compile(optimizer=optimizer, loss=TacotronLoss())

# training the model
history = tacotron2.fit(train_metadata["transcription1"], train_metadata["audio_path"],
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(val_metadata["transcription1"], val_metadata["audio_path"]))
