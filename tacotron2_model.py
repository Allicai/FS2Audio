# Method to create an instance of the tacotron2 model we will be working with.

import tensorflow as tf
from tensorflow_tts.models import Tacotron2
from tensorflow_tts.utils import return_strategy

def create_tacotron2_model():
    return Tacotron2(num_mels=80, 
                     encoder_dim=512,
                     decoder_dim=1024,
                     postnet_dim=512,
                     encoder_type="lstm",
                     decoder_type="lstm",
                     reduction_factor=1,
                     num_heads=4,
                     max_position_embeddings=2048,
                     embedding_hidden_size=512,
                     embedding_dropout_prob=0.1,
                     num_conv1d_bank=8,
                     num_conv1d_proj=4,
                     encoder_conv_filters=512,
                     encoder_conv_kernel_sizes=5,
                     encoder_conv_activation='mish',
                     decoder_conv_filters=512,
                     decoder_conv_kernel_sizes=5,
                     decoder_conv_activation='mish',
                     duration_predictor_filters=256,
                     duration_predictor_kernel_size=5,
                     duration_predictor_dropout_prob=0.1,
                     n_frames_per_step=1,
                     decoder_rnn_dim=1024,
                     prenet_dim=256,
                     max_decoder_steps=2000,
                     gate_threshold=0.5,
                     p_attention_dropout=0.1,
                     p_decoder_dropout=0.1,
                     attention_type="bah_mon_norm",
                     )
