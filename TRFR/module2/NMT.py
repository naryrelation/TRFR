import numpy as np
import os
from keras.models import Model
from keras.layers import Dense, Embedding, Activation, Permute,concatenate,merge,Masking
from keras.layers import Input, Flatten, Dropout
from keras.layers.core import Reshape,Masking,Dropout
from keras.layers.recurrent import LSTM,GRU
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from .custom_recurrents import AttentionDecoder,AttentionLayer
from keras import regularizers
import tensorflow as tf
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

def simpleNMT(pad_length=100,
              n_chars=105,
              entity_labels=100,
              relation_labels=100,
              dim=100,
              embedding_learnable=False,
              encoder_units=256,
              decoder_units=256,
              trainable=True,
              return_probabilities=False):


	input1_ = Input(shape=(pad_length,dim,), dtype='float')
	input2_ = Input(shape=(pad_length,dim,), dtype='float')
	input3_ = Input(shape=(pad_length,dim,), dtype='float')
	input4_ = Input(shape=(pad_length,dim,), dtype='float')
	input5_ = Input(shape=(pad_length,dim,), dtype='float')

	rnn_1=GRU(encoder_units,return_sequences=True,name='gru1',input_shape=(pad_length,-1))(input1_)
	rnn_2 = GRU(encoder_units, return_sequences=True, name='gru2',input_shape=(pad_length,-1))(input2_)
	rnn_3 = GRU(encoder_units, return_sequences=True, name='gru3',input_shape=(pad_length,-1))(input3_)
	
	rnn_att1=AttentionLayer()(rnn_1)
	rnn_att2 = AttentionLayer()(rnn_2)
	rnn_att3 = AttentionLayer()(rnn_3)
	
	input4_4=Flatten()(input4_)
	input5_4 = Flatten()(input5_)
	input45=concatenate([input4_4,input5_4])
	input_45=Reshape((4,5,-1))(input45)
	cnn45=Conv2D(encoder_units,(2,2),activation='relu',name='conv1')(input_45)
	
	cnn45_flatten = Flatten()(cnn45)
	cnn_encoded45=Dense(encoder_units)(cnn45_flatten)
	
	rnn_output=concatenate([rnn_att1,rnn_att2,rnn_att3,cnn_encoded45])
	rnn1_output=Reshape((-1,encoder_units))(rnn_output)

	
	rnn2_output = GRU(encoder_units, return_sequences=True, name='gru4')(rnn1_output)
	
	y_hat1=AttentionLayer()(rnn2_output)

	
	y_dense1 = Dense(units=relation_labels)(y_hat1)
	
	y_final1=Activation('softmax')(y_dense1)
	y_dense11 = Dropout(0.01)(y_final1)
	y_out1=Reshape((-1,relation_labels))(y_dense11)

	
	model = Model(inputs=[input1_,input2_,input3_,input4_,input5_], outputs=[y_out1])

	return model


if __name__ == '__main__':
	model = simpleNMT()
	model.summary()
