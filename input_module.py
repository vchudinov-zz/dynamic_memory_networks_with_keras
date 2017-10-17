
import numpy as np
import keras.backend as K
from keras.layers import Bidirectional, Dropout
from keras.layers.recurrent import GRU,

from preprocess import embed_sentence
from preprocess import get_positional_encoding

class InputModule(Layer):
    def __init__(self, input_shape, question_shape, units, dropout):
        """
        Contains the input module for the DMN+
        The input module consists of a "diffusion layer" - a bidirectional GRU
        into which the input facts are fed and a question layer -
        a GRU layer that takes the question representation as input
        """
        # TODO Add other stuff
        self.build()

    def build(self, units, input_shape, question_shape, dropout=0.0):
        # TODO: GRU parameters
        gru_layer = GRU(input_shape = input_shape,
                        units=units,
                        activation='tanh',
                        use_bias= True,
                        kernel_initializer,
                        recurrent_initializer,
                        bias_initializer,
                        kernel_regularizer,
                        recurrent_regularizer,
                        bias_regularizer,
                        dropout,
                        recurrent dropout
                        )
        self.facts_gru = Bidirectional(gru_layer)
        if dropout >= 0:
            # TODO: Does this make sense for both question and input? No.
            self.dropout = Dropout(rate=dropout)
        self.question_gru = GRU(input_shape=question_shape,
                                units=units)

        self.built=True

    def call(self, inputs, question):

        fact_vectors = self.facts_gru.call(facts)[0]
        fact_vectors = K.sum(K.stack(fact_vectors), axis=0)
        question_vector = self.question_gru.call(question_vector)[0]
        if self.dropout is not None:
            fact_vectors = self.dropout.call(fact_vectors)
            question_vector = self.dropout.call(question_vector)
        return fact_vectors, question_vector
