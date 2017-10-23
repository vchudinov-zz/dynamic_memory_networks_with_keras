
import numpy as np
import keras.backend as K
from keras.layers import Bidirectional, Dropout
from keras.layers.recurrent import GRU
from keras.engine.topology import Layer

class InputModule(Layer):

    def __init__(self, units, input_shape, question_shape,  dropout):
        """
        Contains the input module for the DMN+
        The input module consists of a "diffusion layer" - a bidirectional GRU
        into which the input facts are fed and a question layer -
        a GRU layer that takes the question representation as input
        Args:
            units (int) : number of units in the gru layers. Same for input and question sub-modules
            input_shape (tuple) or (list) : defines the shape of the input facts. should be [batch_size, dims]
            question-shape (tuple) or (list) : defines the shape of the question. should be [batch_size, dims]
            dropout (float) : the dropout rate, between 0 and 1
        """
        # TODO Add other stuff
        self.build(units, input_shape, question_shape, dropout)

    def build(self, units, input_shape, question_shape, dropout=0.0):
        """
        Initialize the layer.
        """
        # TODO: GRU parameters
        gru_layer = GRU(input_shape = input_shape,
                        units=units,
                        activation='tanh',
                        use_bias=True,
                        dropout = dropout
                        )

        self.facts_gru = Bidirectional(gru_layer)

        if dropout >= 0:
            # TODO: Does this make sense for both question and input? No.
            self.dropout = Dropout(rate=dropout)

        self.question_gru = GRU(input_shape=question_shape,
                                units=units)
        self.name = "Input_Module"
        self.built=True

    def call(self, inputs_list):
        inputs = inputs_list[0]
        question = inputs_list[1]
        fact_vectors = self.facts_gru.call(inputs)[0]
        fact_vectors = K.sum(K.stack(fact_vectors), axis=0)
        question_vector = self.question_gru.call(question_vector)[0]
        if self.dropout is not None:
            fact_vectors = self.dropout.call(fact_vectors)
            question_vector = self.dropout.call(question_vector)
        return fact_vectors, question_vector
