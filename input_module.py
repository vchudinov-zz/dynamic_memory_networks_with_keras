import numpy as np
import keras.backend as K
from keras.layers import Bidirectional, Dropout
from keras.layers.recurrent import GRU
from keras.engine.topology import Layer

class InputModule(Layer):

    def __init__(self, units, input_shape=None, question_shape=None,  dropout=0.0):
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
        self.units = units
        self.build(units, dropout)

    def build(self, units, dropout=0.0):
        """
        Initialize the layer.
        """
        # TODO: GRU parameters
        gru_layer = GRU(units=units,
                        activation='tanh',
                        use_bias=True,
                        dropout = dropout,
                        return_sequences=True
                        )

        self.facts_gru = Bidirectional(gru_layer, merge_mode='sum')

        if dropout >= 0:
            # TODO: Does this make sense for both question and input? No.
            self.dropout = Dropout(rate=dropout)

        self.question_gru = GRU(units=units,
                                return_sequences=True)
        self.name = "Input_Module"
        self.built=True
        print("Built")

    def __call__(self, inputs_list):

        inputs = inputs_list[0]
        question = inputs_list[1]

        fact_vectors = self.facts_gru(inputs)
        f_shape = list(inputs.get_shape())
        f_shape[-1] = self.units
        fact_vectors.set_shape(f_shape)
        question_vector = self.question_gru(question)

        if self.dropout is not None:
            fact_vectors = self.dropout.call(fact_vectors)
            question_vector = self.dropout.call(question_vector)

        return fact_vectors, question_vector
