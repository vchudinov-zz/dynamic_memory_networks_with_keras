import numpy as np
import keras.backend as K
from keras.layers import Bidirectional, Dropout
from keras.layers.recurrent import GRU
from keras.engine.topology import Layer
from keras.engine import InputSpec

class InputModule(Layer):

    def __init__(self, units, dropout=0.0):
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
        super(InputModule, self).__init__()
        self.units = units
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

        self.question_gru = GRU(units=units, return_sequences=False)
        self.name = "Input_Module"

    def compute_output_shape(self, input_shape):
        """
        """
        
        out_shape = list(input_shape[0])
        out_shape[-1] = self.units
        q_shape = list(input_shape[1])
        q_shape[-1] = self.units
        q_shape = [q_shape[0], q_shape[-1]]
        return [out_shape, q_shape]

    def build(self,input_shape):
        """
        Initialize the layer.
        """
        self.input_spec = [InputSpec(shape=input_shape[0]), InputSpec(shape=input_shape[1])]
        # TODO: GRU parameters
        super(InputModule, self).build(input_shape)

    def call(self, inputs_list):

        inputs = inputs_list[0]
        question = inputs_list[1]

        fact_vectors = self.facts_gru(inputs)
        question_vector = self.question_gru(question)

        if self.dropout is not None:
            fact_vectors = self.dropout.call(fact_vectors)
            question_vector = self.dropout.call(question_vector)

        shapes = self.compute_output_shape([inputs.get_shape(),question.get_shape()])
        fact_vectors.set_shape(shapes[0])
        question_vector.set_shape(shapes[1])

        return [fact_vectors, question_vector]
