
import numpy as np
import keras.backend as K
from keras.layers import Bidirectional
from keras.layers.recurrent import GRU,

from preprocess import embed_sentence
from preprocess import get_positional_encoding

class InputModule(Layer):
    def __init__(self):
        pass

    def build(self, units, input_shape, question_shape, max_seq_len, emb_dim, emb_matrix):
        # TODO: GRU parameters

        self.facts_gru = Bidirectional(GRU(64))
        self.question_gru = GRU(units=units)
        self.positional_encoding = get_positional_encoding(max_seq_len, emb_dim)
        self.weight_matrix = emb_matrix
        self.built=True

    def call(self, inputs, question):

        fact_vectors = self.facts_gru.call(facts)[0]
        fact_vectors = K.sum(K.stack(fact_vectors), axis=0)

        question_vector = self.question_gru.call(question_vector)
        #TODO Add dropout?
        return fact_vectors, question_vector
