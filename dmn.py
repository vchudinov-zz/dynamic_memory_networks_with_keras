
import keras
import numpy as np
import os
from keras.models import Model, Sequential
from keras.layers import Input, Concatenate, Dense, Dropout
from keras import optimizers
from input_module import InputModule
from episodic_memory_module import EpisodicMemoryModule
from keras.layers import Bidirectional, Dropout
from keras.layers.recurrent import GRU
from keras.engine.topology import Layer
from keras.engine import InputSpec
from keras import regularizers

from keras import backend as K
# TODO: Maybe extend keras.Model
# TODO: Saving and deploying
# TODO: What about recurrent activation of memories?
# TODO: train vs. use. Correct output.

class DynamicMemoryNetwork():
    """
    An attempt to implement the Dynamic Memory Network from https://arxiv.org/pdf/1603.01417.pdf using keras
    """
    def __init__(self, model_folder, input_units, memory_units, output_units, max_seq, attention_type='soft', memory_type='RELU'):
        self.model_folder = model_folder
        self.input_units = input_units
        self.memory_units = memory_units
        self.output_units = output_units
        self.max_seq = max_seq
        self.attention_type = attention_type
        self.memory_type = memory_type
        self.log_folder = os.path.join(model_folder, "log")
        self.num_classes = output_units

        self.loss = "categorical_crossentropy"
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)


    def fit(self,
            train_x,
            train_q,
            train_y,
            batch_size=32,
            epochs=10,
            l_rate=1e-3,
            l_decay=0,
            save_criteria='train_loss',
            validation_split = 0.):


        opt = optimizers.Adam(lr=0.001)
        #labels =  to_categorical(np.array(train_y), num_classes=None)
        checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(self.model_folder, "dmn"),
                                            monitor='categorical_accuracy',
                                            verbose=1,
                                            save_best_only=True,
                                            save_weights_only=False,
                                            mode='max',
                                            period=1)

        logger = keras.callbacks.CSVLogger(os.path.join(self.log_folder, "log.log"), separator=',', append=False)
        self.model.compile(optimizer=opt, loss =self.loss, metrics=["categorical_accuracy"])


        train_history = self.model.fit( x={ 'input_tensor':np.array(train_x),
                                            'question_tensor':np.array(train_q)}, y=np.array(train_y),
            callbacks = [logger, checkpoint],
            batch_size=batch_size,
            validation_split=validation_split,
            epochs=epochs)

        return train_history


    def validate_model(self, x, xq, y):
        loss, acc = model.evaluate([x, xq], y,
                    batch_size=batch_size)
        return loss, acc

    def build_inference_graph(self, raw_inputs, question, units=256, batch_size=32, dropout=0.):
        assert(batch_size is not None)

        inputs_tensor = Input(batch_shape= (batch_size,) +  raw_inputs[0].shape, name='input_tensor')
        question_tensor = Input(batch_shape= (batch_size,) +  question[0].shape, name='question_tensor')


        gru_layer = GRU(units=units,
                        dropout = dropout,
                        return_sequences=True,
                        stateful=True,
                        batch_size=batch_size,
                        kernel_regularizer=regularizers.l2(0.01),
                        recurrent_regularizer= regularizers.l2(0.01)
                        )
        
        facts = Bidirectional(gru_layer, merge_mode='sum')(inputs_tensor)
        facts_shape = list(K.int_shape(facts))
        facts_shape[1] = raw_inputs[0].shape[0]
        facts.set_shape(facts_shape)

        question = GRU(units=units, stateful=True, return_sequences=False,
                                batch_size=batch_size,
                                kernel_regularizer=regularizers.l2(0.01),
                                recurrent_regularizer= regularizers.l2(0.01))(question_tensor)

        answer = EpisodicMemoryModule(
                                      units=units,
                                      batch_size=batch_size,
                                      dropout=dropout,
                                      memory_type='RELU',
                                      emb_dim=100,
                                      memory_steps=self.max_seq)([facts, question])
        if dropout > 0.0:
            answer = Dropout(rate=dropout)(answer)

        answer = Dense(units=self.num_classes, batch_size=batch_size, activation="softmax")(answer)

        self.model = Model(inputs=[inputs_tensor, question_tensor], outputs=answer)

    def get_positional_encoding(self, max_seq, emb_dim):
        """
        Generates  a positional encoding as described in section
        4.1 in "End to End Memory Networks" (http://arxiv.org/pdf/1503.08895v5.pdf)
        Adapted from https://github.com/domluna/memn2n and
        https://github.com/barronalex/Dynamic-Memory-Networks-in-TensorFlow/
        Args:
            max_seq (int) : the maximum sequence size in texts. Data will be padded to that size
            emb_dim (int) : The size of the embedding vectors.
        Return:
            (np.array) with the positional encoding.
        """

        encoding = np.ones((emb_dim, max_seq), dtype=np.float32)
        ls = max_seq+1
        le = emb_dim+1
        for i in range(1, le):
            for j in range(1, ls):
                encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
        encoding = 1 + 4 * encoding / emb_dim / max_seq
        encoding[:, -1] = 1.0 # TODO maybe remove

        return np.transpose(encoding)
