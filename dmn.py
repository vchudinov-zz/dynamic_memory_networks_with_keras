import keras
from keras import backend as K
import tensorflow as tf
import numpy as np
import os
from keras.models import Model
from keras.layers import Input, Concatenate
from keras import optimizers
from input_module import InputModule
from episodic_memory_module import EpisodicMemoryModule

# TODO: Maybe extend keras.Model
# TODO: Define optimizer, epochs, training
# TODO: Define loading, processing data
# TODO: Metrics and logging
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
            save_criteria='train_loss'):


        opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer=opt, loss =self.loss, metrics=["accuracy"])

        checkpoint = keras.callbacks.ModelCheckpoint(self.model_folder,
                                            monitor=save_criteria,
                                            verbose=0,
                                            save_best_only=True,
                                            save_weights_only=False,
                                            mode='auto',
                                            period=1)

        logger = keras.callbacks.CSVLogger(self.log_location, separator=',', append=False)

        train_history = model.fit([x, xq], y,
            callbacks = [checkpoint, logger],
            validation_split = 0.1,
            batch_size=batch_size,
            epochs=epochs)

        return train_history


    def validate_model(self, x, xq, y):
        loss, acc = model.evaluate([x, xq], y,
                    batch_size=batch_size)
        return loss, acc

    def build_inference_graph(self, raw_inputs, question ):

        inputs_tensor = Input(shape = raw_inputs[0].shape)

        question_tensor = Input(shape = question[0].shape)

        facts, question = InputModule( input_shape=raw_inputs.shape,
                                       question_shape=question.shape,
                                       units=16,
                                       dropout=0.0)([inputs_tensor, question_tensor])
        print(facts.get_shape())
        print("----------------")
        memory = EpisodicMemoryModule(attn_units=64,
                                      attention_type='soft',
                                      memory_units=64,
                                      memory_type='RELU',
                                      memory_steps=self.max_seq)([facts, question])
        # Embeddings variant.
        #answer = layers.Dense(units=self.vocab_size, activation=None)(K.concatenate(memory,question))

        # One hot variant
        concatenation = Concatenate()([memory, question])
        answer = layers.Dense(units=self.num_classes, activation='softmax')(concatenation)
        prediction = K.argmax(answer,1)
        # TODO: train vs. use. Correct output.
        self.model = Model(inputs=[facts, question], outputs=answer)

    def save_model(self, path):
        #model_arch = self.model.to_json()
        #TODO dump model_arch
        self.model.save_model(path)

# Convert labels to categorical one-hot encoding
#
#one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)
