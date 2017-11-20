
import keras
import numpy as np
import os
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras import optimizers
from episodic_memory_module import EpisodicMemoryModule
from keras.layers import Bidirectional, Dropout
from keras.layers.recurrent import GRU
from keras import regularizers

from keras import backend as K
# TODO: Saving and deploying
# TODO: What about recurrent activation of memories?
# TODO: train vs. use. Correct output.


class DynamicMemoryNetwork():
    """
    An attempt to implement the Dynamic Memory Network from https://arxiv.org/pdf/1603.01417.pdf using keras
    """

    def __init__(self, save_folder):
        """Short summary.

        Parameters
        ----------
        save_folder : type
            Description of parameter `save_folder`.

        Returns
        -------
        type
            Description of returned object.

        """
        self.save_folder = save_folder
        self.model_path = os.path.join(save_folder, "dmn")
        self.log_folder = os.path.join(save_folder, "log")
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)

    def fit(self,
            train_x,
            train_q,
            train_y,
            batch_size=32,
            epochs=10,
            validation_split=0.15,
            l_rate=1e-3,
            l_decay=0,
            loss="categorical_crossentropy"
            save_criteria='val_loss'
            ):
        """Short summary.

        Parameters
        ----------
        train_x : type
            Description of parameter `train_x`.
        train_q : type
            Description of parameter `train_q`.
        train_y : type
            Description of parameter `train_y`.
        batch_size : type
            Description of parameter `batch_size`.
        epochs : type
            Description of parameter `epochs`.
        l_rate : type
            Description of parameter `l_rate`.
        l_decay : type
            Description of parameter `l_decay`.
        loss : type
            Description of parameter `loss`.
        validation_split : type
            Description of parameter `validation_split`.

        Returns
        -------
        type
            Description of returned object.

        """

        opt = optimizers.Adam(lr=l_rate, decay=l_decay)
        #labels =  to_categorical(np.array(train_y), num_classes=None)
        checkpoint = keras.callbacks.ModelCheckpoint(self.model_path,
                                                     monitor='categorical_accuracy',
                                                     verbose=1,
                                                     save_best_only=True,
                                                     save_weights_only=False,
                                                     mode='max',
                                                     period=1)

        logger = keras.callbacks.CSVLogger(
            os.path.join(
                self.log_folder,
                "log.csv"),
            separator=',',
            append=False)

        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=["categorical_accuracy"])
        train_history = self.model.fit(x={'input_tensor': train_x,
                                          'question_tensor': train_q},
                                       y=train_y,
                                       callbacks=[logger, checkpoint],
                                       batch_size=batch_size,
                                       validation_split=validation_split,
                                       epochs=epochs)

        return train_history

    def validate_model(self, x_val, xq_val, y_val):
        loss, acc = model.evaluate([x, xq], y,
                                   batch_size=batch_size)
        return loss, acc

    def build_inference_graph(self, input_shape, question_shape, num_classes,
                              units=256, emb_dim=100, batch_size=32, memory_steps=3, dropout=0.):
        """Short summary.

        Parameters
        ----------
        input_shape : type
            Description of parameter `input_shape`.
        question_shape : type
            Description of parameter `question_shape`.
        num_classes : type
            Description of parameter `num_classes`.
        units : type
            Description of parameter `units`.
        emb_dim : type
            Description of parameter `emb_dim`.
        batch_size : type
            Description of parameter `batch_size`.
        memory_steps : type
            Description of parameter `memory_steps`.
        dropout : type
            Description of parameter `dropout`.

        Returns
        -------
        type
            Description of returned object.

        """

        assert(batch_size is not None)

        inputs_tensor = Input(
            batch_shape=(
                batch_size,
            ) + input_shape,
            name='input_tensor')
        question_tensor = Input(
            batch_shape=(
                batch_size,
            ) + question_shape,
            name='question_tensor')

        gru_layer = GRU(units=units,
                        dropout=dropout,
                        return_sequences=True,
                        stateful=True,
                        batch_size=batch_size,
                        kernel_regularizer=regularizers.l2(0.01),
                        recurrent_regularizer=regularizers.l2(0.01)
                        )

        facts = Bidirectional(gru_layer, merge_mode='sum')(inputs_tensor)
        facts = Dropout(dropout)(facts)

        # Fix the time dimension
        facts_shape = list(K.int_shape(facts))
        facts_shape[1] = input_shape[0]
        facts.set_shape(facts_shape)

        question = GRU(units=units, stateful=True, return_sequences=False,
                       batch_size=batch_size,
                       kernel_regularizer=regularizers.l2(0.01),
                       recurrent_regularizer=regularizers.l2(0.01))(question_tensor)

        answer = EpisodicMemoryModule(
            units=units,
            batch_size=batch_size
            emb_dim=emb_dim,
            memory_steps=mem_hops)([facts, question])

        answer = Dropout(dropout)(answer)
        answer = Dense(
            units=num_classes,
            batch_size=batch_size,
            activation="softmax")(answer)

        self.model = Model(
            inputs=[
                inputs_tensor,
                question_tensor],
            outputs=answer)
