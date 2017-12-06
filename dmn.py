
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

    def __init__(self, save_folder,
                 input_shape,
                 question_shape,
                 num_classes,
                 units=256,batch_size=32,
                 memory_steps=3,
                 dropout=0.1,
                 l_2=1e-3,
                 weights=None):

        """Short summary.

        Parameters
        ----------
        save_folder : (str)
            Locaiton where the model and logs will be saved

        Returns
        -------
        None
        """
        self.save_folder = save_folder
        self.model_path = os.path.join(save_folder, "dmn-{epoch:02d}")
        self.log_folder = os.path.join(save_folder, "log")
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)

        self.build_inference_graph(input_shape,
                                  question_shape,
                                  num_classes,
                                  units,
                                  batch_size,
                                  memory_steps,
                                  dropout,
                                  l_2,
                                  weights)
        self.model_compiled = False


    def fit(self,
            train_x,
            train_q,
            train_y,
            epochs=256,
            validation_split=0.15,
            l_rate=1e-3,
            l_decay=0,
            save_criteria='val_loss',
            save_criteria_mode='min'
            ):
        """
        Trains the DMN model. Will save the model based on save_criteria and mode.

        Parameters
        ----------
        train_x : (np.array)
            np.array containing the input examples. Each example should have dimensions (time_steps, emb_size)
        train_q : (np.array)
            np.array containing the question examples. Each example should have dimensions (time_steps, emb_size)
        train_y : (np.array)
            An array holding the labels for each example. Each label must be a one-hot-vector with length (num_classes)
        epochs : (int)
            Number of epochs to train
        l_rate : (float)
            Learning rate
        l_decay : (float)
            The decay that will be applied to the learning rate
        validation_split : (float)
            Proportion of the dataset to reserve for validation. If left at 0 the
            entire set will be used for training

        Returns
        -------
        (keras.callbacks)
            The training history as a keras callback object.

        """



        checkpoint = keras.callbacks.ModelCheckpoint(self.model_path,
                                                     monitor=save_criteria,
                                                     verbose=1,
                                                     save_best_only=True,
                                                     save_weights_only=True,
                                                     mode=save_criteria_mode,
                                                     period=1)

        logger = keras.callbacks.CSVLogger(
            os.path.join(
                self.log_folder,
                "log.csv"),
            separator=',',
            append=False)

        stopper = keras.callbacks.EarlyStopping(monitor="loss",
                                                mode='min',
                                                patience=25,
                                                min_delta=1e-4

                                              )

        self.compile_model(l_rate, l_decay)
        print(f'Metrics: {self.model.metrics_names}')
        train_history = self.model.fit(x={'input_tensor': train_x,
                                          'question_tensor': train_q},
                                       y=train_y,
                                       callbacks=[logger, checkpoint, stopper],
                                       batch_size=self.batch_size,
                                       validation_split=validation_split,
                                       epochs=epochs)
        self.model.save_weights(self.model_path + "_trained")
        return train_history

    def compile_model(self, l_rate, l_decay):
        assert self.model is not None
        if not self.model_compiled:
            opt = optimizers.Adam(lr=l_rate, decay=l_decay, clipvalue=10.)
            self.model.compile(
                optimizer=opt,
                loss="categorical_crossentropy",
                metrics=["categorical_accuracy"])
            self.model_compiled = True

    def validate_model(self, x_val, xq_val, y_val):
        """
        Validates a model on supplied validation set.

        Parameters
        ----------
        x_val : (np.array)
            np.array containing the input examples. Each example should have dimensions (time_steps, emb_size)
        xq_val : (np.array)
            np.array containing the question examples. Each example should have dimensions (time_steps, emb_size)
        y_val : (np.array)
            An array holding the labels for each example. Each label must be a one-hot-vector with length (num_classes)

        Returns
        -------
        (float)
            Validation loss
        (float)
            Validation set accuracy
        """

        self.compile_model(0.0, 0.0)

        loss, acc = model.evaluate([x_val, xq_val], y_val,
                                   batch_size=self.batch_size)

        return loss, acc

    def predict(self, x, xq, batch_size=1):
        self.compile_model(0.0, 0.0)
        return self.model.predict([x, xq], batch_size=batch_size)

    def build_inference_graph(self, input_shape, question_shape, num_classes,
                              units=256,batch_size=32, memory_steps=3, dropout=0.1, l_2=1e-3, weights=None):
        """Builds the model.

        Parameters
        ----------
        input_shape : (tuple) or (list)
            A tuple or list specifying the shape of an individual input example
            Excludes batch size
        question_shape : (tuple) or (list)
            A tuple or list specifying the shape of an individual question example.
            Excludes batch size
        num_classes : (int)
            The number of possible output classes
        units : (int)
            Number of hidden units. Used for all layers
        batch_size : (int)
            Batch Size.
        memory_steps : (int)
            Number of steps to take when generating new memories
        dropout : (float)
            The dropout rate for the model.
        l_2 : (float)
            L_2 regularization is applied on all layers. This parameter specifies the rate
        weights : (str)
            Path to saved model weights. If specified, will attempt to load them into the model.
            Leave to None for new models.

        Returns
        -------
        None
        """

        assert(batch_size is not None)

        emb_dim = input_shape[-1]
        self.batch_size = batch_size

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
                        kernel_regularizer=regularizers.l2(l_2),
                        recurrent_regularizer=regularizers.l2(l_2)
                        )

        facts = Bidirectional(gru_layer, merge_mode='sum')(inputs_tensor)
        facts = Dropout(dropout)(facts)

        # Fix the time dimension
        facts_shape = list(K.int_shape(facts))
        facts_shape[1] = input_shape[0]
        facts.set_shape(facts_shape)

        question = GRU(units=units, stateful=True, return_sequences=False,
                       batch_size=batch_size,
                       kernel_regularizer=regularizers.l2(l_2),
                       recurrent_regularizer=regularizers.l2(l_2))(question_tensor)

        answer = EpisodicMemoryModule(
            units=units,
            batch_size=batch_size,
            emb_dim=emb_dim,
            memory_steps=memory_steps,
            regularization=l_2)([facts, question])

        answer = Dropout(dropout)(answer)

        answer = Dense(
            units=num_classes,
            batch_size=batch_size,
            activation="softmax",
            kernel_regularizer=regularizers.l2(l_2))(answer)

        self.model = Model(
            inputs=[
                inputs_tensor,
                question_tensor],
            outputs=answer)
        # If weights is passed load them into the model.
        if weights is not None:
            self.model.load_weights(weights)
