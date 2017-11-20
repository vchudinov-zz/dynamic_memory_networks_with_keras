
import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import (Bidirectional, Dense)
from attention_cells import SoftAttnGRU

from keras import regularizers


class EpisodicMemoryModule(Layer):

    def __init__(self, units, memory_steps, emb_dim,
                 batch_size, dropout=0.0, **kwargs):
        """Short summary.

        Parameters
        ----------
        units : type
            Description of parameter `units`.
        memory_steps : type
            Description of parameter `memory_steps`.
        emb_dim : type
            Description of parameter `emb_dim`.
        batch_size : type
            Description of parameter `batch_size`.
        dropout : type
            Description of parameter `dropout`.
        **kwargs : type
            Description of parameter `**kwargs`.

        Returns
        -------
        type
            Description of returned object.

        """

        # TODO: Dropout

        self.memory_steps = memory_steps
        self.dropout = dropout
        self.name = "episodic_memory_module"
        self._input_map = {}
        self.supports_masking = True
        self.units = units

        # attention net.
        self.l_1 = Dense(units=emb_dim,
                         batch_size=batch_size,
                         activation='tanh',
                         kernel_regularizer=regularizers.l2(0.01))

        self.l_2 = Dense(units=1,
                         batch_size=batch_size,
                         activation=None,
                         kernel_regularizer=regularizers.l2(0.01))

        # Episode net
        self.episode_GRU = SoftAttnGRU(units=units,
                                       return_sequences=False,
                                       batch_size=batch_size,
                                       kernel_regularizer=regularizers.l2(
                                           0.01),
                                       recurrent_regularizer=regularizers.l2(0.01))

        # Memory generating net.
        self.memory_net = Dense(units=units,
                                activation='relu',
                                kernel_regularizer=regularizers.l2(0.01))

        super(EpisodicMemoryModule, self).__init__()

    def compute_output_shape(self, input_shape):

        q_shape = list(input_shape[1])
        q_shape[-1] = self.units * 2
        return tuple(q_shape)

    def build(self, input_shape):
        super(EpisodicMemoryModule, self).build(input_shape)

    def call(self, inputs):
        """Short summary.

        Parameters
        ----------
        inputs : type
            Description of parameter `inputs`.

        Returns
        -------
        type
            Description of returned object.

        """
        # TODO: Add Dropout and BatchNorm.

        def compute_attention(fact, question, memory):
            """Short summary.

            Parameters
            ----------
            fact : type
                Description of parameter `fact`.
            question : type
                Description of parameter `question`.
            memory : type
                Description of parameter `memory`.

            Returns
            -------
            type
                Description of returned object.

            """

            f_i = [
                fact * question,
                fact * memory,
                K.abs(
                    fact - question),
                K.abs(
                    fact - memory)]
            g_t_i = self.l_1(K.concatenate(f_i, axis=1))
            g_t_i = self.l_2(g_t_i)
            return g_t_i

        facts = inputs[0]
        question = inputs[1]
        memory = K.identity(question)   # Initialize memory to the question
        fact_list = tf.unstack(facts, axis=1)

        for step in range(self.memory_steps):

            # Adapted from
            # https://github.com/barronalex/Dynamic-Memory-Networks-in-TensorFlow/
            attentions = [tf.squeeze(
                compute_attention(fact, question, memory), axis=1)
                for i, fact in enumerate(fact_list)]
            attentions = tf.stack(attentions)
            attentions = tf.transpose(attentions)
            attentions = tf.nn.softmax(attentions)
            attentions = tf.expand_dims(attentions, axis=-1)

            episode = K.concatenate([facts, attentions], axis=2)
            # Last state. Correct? Maybe not
            episode = self.episode_GRU(episode)

            memory = self.memory_net(K.concatenate(
                [memory, episode, question], axis=1))

        return K.concatenate([memory, question], axis=1)
