
import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import (Bidirectional, Dense, Dropout)
from attention_cells import SoftAttnGRU

from keras import regularizers
class EpisodicMemoryModule(Layer):

    def __init__(self, units, memory_steps, emb_dim=100, batch_size=32, dropout=0.0,  **kwargs):
        """
        The episodic memory consists of two nested networks + other details.
        The inner network is used to generate an episode, based on the incoming
        facts F_t that are sent from the input module.
        The outer network applies the attention to the facts to update the memory
        with the new episodes.

        """
        # TODO Readd attention type once this whole thing is solved
        # Attention parameters
        # TODO: Change units to EMB_DIM
        # TODO: Dropout

        self.memory_steps = memory_steps
        self.name = "episodic_memory_module"
        self._input_map = {}
        self.supports_masking = True
        self.units = units

        # attention net.
        self.l_1 = Dense(units=emb_dim, batch_size=batch_size, activation = 'tanh',
                         kernel_regularizer=regularizers.l2(0.01))

        self.l_2 = Dense(units=1, batch_size=batch_size, activation=None,
                         kernel_regularizer=regularizers.l2(0.01))

        # Episode net
        self.episode_GRU = SoftAttnGRU(units=units, return_sequences=False, batch_size=batch_size, dropout=dropout,
                                       kernel_regularizer=regularizers.l2(0.01),
                                       recurrent_regularizer=regularizers.l2(0.01))

        # Memory generating net.
        self.memory_net = Dense(units=units, activation='relu',
                                kernel_regularizer=regularizers.l2(0.01))

        super(EpisodicMemoryModule, self).__init__()


    def compute_output_shape(self, input_shape):

        q_shape = list(input_shape[1])
        q_shape[-1] = self.units*2
        return tuple(q_shape)

    def build(self, input_shape):
        super(EpisodicMemoryModule, self).build(input_shape)

    def call(self, inputs):
        # TODO: Add Dropout and BatchNorm.

        def compute_attention(fact, question, memory):

            f_i = [fact * question, fact * memory, K.abs(fact - question), K.abs(fact - memory)]
            g_t_i = self.l_1(K.concatenate(f_i, axis=1))
            g_t_i = self.l_2(g_t_i)
            return g_t_i

        facts = inputs[0]
        question = inputs[1]
        memory = K.identity(question)   # Initialize memory to the question
        fact_list = tf.unstack(facts, axis=1)

        for step in range(self.memory_steps):

            # See https://github.com/barronalex/Dynamic-Memory-Networks-in-TensorFlow/
            attentions = [tf.squeeze(
                    compute_attention(fact, question, memory), axis=1)
                    for i, fact in enumerate(fact_list)]
            attentions = tf.transpose(tf.stack(attentions))
            attentions = tf.nn.softmax(attentions)
            attentions = tf.expand_dims(attentions, axis=-1)

            episode = K.concatenate([facts, attentions], axis=2)
            episode = self.episode_GRU(episode) # Last state. Correct? Maybe not
            memory = self.memory_net(K.concatenate([memory, episode, question], axis=1))
            
        return K.concatenate([memory, question], axis=1)
