from keras.layers import Bidirectional, Dense, Concatenate, Multiply, Subtract, TimeDistributed,Dropout
from keras.engine.topology import Layer
from keras.layers.recurrent import GRU
from keras import backend as K
import tensorflow as tf
from keras.utils.generic_utils import has_arg
from attention_cells import SoftAttnGRU
from keras.engine import InputSpec


class EpisodicMemoryModule(Layer):

    def __init__(self, units, memory_steps, batch_size=32, dropout=0.0,  **kwargs):
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

        self.memory_steps = 3
        self.name = "episodic_memory_module"
        self._input_map = {}
        self.supports_masking = True
        self.units = units

        # attention net.
        self.l_1 = Dense(units=50, batch_size=batch_size, activation = 'tanh')
        self.l_2 = Dense(units=1, batch_size=batch_size, activation=None)

        # Episode net
        self.episode_GRU = SoftAttnGRU(units=units, return_sequences=False, batch_size=batch_size, dropout=dropout)

        # Memory generating net. TODO Check output size.
        self.memory_net = Dense(units=units, activation='relu')

        super(EpisodicMemoryModule, self).__init__()


    def compute_output_shape(self, input_shape):

        out_shape = list(input_shape[0])
        out_shape[-1] = self.units
        q_shape = list(input_shape[1])
        q_shape[-1] = self.units
        return tuple(q_shape)

    def build(self, input_shape):
        super(EpisodicMemoryModule, self).build(input_shape)

    def call(self, inputs):
        # TODO: Add Dropout and BatchNorm.

        def compute_attention(fact, question, memory):

            f_i = [fact * question, fact * memory, K.abs(fact - question), K.abs(fact - memory)]
            g_t_i = self.l_1(K.concatenate(f_i))
            g_t_i = self.l_2(g_t_i)

            return g_t_i

        facts = inputs[0]
        question = inputs[1]

        memory = K.identity(question)   # Initialize memory to the question

        for step in range(self.memory_steps):

            # See https://github.com/barronalex/Dynamic-Memory-Networks-in-TensorFlow/
            fact_list = tf.unstack(facts, axis=1)

            attentions = [tf.squeeze(
                    compute_attention(fact, question, memory), axis=1)
                    for i, fact in enumerate(fact_list)]
            attentions = tf.transpose(tf.stack(attentions))
            attentions = tf.nn.softmax(attentions)
            attentions = tf.expand_dims(attentions, axis=-1)

            episode = self.episode_GRU([facts, attentions])[:,-1] # Last state. Correct? Maybe not.
            memory = self.memory_net(K.concatenate([memory, episode, question], axis=1))

        return memory
