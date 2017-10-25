from keras.layers import Bidirectional, Dense, Concatenate, Multiply, Subtract
from keras.engine.topology import Layer
from keras.layers.recurrent import GRU
from keras import backend as K
import tensorflow as tf
from attention_cells import SoftAttnGRU

class EpisodicMemoryModule(Layer):
    def __init__(self, attn_units, attention_type, memory_units, memory_type, memory_steps, **kwargs):
        """
        The episodic memory consists of two nested networks + other details.
        The inner network is used to generate an episode, based on the incoming
        facts F_t that are sent from the input module.
        The outer network applies the attention to the facts to update the memory
        with the new episodes.

        """
        self.memories = []
        self.memory_type = memory_type
        self.memory_steps = memory_steps
        self.attention_type = attention_type

        self.build(attn_units, attention_type, memory_units, memory_type, )
        super(EpisodicMemoryModule, self).__init__(**kwargs)

    def build(self,  attn_units, memory_units, attention_type="soft", memory_type='GRU'):

        # Memory parameters for attention and episodes
        if memory_type == 'GRU':
            self.memory_net = GRU(units=memory_units)
        elif memory_type == 'RELU':
            self.memory_net = Dense(units=memory_units, activation='relu')

        if attention_type == 'soft':
            self.attention_GRU = SoftAttnGRU(units=attn_units)
        elif attention_type == 'gate':
            raise NotImplementedError
        self.concat = Concatenate()
        self.multiply = Multiply()
        self.sub = Subtract()

        # Attention parameters
        self.l_1 = Dense(units=attn_units, input_shape=(4*attn_units,), activation = 'tanh')
        self.l_2 = Dense(units=1, input_shape=(attn_units,), activation='softmax')

        self.built = True


    def generate_episode(self, facts, question, memory):
        facts = tf.unstack(facts, axis=1)
        episode = K.zeros(shape=facts[0].get_shape())


        #print(episode.shape)
        #print(len(facts))
        #raise SystemExit
        # TODO: Consider stacking these, instead of running with a loop.

        for f_i in facts:
            # Attention! Attention! Attention!
            z_t_0 = self.multiply([f_i, question, memory])
            z_t_1 = K.abs(self.sub([f_i, question]))
            z_t_2 = K.abs(self.sub([f_i, memory]))
            z_t_i = self.concat([z_t_0, z_t_1, z_t_2], axis=1)
            g_t_i = self.l_1(z_t_i)
            g_t_i = self.l_2(g_t_i)
            episode = self.attention_GRU(inputs=f_i, state=episode, attn_gate=g_t_i)

        return episode


    def __call__(self, inputs):
        # TODO: Determine if I need transposes.

        facts = inputs[0]
        question = inputs[1]
        # TODO: Add Dropout and BatchNorm.
        self.memories = []
        # Initialize memory to the question
        memory = K.identity(question)
        self.memories.append(memory)
        for step in range(self.memory_steps):
            # iteratively update memory
            if self.memory_type == 'GRU':
                memory = self.memory_net.call(self.generate_episode(memory), memory)[0]
            elif self.memory_type == 'RELU':
                episode = self.generate_episode(facts, question, memory)
                conc = self.concat([memory, episode, question])
                memory = self.memory_net.call(conc)
                #self.memories.append(memory) # Not sure if this will work
        return memory
