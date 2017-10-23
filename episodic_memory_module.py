from keras.layers import Bidirectional, Dense, Concatenate
from keras.engine.topology import Layer
from keras.layers.recurrent import GRU

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
        self.memory_steps
        self.attention_type = attention_type


        self.output_dim = output_dim
        self.build(attn_units, attention_type, memory_units, memory_type)
        super(EpisodicMemoryModule, self).__init__(**kwargs)

    def build(self, attn_units, memory_units,attention_type="soft", memory_type='GRU'):

        # Memory parameters for attention and episodes
        if memory_type == 'GRU':
            self.memory_net = GRU(units=memory_size)
        elif memory_type == 'RELU':
            self.memory_net = Dense(units=memory_size, activation='relu')

        if attention_type == 'soft':
            self.attention_GRU = SoftAtnnGRU(units=attn_units)
        elif attention_type == 'gate':
            raise NotImplementedError
        self.concat = Concatenate
        # Attention parameters
        self.W1 = self.add_weight(shape=(attn_units, 4*attn_units),
                                  name='g_W_1',
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint,
                                  trainable=True)

        self.b1 = self.add_weight(shape=self.attn_units,
                                name='G_bias_1',
                                initializer=self.bias_initializer,
                                regularizer=self.bias_regularizer,
                                constraint=self.bias_constraint,
                                trainable=True)


        self.W2 = self.add_weight(shape=(1, self.attn_units), # I think.
                                  name='g_W_2',
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint,
                                  trainable=True)

        self.b2 = self.add_weight(shape=(1,1),
                                name='G_bias_2',
                                initializer=self.bias_initializer,
                                regularizer=self.bias_regularizer,
                                constraint=self.bias_constraint,
                                trainable=True)
        self.built = True


    def generate_episode(self, facts, question, memory):
        episode = K.zeros(shape=facts[0].get_shape())
        # TODO: Consider stacking these, instead of running with a loop.

        for f_i in facts:
            # Attention! Attention! Attention!
            z_t_0 = K.multiply([f_i, question, memory])
            z_t_1 = K.abs(K.substract([f_i, question]))
            z_t_2 = K.abs(K.substract([f_i, memory]))
            z_t_i = K.concatenate([z_t_0, z_t_1, z_t_2])
            g_t_i = K.softmax(self.W2 * K.tanh( self.W1 * z_t_i + self.b1) + self.b2)

            episode = self.attention_GRU.call(inputs=f_i, state=episode, attn_gate=g_t_i)[0]

        return episode


    def call(self, inputs):
        facts = inputs[0]
        question = inputs[1]
        # TODO: Add Dropout and BatchNorm.
        self.memories = []
        # Initialize memory to the question
        memory = K.identity(question)
        self.memories.append(memory)
        for step in self.memory_steps:
            # iteratively update memory
            if self.memory_type == 'GRU':
                memory = self.memory_net.call(self.generate_episode(memory), memory)[0]
            elif self.memory_type == 'RELU':
                episode = self.generate_episode(facts, question, memory)
                conc = self.concat([memory, episode, question])
                memory = self.memory_net.call(conc)
                #self.memories.append(memory) # Not sure if this will work
        return memory
