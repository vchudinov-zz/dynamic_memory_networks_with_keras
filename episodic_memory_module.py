from keras.layers import Bidirectional, Dense, Concatenate, Multiply, Subtract, TimeDistributed,Dropout
from keras.engine.topology import Layer
from keras.layers.recurrent import GRU
from keras import backend as K
import tensorflow as tf
from keras.utils.generic_utils import has_arg
from attention_cells import SoftAttnGRU
from keras.engine import InputSpec


class AttentionGate(Layer):
    """
    A horrible bastardization of TimeDistributed
    """
    # TODO Fix training mode.
    def __init__(self, units, batch_size=32, dropout=0.0):
        super(AttentionGate, self).__init__()
        self.concat = Concatenate()
        self.sub = Subtract()
        self._input_map = {}
        self.units = units
        self.supports_masking = True

        # Attention parameters
        # TODO: Change units to EMB_DIM
        # TODO: Dropout
        self.l_1 = Dense(units=50, batch_size=batch_size, activation = 'tanh')
    #    self.d_0 = Dropout(rate=dropout)
        self.l_2 = Dense(units=1, batch_size=batch_size, activation='softmax')
    #    self.d_1 = Dropout(rate=dropout)

    def build(self, input_shape):

        super(AttentionGate, self).build(input_shape)

    def compute_output_shape(self, input_shape):

        s_0 = list(input_shape[0]) # Input
        s_0[-1] = 1 # Attention

        return tuple(s_0)

    def call(self, inputs, training=None, mask=None):
        """
        Adapted from TimeDistributed
        """
        facts = inputs[0]
        question = inputs[1]
        memory = inputs[2]

        def compute_attention(fact, question, memory):

            f_i = [fact * question, fact * memory, K.abs(fact - question), K.abs(fact - memory)]
            z_t = self.concat(f_i)
        #    d_0 = self.d_0(z_t)
            g_t = self.l_1(z_t)
        #    d_1  = self.d_1(g_t)
            o = self.l_2(g_t)

            return o

        kwargs = {}
        if has_arg(self.call, 'training'):
            kwargs['training'] = training
        uses_learning_phase = False

        input_shape = tuple(K.int_shape(facts))

        if not input_shape[0]:
            raise ValueError("Missing batch size shape. Arbitrary batch shape is currently not supported. ")
            # batch size matters, use rnn-based implementation
        def step(inputs, states):

            fact = inputs
            global uses_learning_phase
            output = compute_attention(fact, question, memory)

            if hasattr(self, '_uses_learning_phase'):
                uses_learning_phase = (self._uses_learning_phase or
                                       uses_learning_phase)
            return output, []

        last_output, outputs, new_states = K.rnn(step, inputs=facts,
                              initial_states=[None],
                              input_length=input_shape[1],
                              unroll=True)
        y = outputs

        # Apply activity regularizer if any:
        if (hasattr(self, 'activity_regularizer') and self.activity_regularizer is not None):
            regularization_loss = self.layer.activity_regularizer(y)
            self.add_loss(regularization_loss, inputs)

        if uses_learning_phase:
            y._uses_learning_phase = True
        timesteps = input_shape[1]
        new_time_steps = list(y.get_shape())
        new_time_steps[1] = timesteps
        y.set_shape(new_time_steps)

        return y

class EpisodicMemoryModule(Layer):

    def __init__(self, units, memory_steps, batch_size=32, dropout=0.0, memory_type='RELU',  **kwargs):
        """
        The episodic memory consists of two nested networks + other details.
        The inner network is used to generate an episode, based on the incoming
        facts F_t that are sent from the input module.
        The outer network applies the attention to the facts to update the memory
        with the new episodes.

        """
        # TODO Readd attention type once this whole thing is solved

        self.memories = []
        self.memory_type = memory_type
        self.memory_steps = memory_steps
        self.name = "episodic_memory_module"
        self.concat = Concatenate()

        self.units = units
        #if memory_type == 'GRU':
        #self.memory_net = GRU(units=units, return_sequences=True, stateful=True)
        #elif memory_type == 'RELU':
        self.memory_net = Dense(units=units, activation='relu')

        self.attention_gate = AttentionGate(50, batch_size=batch_size, dropout=dropout)
        self.attention_GRU = SoftAttnGRU(units=units, return_sequences=False, batch_size=batch_size, dropout=dropout)
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
        # TODO: Determine if I need transposes.

        facts = inputs[0]
        question = inputs[1]
        # TODO: Add Dropout and BatchNorm.
        # Initialize memory to the question
        memory = K.identity(question)

        for step in range(self.memory_steps):
            # iteratively update memory
            if self.memory_type == 'GRU':
                memory = self.memory_net(self.generate_episode(memory), memory)[0]
            elif self.memory_type == 'RELU':
                attentions = self.attention_gate([facts, question, memory])
                attentions = K.softmax(attentions) # TODO verify axis
                episode = self.attention_GRU([facts, attentions])
                conc = K.concatenate([memory, episode[:,-1], question], axis=1)
                memory = self.memory_net(conc)

        return memory
